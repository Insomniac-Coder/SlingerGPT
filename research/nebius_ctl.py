"""SlingerGPT 492M: the rented server and its data disk on Nebius (TRAINING_PLAN.md stage A.5).

    python research/nebius_ctl.py check                      sign-in, GPU quotas, B200 platform and presets (free)
    python research/nebius_ctl.py prices                     today's spot price per GPU-hour, by platform (free)
    python research/nebius_ctl.py status                     what exists now, and its state (free)
    python research/nebius_ctl.py price --max 3.000 --yes    the most to pay per GPU-hour for spot (a pricing policy)
    (price and create-server take --gpu b200 (default) or --gpu h200)
    python research/nebius_ctl.py create-disk --yes          the 1 TB data disk (network SSD; owner, 18 November)
    python research/nebius_ctl.py create-server --yes        8x B200 (or --gpu h200) spot server, data disk at /data
    python research/nebius_ctl.py stop --yes | start --yes | delete-server --yes | delete-disk --yes
    python research/nebius_ctl.py ssh                        the command to log in

Everything that costs money needs --yes, and the session asks the owner first (TRAINING_PLAN.md section 1).
The server is disposable; the data disk holds everything that matters. If this sandbox loses its SSH key, run
create-server again: a new key is made and the new server gets the same data disk.

Credentials come from the environment (the cloud session's settings): NEBIUS_PRIVATE_KEY_B64 (the private key, as
one base64 line or as the PEM text itself), NEBIUS_KEY_ID, NEBIUS_SERVICE_ACCOUNT_ID, NEBIUS_PROJECT_ID, NEBIUS_REGION.
The key is written to a file only this user can read and is never printed.
"""
import argparse, asyncio, base64, json, os, subprocess, sys
from pathlib import Path

NAME = "slingergpt-492m"
DISK = "slingergpt-492m-data"
BOOT = "slingergpt-492m-boot"
POLICY = "slingergpt-492m-spot"
GPUS = {   # --gpu: the platform and its 8-GPU preset (checked against the platform's presets by `check`)
    "b200": ("gpu-b200-sxm", "8gpu-160vcpu-1792gb"),
    "h200": ("gpu-h200-sxm", "8gpu-128vcpu-1600gb"),   # the owner already has 8 of these in us-central1 (18 November)
}
PLATFORM, PRESET = GPUS["b200"]
IMAGE_FAMILY = "ubuntu24.04-cuda12"
SSH_KEY = Path.home() / ".ssh" / "slingergpt_492m"
KEY_FILE = Path.home() / ".nebius" / "slingergpt_sa_key.pem"


def credentials():
    need = ["NEBIUS_PRIVATE_KEY_B64", "NEBIUS_KEY_ID", "NEBIUS_SERVICE_ACCOUNT_ID", "NEBIUS_PROJECT_ID"]
    missing = [n for n in need if not os.environ.get(n)]
    if missing:
        sys.exit(f"missing in the environment settings: {', '.join(missing)}")
    raw = os.environ["NEBIUS_PRIVATE_KEY_B64"].strip()
    pem = raw if raw.startswith("-----BEGIN") else base64.b64decode(raw).decode()
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(pem if pem.endswith("\n") else pem + "\n")
    os.chmod(KEY_FILE, 0o600)
    return os.environ["NEBIUS_PROJECT_ID"]


def sdk():
    from nebius.sdk import SDK
    return SDK(service_account_private_key_file_name=str(KEY_FILE),
               service_account_public_key_id=os.environ["NEBIUS_KEY_ID"],
               service_account_id=os.environ["NEBIUS_SERVICE_ACCOUNT_ID"])


async def find(service, project, name):
    """The resource called `name` in the project, or None."""
    from nebius.aio.service_error import RequestError
    from nebius.api.nebius.common.v1 import GetByNameRequest
    try:
        return await service.get_by_name(GetByNameRequest(parent_id=project, name=name))
    except RequestError as e:
        if "NOT_FOUND" in str(e) or "not found" in str(e).lower():
            return None
        raise


def yes_or_stop(args, what):
    if not args.yes:
        sys.exit(f"Not done: {what} costs money. Ask the owner, then run again with --yes.")


async def check(s, project, args):
    from nebius.api.nebius.iam.v1 import ProjectServiceClient, GetProjectRequest
    from nebius.api.nebius.quotas.v1 import QuotaAllowanceServiceClient, ListQuotaAllowancesRequest
    from nebius.api.nebius.compute.v1 import PlatformServiceClient, ListPlatformsRequest
    p = await ProjectServiceClient(s).get(GetProjectRequest(id=project))
    tenant, region = p.metadata.parent_id, p.status.region if hasattr(p.status, "region") else ""
    print(f"signed in; project {project} ({p.metadata.name}) in tenant {tenant}, region {region or os.environ.get('NEBIUS_REGION')}")
    rows, token = [], ""
    while True:
        r = await QuotaAllowanceServiceClient(s).list(ListQuotaAllowancesRequest(parent_id=tenant, page_size=1000, page_token=token))
        rows += list(r.items)
        token = r.next_page_token
        if not token:
            break
    gpu = [q for q in rows if "gpu" in q.metadata.name.lower()]
    print(f"\nGPU quotas ({len(gpu)} of {len(rows)} quotas):")
    for q in sorted(gpu, key=lambda q: (q.metadata.name, q.spec.region)):
        print(f"  {q.metadata.name:45s} {q.spec.region:14s} limit {q.spec.limit!s:>5}  in use {q.status.usage!s:>5}  "
              f"state {q.status.state.name}")
    platforms = await PlatformServiceClient(s).list(ListPlatformsRequest(parent_id=project, page_size=100))
    print("\nGPU platforms and their presets:")
    for pl in platforms.items:
        if "gpu" not in pl.metadata.name:
            continue
        presets = [pr.name for pr in pl.spec.presets]
        print(f"  {pl.metadata.name}: {', '.join(presets)}")
    for gpu, (platform, preset) in GPUS.items():
        found = [pl for pl in platforms.items if pl.metadata.name == platform]
        if not found:
            print(f"WARNING: platform {platform} is not offered to this project")
        elif preset not in [pr.name for pr in found[0].spec.presets]:
            print(f"WARNING: preset {preset} is not one of {platform}'s presets; change GPUS in this file")
        else:
            print(f"--gpu {gpu}: {platform} with {preset} is offered to this project")


async def prices(s, project, args):
    """Today's spot market price per GPU-hour for each GPU platform (free)."""
    from nebius.api.nebius.billing.v1alpha1 import SpotPriceServiceClient, ListSpotPricesRequest
    items, token = [], ""
    while True:
        r = await SpotPriceServiceClient(s).list(ListSpotPricesRequest(page_size=100, page_token=token))
        items += list(r.items)
        token = r.next_page_token
        if not token:
            break
    region = os.environ.get("NEBIUS_REGION", "us-central1")
    print(f"spot prices (taken {r.snapshot_time}):")
    for it in sorted(items, key=lambda i: (i.region != region, i.region, i.compute_instance.platform)):
        if it.compute_instance.platform.startswith("gpu"):
            print(f"  {it.region:12s} {it.compute_instance.platform:16s} ${float(it.price.value):.2f} per {it.price.unit_title}"
                  f"  (8 GPUs: ${8 * float(it.price.value):.2f} an hour)")


async def status(s, project, args):
    from nebius.api.nebius.compute.v1 import (InstanceServiceClient, ListInstancesRequest, DiskServiceClient,
                                              ListDisksRequest)
    inst = await InstanceServiceClient(s).list(ListInstancesRequest(parent_id=project, page_size=100))
    disks = await DiskServiceClient(s).list(ListDisksRequest(parent_id=project, page_size=100))
    print("servers:" if inst.items else "servers: none")
    for i in inst.items:
        ips = [n.public_ip_address.address for n in i.status.network_interfaces if n.public_ip_address]
        spot = "spot" if i.spec.preemptible else "regular"
        print(f"  {i.metadata.name}: {i.status.state.name} ({spot}, {i.spec.resources.platform} {i.spec.resources.preset}) "
              f"public address {', '.join(ips) or 'none'}")
    print("disks:" if disks.items else "disks: none")
    for d in disks.items:
        print(f"  {d.metadata.name}: {d.status.state.name}, {d.spec.size_gibibytes or d.status.size_bytes // 2**30} GiB "
              f"{d.spec.type.name}")


async def price(s, project, args):
    """A billing pricing policy: the most to pay per GPU-hour on spot. Above it the server is stopped."""
    yes_or_stop(args, "setting a spot price limit")
    from nebius.api.nebius.billing.v1 import (PricingPolicyServiceClient, PricingPolicy, PricingPolicySpec,
                                              ComputeInstanceSpec, ComputeInstanceSpecV1, PricingMethod, MaxPriceV1,
                                              CreatePricingPolicyRequest,
                                              UpdatePricingPolicyRequest)
    from nebius.api.nebius.common.v1 import ResourceMetadata
    svc = PricingPolicyServiceClient(s)
    spec = PricingPolicySpec(compute_instance_spec=ComputeInstanceSpec(v1=ComputeInstanceSpecV1(platform=PLATFORM)),
                             pricing=PricingMethod(max_price_v1=MaxPriceV1(max_price=f"{args.max:.3f}")))
    old = await find(svc, project, POLICY)
    if old:
        op = await svc.update(UpdatePricingPolicyRequest(metadata=old.metadata, spec=spec))
    else:
        op = await svc.create(CreatePricingPolicyRequest(metadata=ResourceMetadata(parent_id=project, name=POLICY), spec=spec))
    await op.wait()
    print(f"spot price limit set: at most ${args.max:.3f} per GPU-hour (${8 * args.max:.2f} an hour for 8 GPUs)")


async def create_disk(s, project, args):
    yes_or_stop(args, "a data disk")
    from nebius.api.nebius.compute.v1 import DiskServiceClient, CreateDiskRequest, DiskSpec
    from nebius.api.nebius.common.v1 import ResourceMetadata
    svc = DiskServiceClient(s)
    if await find(svc, project, DISK):
        print(f"{DISK} already exists")
        return
    op = await svc.create(CreateDiskRequest(metadata=ResourceMetadata(parent_id=project, name=DISK),
                                            spec=DiskSpec(size_gibibytes=args.gb, type=DiskSpec.DiskType.NETWORK_SSD)))
    await op.wait()
    print(f"created {DISK}: {args.gb} GiB network SSD")


def ssh_key():
    if not SSH_KEY.exists():
        SSH_KEY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "slingergpt-492m", "-f", str(SSH_KEY)], check=True)
    return SSH_KEY.with_suffix(".pub").read_text().strip()


CLOUD_INIT = """#cloud-config
users:
  - name: slinger
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - {key}
runcmd:
  - [bash, -c, "dev=/dev/disk/by-id/virtio-data; for i in $(seq 60); do [ -e $dev ] && break; sleep 2; done; blkid $dev || mkfs.ext4 -q -L data $dev; mkdir -p /data; grep -q ' /data ' /etc/fstab || echo \\"$dev /data ext4 defaults,nofail 0 2\\" >> /etc/fstab; mount -a; chown slinger:slinger /data"]
  - [bash, -c, "apt-get update -q && apt-get install -y -q tmux htop"]
"""


async def create_server(s, project, args):
    yes_or_stop(args, f"an 8x {args.gpu.upper()} spot server")
    from nebius.api.nebius.compute.v1 import (
        InstanceServiceClient, CreateInstanceRequest, InstanceSpec, ResourcesSpec, AttachedDiskSpec, ExistingDisk,
        NetworkInterfaceSpec, IPAddress, PublicIPAddress, PreemptibleSpec, SpotPricingPolicySpec, FollowsSpotPriceSpec,
        DiskServiceClient, CreateDiskRequest, DiskSpec, SourceImageFamily, )
    from nebius.api.nebius.vpc.v1 import SubnetServiceClient, ListSubnetsRequest
    from nebius.api.nebius.billing.v1 import PricingPolicyServiceClient, GetByNameRequest
    from nebius.api.nebius.common.v1 import ResourceMetadata
    inst, disks = InstanceServiceClient(s), DiskServiceClient(s)
    if await find(inst, project, NAME):
        sys.exit(f"{NAME} already exists: use start, or delete-server first")
    data = await find(disks, project, DISK)
    if not data:
        sys.exit(f"no data disk yet: run create-disk first")
    boot = await find(disks, project, BOOT)
    if boot is None:
        op = await disks.create(CreateDiskRequest(
            metadata=ResourceMetadata(parent_id=project, name=BOOT),
            spec=DiskSpec(size_gibibytes=200, type=DiskSpec.DiskType.NETWORK_SSD,
                          source_image_family=SourceImageFamily(image_family=IMAGE_FAMILY))))
        await op.wait()
        boot = await find(disks, project, BOOT)
    subnets = await SubnetServiceClient(s).list(ListSubnetsRequest(parent_id=project, page_size=10))
    if not subnets.items:
        sys.exit("the project has no subnet; make the default network in the console first")
    policy = await find(PricingPolicyServiceClient(s), project, POLICY)
    pricing = dict(spot_pricing_policy=SpotPricingPolicySpec(id=policy.metadata.id)) if policy else \
        dict(follows_spot_price=FollowsSpotPriceSpec())
    if not policy:
        print("note: no spot price limit set (`price`), so the server pays the market price")
    spec = InstanceSpec(
        resources=ResourcesSpec(platform=PLATFORM, preset=PRESET),
        boot_disk=AttachedDiskSpec(attach_mode=AttachedDiskSpec.AttachMode.READ_WRITE, existing_disk=ExistingDisk(id=boot.metadata.id)),
        secondary_disks=[AttachedDiskSpec(attach_mode=AttachedDiskSpec.AttachMode.READ_WRITE,
                                          existing_disk=ExistingDisk(id=data.metadata.id), device_id="data")],
        network_interfaces=[NetworkInterfaceSpec(subnet_id=subnets.items[0].metadata.id, name="eth0",
                                                 ip_address=IPAddress(), public_ip_address=PublicIPAddress())],
        cloud_init_user_data=CLOUD_INIT.format(key=ssh_key()),
        preemptible=PreemptibleSpec(on_preemption=PreemptibleSpec.PreemptionPolicy.STOP), **pricing)
    op = await inst.create(CreateInstanceRequest(metadata=ResourceMetadata(parent_id=project, name=NAME), spec=spec))
    await op.wait()
    print(f"created {NAME}; it starts in a few minutes. `status` shows its address, `ssh` the login command.")


async def power(s, project, args, on):
    yes_or_stop(args, "starting the server" if on else "stopping the server")
    from nebius.api.nebius.compute.v1 import (InstanceServiceClient, StartInstanceRequest,
                                              StopInstanceRequest)
    svc = InstanceServiceClient(s)
    i = await find(svc, project, NAME)
    if not i:
        sys.exit(f"{NAME} does not exist")
    op = await (svc.start(StartInstanceRequest(id=i.metadata.id)) if on else svc.stop(StopInstanceRequest(id=i.metadata.id)))
    await op.wait()
    print(f"{NAME} {'started' if on else 'stopped (the data disk keeps everything; it still costs money each day)'}")


async def delete(s, project, args, what):
    yes_or_stop(args, f"deleting the {what}")
    from nebius.api.nebius.compute.v1 import (InstanceServiceClient, DeleteInstanceRequest,
                                              DiskServiceClient, DeleteDiskRequest)
    if what == "server":
        svc = InstanceServiceClient(s)
        i = await find(svc, project, NAME)
        if i:
            await (await svc.delete(DeleteInstanceRequest(id=i.metadata.id))).wait()
        names = [BOOT]
    else:
        names = [DISK]
    svc = DiskServiceClient(s)
    for n in names:
        d = await find(svc, project, n)
        if d:
            await (await svc.delete(DeleteDiskRequest(id=d.metadata.id))).wait()
    print(f"deleted the {what}" + (" (the data disk is kept)" if what == "server" else ""))


async def ssh(s, project, args):
    from nebius.api.nebius.compute.v1 import InstanceServiceClient
    i = await find(InstanceServiceClient(s), project, NAME)
    if not i:
        sys.exit(f"{NAME} does not exist")
    ips = [n.public_ip_address.address.split("/")[0] for n in i.status.network_interfaces if n.public_ip_address]
    print(f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=accept-new slinger@{ips[0] if ips else '?'}")


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["check", "prices", "status", "price", "create-disk", "create-server", "start", "stop",
                                    "delete-server", "delete-disk", "ssh"])
    p.add_argument("--yes", action="store_true", help="the owner said yes to what this costs")
    p.add_argument("--gb", type=int, default=1000, help="data disk size (owner: 1 TB is enough)")
    p.add_argument("--max", type=float, default=0.0,
                   help="price: the most to pay per GPU-hour in US dollars (default: the owner's limit for --gpu)")
    p.add_argument("--gpu", choices=sorted(GPUS), default="b200", help="which 8-GPU server (price and create-server)")
    args = p.parse_args()
    global PLATFORM, PRESET, POLICY
    PLATFORM, PRESET = GPUS[args.gpu]
    POLICY = f"slingergpt-492m-spot-{args.gpu}"
    args.max = args.max or {"b200": 3.00, "h200": 2.20}[args.gpu]   # owner, 18 November: the day's spot price
    project = credentials()
    s = sdk()
    try:
        await {"check": check, "prices": prices, "status": status, "price": price, "create-disk": create_disk, "create-server": create_server,
               "start": lambda *a: power(*a, on=True), "stop": lambda *a: power(*a, on=False),
               "delete-server": lambda *a: delete(*a, what="server"), "delete-disk": lambda *a: delete(*a, what="disk"),
               "ssh": ssh}[args.what](s, project, args)
    finally:
        await s.close()


if __name__ == "__main__":
    asyncio.run(main())
