"""CPU tests of train_492m.py (TRAINING_PLAN.md stage A.6), on made-up text with a tiny model and 3 processes (gloo).

    python research/test_train_492m.py [folder]
    python research/test_train_492m.py gpu         on the server: 8-bit build, flex mask, fused loss, attention speed

1. The loss falls, the copies of the model stay identical, and only process 0 writes saves and logs.
2. A run stopped after 25 updates and carried on reads exactly the same windows as one that never stopped. With one
   process it also ends with exactly the same weights. With several, the weights differ slightly: the processes add
   their gradients in an order that varies from run to run, and the rounding grows during training.
3. A real kill -TERM in the middle of training: it saves and stops within seconds, and carries on to the end.
4. Copies in storage (--bucket), against a local S3 stand-in (moto): latest.pt after a stop by SIGTERM, then every
   milestone, the final model and the full save once the run ends; a copy loads as a usable model.
Every schedule segment is crossed, including the long phase's document mask and the rotary base switch.
"""
import json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
TRAIN = [sys.executable, str(HERE / "train_492m.py")]
VOCAB, GPUS = 256, 3
COMMON = ["--device", "cpu", "--gpus", str(GPUS), "--size", "1m", "--no-fp8", "--no-compile", "--windows",
          "32,64,128,128,256", "--per-update", "1536", "--total-pieces", str(1536 * 60), "--micro-pieces", "768",
          "--score-every", "20", "--log-every", "5", "--save-minutes", "1000", "--trace", "--warm-updates", "5",
          "--lr", "0.003"]


def fake_text(folder):
    """Every source of the mix: 3 files of a learnable pattern (each source its own), plus held-out text."""
    mix = json.loads((HERE / "492m" / "mix_492m.json").read_text())
    names = sorted({k for shares in mix["phases"].values() for k in shares})
    manifest = {"pieces in the list": VOCAB, "end of text": 0, "sources": {}}
    for i, name in enumerate(names):
        rng = np.random.default_rng(i)
        (folder / name).mkdir(parents=True)
        files = []
        for f in range(3):
            n = int(rng.integers(20_000, 40_000)) + (4000 if f == 0 else 0)
            x = np.zeros(n, dtype=np.uint16)
            x[0] = 1
            step = 1 + i % 7
            for t in range(1, n):   # a counting pattern with some noise, and a document end every ~300 pieces
                x[t] = 0 if rng.random() < 1 / 300 else (1 + (x[t - 1] + step) % (VOCAB - 1) if rng.random() > 0.1
                                                         else rng.integers(1, VOCAB))
            if f == 0:   # held-out text: more of the same pattern, never trained on
                x[-4000:].tofile(folder / name / "val.bin")
                x, n = x[:-4000], n - 4000
            x.tofile(folder / name / f"{f:04d}.bin")
            files.append({"name": f"{f:04d}.bin", "pieces": n, "letters": 4 * n, "documents": n // 300})
        manifest["sources"][name] = {"files": files, "complete": True,
                                     "val": {"name": "val.bin", "pieces": 4000, "letters": 16000, "documents": 13}}
    (folder / "manifest.json").write_text(json.dumps(manifest))


def run(folder, name, *extra, wait=True, gpus=GPUS):
    common = COMMON[:]
    common[common.index("--gpus") + 1] = str(gpus)
    cmd = TRAIN + ["train", "--data", str(folder), "--name", name] + common + list(extra)
    if not wait:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    return out.stdout


def records(folder, name):
    return [json.loads(l) for l in (folder / "runs" / f"{name}.log").read_text().splitlines() if l.startswith("{")]


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


def weights(folder, name):
    """The full 32-bit weights of the copy saved at the end."""
    import torch
    last = max((folder / "runs").glob(f"{name}.*B.ckpt"), key=lambda p: float(p.name.split(".", 1)[1][:-6]))
    return torch.load(last, map_location="cpu", weights_only=False)["model"]


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_train_492m")
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    fake_text(folder)
    began = time.time()

    # 1. a whole run
    run(folder, "whole")
    log = records(folder, "whole")
    losses = [r["loss"] for r in log if "loss" in r]
    scores = [r for r in log if "held_out_wrongness_per_letter" in r]
    assert losses[-1] < 0.6 * losses[0], f"the loss did not fall enough: {losses}"
    assert all(r["copies_identical"] for r in scores), "the copies drifted apart"
    events = [r.get("event", "") for r in log]
    assert any("rotary base now 1,000,000" in e for e in events), "the rotary base did not switch"
    assert sum("milestone copy" in e for e in events) == 6, events
    starts = [r for r in log if r.get("event") == "start"]
    assert len(starts) == 1, "more than one process wrote to the log"
    files = sorted(p.name for p in (folder / "runs").iterdir() if p.name.startswith("whole."))
    print(f"1. loss {losses[0]:.3f} -> {losses[-1]:.3f}; held-out {scores[0]['held_out_wrongness_per_letter']} -> "
          f"{scores[-1]['held_out_wrongness_per_letter']}; copies identical at all {len(scores)} checks; files: {files}")

    # 2. stop after 25 updates, carry on, compare with the whole run
    run(folder, "resumed", "--stop-after", "25")
    run(folder, "resumed")
    for r in range(GPUS):
        a = (folder / "runs" / f"whole.windows.{r}.txt").read_text().splitlines()
        b = (folder / "runs" / f"resumed.windows.{r}.txt").read_text().splitlines()
        assert a == b, f"GPU {r} read different windows after the resume"
    wa, wb = weights(folder, "whole"), weights(folder, "resumed")
    diff = max((wa[k].float() - wb[k].float()).abs().max().item() for k in wa if wa[k].is_floating_point())
    # several processes add their gradients in an order that varies from run to run (even two runs that never stop
    # differ), and Adam enlarges rounding on near-zero gradients, by up to the rate (0.003) an update; so only a gross
    # difference fails here, and the exact check is the one-process run below
    assert diff < 0.1, f"the resumed run ended with very different weights (largest difference {diff})"
    run(folder, "one", gpus=1)
    run(folder, "one_resumed", "--stop-after", "25", gpus=1)
    run(folder, "one_resumed", gpus=1)
    w1, w2 = weights(folder, "one"), weights(folder, "one_resumed")
    same = all(torch_equal(w1[k], w2[k]) for k in w1)
    assert same, "with one process, the resumed run did not end with exactly the same weights"
    print(f"2. stop at 25 and carry on: the same {len(a)} windows on each of {GPUS} processes (largest weight "
          f"difference {diff:.1e}, from DDP's summing order); with 1 process exactly the same weights")

    # 3. a real SIGTERM in the middle
    p = run(folder, "killed", wait=False)
    logfile = folder / "runs" / "killed.log"
    while not (logfile.exists() and any('"update": 3' in l for l in logfile.read_text().splitlines())):
        assert p.poll() is None, p.stdout.read()
        time.sleep(0.5)
    sent = time.time()
    os.killpg(p.pid, signal.SIGTERM)   # the launcher and every process, as a shutting-down server does
    p.wait(timeout=120)
    took = time.time() - sent
    stopped = [r for r in records(folder, "killed") if r.get("event") == "stopped and saved"]
    assert p.returncode == 0 and stopped, p.stdout.read()
    run(folder, "killed")
    end = [r for r in records(folder, "killed") if r.get("event") == "done"]
    assert end, "the killed run did not finish after carrying on"
    print(f"3. SIGTERM at update {stopped[0]['update']}: saved in {stopped[0]['save_seconds']} s, all stopped "
          f"{took:.1f} s after the signal; carried on to the end")
    storage_test(folder)
    print(f"all trainer tests passed ({time.time() - began:.0f} s)")


def storage_test(folder):
    import boto3, socket, torch
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = subprocess.Popen([sys.executable, "-m", "moto.server", "-p", str(port)], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}"
        s3 = boto3.client("s3", endpoint_url=url, region_name="us-central1", aws_access_key_id="test",
                          aws_secret_access_key="test")
        for attempt in range(100):   # the stand-in takes a moment to start
            try:
                s3.create_bucket(Bucket="models", CreateBucketConfiguration={"LocationConstraint": "us-central1"})
                break
            except Exception:
                if attempt == 99:
                    raise
                time.sleep(0.2)
        keys = folder / "keys.json"
        keys.write_text(json.dumps({"aws_access_key_id": "test", "secret": "test"}))
        extra = ["--bucket", "models", "--bucket-endpoint", url, "--bucket-keys", str(keys)]
        p = run(folder, "stored", *extra, wait=False)
        logfile = folder / "runs" / "stored.log"
        while not (logfile.exists() and any('"update": 3' in l for l in logfile.read_text().splitlines())):
            assert p.poll() is None, p.stdout.read()
            time.sleep(0.5)
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=120)
        stop = [r for r in records(folder, "stored") if r.get("event") == "stopped and saved"][0]
        names = {o["Key"] for o in s3.list_objects_v2(Bucket="models").get("Contents", [])}
        assert stop.get("copied_to_storage") and "stored/latest.pt" in names, (stop, names)
        run(folder, "stored", *extra)
        names = {o["Key"] for o in s3.list_objects_v2(Bucket="models").get("Contents", [])}
        assert {"stored/latest.pt", "stored/tokenizer.json", "stored/stored.pt"} <= names or \
            {"stored/latest.pt", "stored/stored.pt"} <= names, names
        assert any(n.endswith("B.ckpt") for n in names) and sum(n.endswith("B.pt") for n in names) >= 1, names
        s3.download_file("models", "stored/latest.pt", str(folder / "from_storage.pt"))
        saved = torch.load(folder / "from_storage.pt", map_location="cpu", weights_only=False)
        sys.path.insert(0, str(HERE))
        import arch
        m = arch.build(device="cpu", **saved["cfg"])
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in saved["model"].items()})
        print(f"4. storage: latest.pt was there right after the SIGTERM stop (update {stop['update']}); at the end "
              f"{len(names)} files: {sorted(names)}; the downloaded copy loads (rotary base {saved['cfg']['rope_base']:g})")
    finally:
        server.terminate()


def gpu_checks():
    """On the rented server, before the test run (one GPU, about a minute): what the CPU tests cannot cover."""
    import torch, torch.nn.functional as F
    sys.path.insert(0, str(HERE))
    import arch, train_492m as t
    from fused_ce import fused_linear_ce
    torch.manual_seed(0)
    m = arch.build(**{**t.CFG, "fp8": True})
    n = sum(p.numel() for p in m.parameters())
    assert n == t.WEIGHTS, n
    print(f"492M built with 8-bit block layers: {n:,} weights")
    x = torch.randint(1, 16384, (2, 4096), device="cuda")
    x[:, 1000] = 0   # two documents in each window
    dense, flex = arch.document_mask(x, dense=True), arch.document_mask(x, _compile=True)
    m.eval()
    with torch.no_grad():
        a, b = m.features(x, mask=dense).float(), m.features(x, mask=flex).float()
    print(f"document mask: flex vs written-out mask, largest difference {(a - b).abs().max().item():.3g} (bf16)")
    m.train()
    h = m.features(x[:, :-1]).flatten(0, 1)
    y = x[:, 1:].flatten()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        plain = F.cross_entropy(m.out(h).float(), y)
    fused = fused_linear_ce(h, m.out.weight, y, m.out.bias)
    print(f"fused loss {fused.item():.5f} vs plain {plain.item():.5f}")
    assert abs(fused.item() - plain.item()) < 1e-3
    q = torch.randn(8, 20, 4096, 64, device="cuda", dtype=torch.bfloat16)
    for name, kind in [("cudnn", arch.SDPBackend.CUDNN_ATTENTION), ("flash", arch.SDPBackend.FLASH_ATTENTION),
                       ("efficient", arch.SDPBackend.EFFICIENT_ATTENTION)]:
        try:
            with arch.sdpa_kernel([kind]):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(20):
                    F.scaled_dot_product_attention(q, q, q, is_causal=True)
                torch.cuda.synchronize()
            print(f"attention {name}: {(time.time() - t0) / 20 * 1000:.2f} ms for 8 x 4,096 pieces")
        except Exception as e:
            print(f"attention {name}: not available ({type(e).__name__})")
    print("GPU checks passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        gpu_checks()
    else:
        main()
