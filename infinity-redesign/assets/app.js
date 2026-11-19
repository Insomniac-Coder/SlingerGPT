/* Infinity — redesign concept: shared header, footer, cart, product cards, notes. */
(function () {
  const C = window.CATALOG || [];
  const FREE_SHIP = 25; // suggested free-shipping threshold (to confirm)
  const byId = Object.fromEntries(C.map(p => [p.id, p]));
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const money = n => '$' + (Math.round(n * 100) / 100).toFixed(n % 1 ? 2 : 0);
  const img = id => `assets/img/${id}.jpg`;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const SINGULAR = {
    'Phone Charms': 'Phone charm', 'Wrist Keychains': 'Wrist keychain', 'Badge Reels': 'Badge reel', 'Bracelets': 'Bracelet',
    'Pens': 'Beaded pen', 'Bundles': 'Bundle', 'Lip Balm Keychains': 'Lip balm keychain', 'Mystery Bags': 'Mystery bag'
  };
  const VIBES = {
    holiday: { n: 'Holiday Drop', e: '❄' }, spooky: { n: 'Stay Spooky', e: '☠' }, scrubs: { n: 'Scrub Life', e: '✚' },
    sweet: { n: 'Sweet & Pretty', e: '♡' }, cause: { n: 'Wear It for a Cause', e: '🎗' }
  };

  const LOGO_MARK = `<svg viewBox="0 0 120 60" class="mark" aria-hidden="true"><path d="M106.0,30.0 L104.9,35.6 L102.0,40.4 L97.7,43.9 L92.7,45.8 L87.7,46.3 L82.8,45.6 L78.4,44.2 L74.4,42.1 L70.8,39.7 L67.5,37.1 L64.4,34.3 L61.4,31.4 L58.6,28.6 L55.6,25.7 L52.5,22.9 L49.2,20.3 L45.6,17.9 L41.6,15.8 L37.2,14.4 L32.3,13.7 L27.3,14.2 L22.3,16.1 L18.0,19.6 L15.1,24.4 L14.0,30.0 L15.1,35.6 L18.0,40.4 L22.3,43.9 L27.3,45.8 L32.3,46.3 L37.2,45.6 L41.6,44.2 L45.6,42.1 L49.2,39.7 L52.5,37.1 L55.6,34.3 L58.6,31.4 L61.4,28.6 L64.4,25.7 L67.5,22.9 L70.8,20.3 L74.4,17.9 L78.4,15.8 L82.8,14.4 L87.7,13.7 L92.7,14.2 L97.7,16.1 L102.0,19.6 L104.9,24.4 L106.0,30.0Z" fill="none" stroke="currentColor" stroke-width="1.4" opacity=".55"/><circle cx="106.0" cy="30.0" r="5.2" fill="var(--pink)" stroke="currentColor" stroke-width="1.2"/><circle cx="102.8" cy="39.5" r="4.3" fill="var(--teal)" stroke="currentColor" stroke-width="1.2"/><circle cx="94.7" cy="45.2" r="4.3" fill="var(--lilac)" stroke="currentColor" stroke-width="1.2"/><circle cx="84.8" cy="46.0" r="5.2" fill="var(--butter)" stroke="currentColor" stroke-width="1.2"/><circle cx="75.4" cy="42.7" r="4.3" fill="var(--pink)" stroke="currentColor" stroke-width="1.2"/><circle cx="67.2" cy="36.9" r="4.3" fill="var(--mint)" stroke="currentColor" stroke-width="1.2"/><circle cx="52.7" cy="23.0" r="4.3" fill="var(--teal)" stroke="currentColor" stroke-width="1.2"/><circle cx="44.5" cy="17.2" r="4.3" fill="var(--lilac)" stroke="currentColor" stroke-width="1.2"/><circle cx="35.0" cy="14.0" r="5.2" fill="var(--butter)" stroke="currentColor" stroke-width="1.2"/><circle cx="25.1" cy="14.9" r="4.3" fill="var(--pink)" stroke="currentColor" stroke-width="1.2"/><circle cx="17.1" cy="20.7" r="4.3" fill="var(--mint)" stroke="currentColor" stroke-width="1.2"/><circle cx="14.0" cy="30.0" r="5.2" fill="var(--pink)" stroke="currentColor" stroke-width="1.2"/><circle cx="17.2" cy="39.5" r="4.3" fill="var(--teal)" stroke="currentColor" stroke-width="1.2"/><circle cx="25.3" cy="45.2" r="4.3" fill="var(--lilac)" stroke="currentColor" stroke-width="1.2"/><circle cx="35.2" cy="46.0" r="5.2" fill="var(--butter)" stroke="currentColor" stroke-width="1.2"/><circle cx="44.6" cy="42.7" r="4.3" fill="var(--pink)" stroke="currentColor" stroke-width="1.2"/><circle cx="52.8" cy="36.9" r="4.3" fill="var(--mint)" stroke="currentColor" stroke-width="1.2"/><circle cx="67.3" cy="23.0" r="4.3" fill="var(--teal)" stroke="currentColor" stroke-width="1.2"/><circle cx="75.5" cy="17.2" r="4.3" fill="var(--lilac)" stroke="currentColor" stroke-width="1.2"/><circle cx="85.0" cy="14.0" r="5.2" fill="var(--butter)" stroke="currentColor" stroke-width="1.2"/><circle cx="94.9" cy="14.9" r="4.3" fill="var(--pink)" stroke="currentColor" stroke-width="1.2"/><circle cx="102.9" cy="20.7" r="4.3" fill="var(--mint)" stroke="currentColor" stroke-width="1.2"/><path d="M60,33.5 c-5,-4 -7,-9 -3.4,-10.6 c1.8,-.8 3,.4 3.4,1.6 c.4,-1.2 1.6,-2.4 3.4,-1.6 c3.6,1.6 1.6,6.6 -3.4,10.6z" fill="var(--pink-deep)" stroke="currentColor" stroke-width="1.1"/></svg>`;

  const ICON = {
    search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    bag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M5 8h14l-1 12H6L5 8Z"/><path d="M9 8V6a3 3 0 0 1 6 0v2"/></svg>',
    menu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h10"/></svg>',
    x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg>',
    heart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10Z"/></svg>',
    ig: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="5"/><circle cx="12" cy="12" r="4"/><circle cx="17.5" cy="6.5" r="1" fill="currentColor"/></svg>',
    tt: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3v11.5a3.5 3.5 0 1 1-3.5-3.5"/><path d="M14 3c.5 2.6 2.4 4.4 5 4.6"/></svg>',
    pin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="10" r="7"/><path d="M11 8.5 9 21M11.5 6.5c3-1 5 1 4 3.5-.8 2-3.5 2.2-4.6 1"/></svg>',
    mail: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="3"/><path d="m4 7 8 6 8-6"/></svg>'
  };

  /* ---------- storage (per-viewer convenience only) ---------- */
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode: keep in memory */ } }
  };
  let cart = store.get('inf-cart', []);

  /* ---------- header / footer ---------- */
  function header(page) {
    const nav = [['shop', 'shop.html', 'Shop all'], ['charms', 'shop.html?type=Phone%20Charms', 'Phone charms'],
      ['holiday', 'shop.html?vibe=holiday', 'Holiday drop'], ['custom', 'custom.html', 'Custom orders'], ['about', 'about.html', 'About & help']];
    const msgs = ['❄ The Holiday Drop is live — snowmen, reindeer & sweater-weather charms', 'Free shipping on orders $25+', 'Handmade one at a time, by one person',
      'Custom orders welcome — names, colors, team & unit charms'];
    const run = msgs.map(m => `<span>${esc(m)} <i></i></span>`).join('');
    return `
    <div class="announce" role="region" aria-label="Announcements"><div class="track">${run}${run}</div></div>
    <header class="site-header">
      <div class="wrap">
        <button class="icon-btn menu-btn" data-open="menu" aria-label="Open menu">${ICON.menu}</button>
        <a class="logo" href="home.html" aria-label="Infinity home">${LOGO_MARK}<span class="word"><b>infinity</b><small>phone charms &amp; handmade goodies</small></span></a>
        <nav class="nav" aria-label="Main">${nav.map(([k, h, t]) => `<a href="${h}"${k === page ? ' aria-current="page"' : ''}>${t}</a>`).join('')}</nav>
        <div class="hdr-tools">
          <button class="icon-btn" data-search aria-label="Search">${ICON.search}</button>
          <button class="icon-btn" data-open="cart" aria-label="Open bag">${ICON.bag}<span class="cart-count" data-cart-count></span></button>
        </div>
      </div>
      <div class="search-pop" id="searchPop"><div class="wrap">
        <label class="sr" for="q">Search the shop</label>
        <input id="q" type="search" placeholder="Search 100+ handmade pieces — try “nurse”, “ghost”, “snowman”…" autocomplete="off">
        <div class="search-results" id="searchResults"></div>
      </div></div>
    </header>
    <div class="drawer-scrim" data-close></div>
    <aside class="drawer left" id="menu" aria-label="Menu">
      <div class="drawer-hd"><a class="logo" href="home.html">${LOGO_MARK}<span class="word"><b>infinity</b></span></a><button class="icon-btn" data-close aria-label="Close">${ICON.x}</button></div>
      <div class="drawer-bd mnav">
        <a href="shop.html">Shop all <small>${C.length}</small></a>
        ${Object.keys(SINGULAR).map(t => { const n = C.filter(p => p.type === t).length; return `<a href="shop.html?type=${encodeURIComponent(t)}">${t} <small>${n}</small></a>`; }).join('')}
        <a href="shop.html?vibe=holiday">❄ Holiday drop</a>
        <a href="custom.html">Custom orders</a>
        <a href="about.html">About &amp; help</a>
      </div>
    </aside>
    <aside class="drawer right" id="cart" aria-label="Your bag">
      <div class="drawer-hd"><h3>Your bag</h3><button class="icon-btn" data-close aria-label="Close">${ICON.x}</button></div>
      <div class="drawer-bd" id="cartBody"></div>
      <div class="drawer-ft" id="cartFoot"></div>
    </aside>
    <div class="toast" id="toast" role="status" aria-live="polite"></div>`;
  }

  function footer() {
    return `
    <footer class="site-footer">
      <div class="wrap">
        <div class="ft-news" data-note="The old footer newsletter block has no storage connected, so sign-ups likely went nowhere. Connect it to Squarespace Email Campaigns or Mailchimp and give people a reason: first dibs on drops.">
          <div><span class="eyebrow" style="color:var(--mint)">The bead list</span><h2 style="margin-top:12px">Get <em>first dibs</em> on every drop.</h2></div>
          <div>
            <form onsubmit="event.preventDefault(); INF.toast('Mockup: this would add you to the list ✨')">
              <label class="sr" for="em">Email address</label>
              <input id="em" type="email" placeholder="you@email.com" required>
              <button class="btn pink" type="submit">Sign me up</button>
            </form>
            <p class="fine">One-of-one pieces sell out fast — subscribers hear about new drops a day early. No spam, ever.</p>
          </div>
        </div>
        <div class="ft-cols">
          <div class="ft-brand">
            <a class="logo" href="home.html">${LOGO_MARK}<span class="word"><b>infinity</b><small style="color:var(--mint)">phone charms &amp; handmade goodies</small></span></a>
            <p>Beaded phone charms, wrist keychains, badge reels &amp; bracelets — strung by hand, one at a time.</p>
            <div class="socials" data-note="No social links exist on the current site. Add Instagram/TikTok — bead-making videos are great content and drive sales."><a href="#" aria-label="Instagram">${ICON.ig}</a><a href="#" aria-label="TikTok">${ICON.tt}</a><a href="#" aria-label="Pinterest">${ICON.pin}</a><a href="mailto:infinityphonecharms@gmail.com" aria-label="Email">${ICON.mail}</a></div>
          </div>
          <div><h4>Shop</h4><ul>
            <li><a href="shop.html?type=Phone%20Charms">Phone charms</a></li><li><a href="shop.html?type=Wrist%20Keychains">Wrist keychains</a></li>
            <li><a href="shop.html?type=Badge%20Reels">Badge reels</a></li><li><a href="shop.html?type=Bracelets">Bracelets</a></li>
            <li><a href="shop.html?price=under10">Gifts under $10</a></li><li><a href="shop.html?type=Mystery%20Bags">Mystery bags</a></li></ul></div>
          <div><h4>Help</h4><ul>
            <li><a href="about.html#shipping">Shipping</a></li><li><a href="about.html#returns">Returns &amp; exchanges</a></li>
            <li><a href="about.html#fit">Will it fit my phone?</a></li><li><a href="about.html#care">Charm care</a></li><li><a href="about.html#contact">Contact</a></li></ul></div>
          <div><h4>Infinity</h4><ul>
            <li><a href="about.html">Meet the maker</a></li><li><a href="custom.html">Custom orders</a></li>
            <li><a href="custom.html#bulk">Group &amp; unit orders</a></li><li><a href="mailto:infinityphonecharms@gmail.com">infinityphonecharms@gmail.com</a></li></ul></div>
        </div>
        <div class="ft-base"><span>© 2026 Infinity Phone Charms · Handmade in the USA <span class="ph-mark" title="Placeholder — confirm location">✎</span></span><span class="pay"><span>VISA</span><span>MC</span><span>AMEX</span><span>PayPal</span><span>Apple Pay</span></span></div>
      </div>
    </footer>
    <div class="mock-ribbon"><a href="index.html">← All mockups</a><button type="button" data-notes>Design notes</button></div>`;
  }

  /* ---------- product card ---------- */
  function card(p, o = {}) {
    const tags = [];
    if (p.sold) tags.push('<span class="badge sold">Sold out</span>');
    else {
      if (p.vibes.includes('holiday')) tags.push('<span class="badge mint">❄ Holiday</span>');
      else if (p.new) tags.push('<span class="badge new">New</span>');
      if (!p.unl && !p.v.length && p.stock === 1) tags.push('<span class="badge">One of a kind</span>');
      if (p.v.length) tags.push(`<span class="badge pink">${p.v.filter(v => v.s > 0).length} options</span>`);
    }
    const href = `product.html?id=${p.id}`;
    const second = p.img[1] ? `<img src="${img(p.img[1])}" alt="" loading="lazy">` : '';
    const action = p.sold ? '' : (p.v.length
      ? `<a class="quick" href="${href}" style="display:grid;place-items:center;text-decoration:none">Choose an option</a>`
      : `<button class="quick" data-add="${p.id}">Add to bag — ${money(p.p)}</button>`);
    return `<article class="card${p.sold ? ' is-sold' : ''}">
      <div class="media">
        <a class="ph" href="${href}" tabindex="-1" aria-hidden="true"><img src="${img(p.img[0])}" alt="" loading="${o.eager ? 'eager' : 'lazy'}">${second}</a>
        <div class="tags">${tags.join('')}</div>
        ${action}
      </div>
      <div class="meta">
        <span class="kind">${SINGULAR[p.type] || p.type}</span>
        <a class="nm" href="${href}">${esc(p.n)}</a>
        <div class="row"><span class="price">${money(p.p)}</span>
        ${p.sold ? `<a class="remake" href="custom.html?remake=${p.id}">Request a remake</a>` : ''}</div>
      </div>
    </article>`;
  }

  /* ---------- bead divider ---------- */
  function beadDivider(el) {
    const w = 1200, h = 26, n = +(el.dataset.beads || 46);
    const cols = ['var(--pink)', 'var(--teal)', 'var(--lilac)', 'var(--butter)', 'var(--mint)', '#fff'];
    let s = 7, out = `<path d="M0 13 Q ${w / 4} 17 ${w / 2} 13 T ${w} 13" fill="none" stroke="currentColor" stroke-width="1.5" opacity=".5"/>`;
    const rnd = () => (s = (s * 9301 + 49297) % 233280) / 233280;
    for (let i = 0; i < n; i++) {
      const x = (i + .5) * (w / n), r = rnd() > .75 ? 9 : rnd() > .4 ? 6.5 : 4.5;
      const y = 13 + Math.sin(i / n * Math.PI * 2) * 2;
      out += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${cols[Math.floor(rnd() * cols.length)]}" stroke="currentColor" stroke-width="1.3"/>`;
    }
    el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" class="bead-divider" aria-hidden="true">${out}</svg>`;
  }

  /* ---------- cart ---------- */
  function cartCount() { return cart.reduce((a, l) => a + l.q, 0); }
  function cartTotal() { return cart.reduce((a, l) => a + (byId[l.id] ? byId[l.id].p * l.q : 0), 0); }
  function renderCart() {
    $$('[data-cart-count]').forEach(e => { const n = cartCount(); e.textContent = n || ''; e.dataset.n = n; });
    const body = $('#cartBody'), foot = $('#cartFoot'); if (!body) return;
    const t = cartTotal(), left = Math.max(0, FREE_SHIP - t), beads = 10, on = Math.min(beads, Math.round(t / FREE_SHIP * beads));
    const meter = `<div class="ship-meter"><p>${left > 0 ? `You're <b>${money(left)}</b> away from free shipping` : '🎉 You unlocked free shipping!'}</p>
      <div class="bead-bar">${Array.from({ length: beads }, (_, i) => `<i class="${i < on ? 'on' : ''}"></i>`).join('')}</div></div>`;
    if (!cart.length) {
      body.innerHTML = meter + `<div class="cart-empty"><p style="font-size:44px;line-height:1">🧿</p><p style="margin-top:10px">Your bag is empty — let's fix that.</p></div>`;
      foot.innerHTML = `<a class="btn block" href="shop.html">Start shopping</a>`;
      return;
    }
    body.innerHTML = meter + cart.map((l, i) => {
      const p = byId[l.id]; if (!p) return '';
      const vi = l.v && p.v.find(v => v.n === l.v);
      return `<div class="cart-line"><img src="${img((vi && vi.img) || p.img[0])}" alt=""><div><b>${esc(p.n)}</b><small>${l.v ? esc(l.v) + ' · ' : ''}Qty ${l.q}</small><br><button data-rm="${i}">Remove</button></div><span class="price">${money(p.p * l.q)}</span></div>`;
    }).join('') + `<p class="muted" style="font-size:13px;margin-top:14px">🎁 Add a free gift note at checkout.</p>`;
    foot.innerHTML = `<div class="cart-total"><span>Subtotal</span><span class="price">${money(t)}</span></div>
      <button class="btn block" onclick="INF.toast('Mockup: this goes to Squarespace checkout')">Checkout</button>`;
  }
  function add(id, v, q = 1) {
    const p = byId[id]; if (!p) return;
    const ex = cart.find(l => l.id === id && l.v === v);
    if (ex) ex.q += q; else cart.push({ id, v: v || null, q });
    store.set('inf-cart', cart); renderCart(); open('cart');
  }

  /* ---------- drawers, search, toast ---------- */
  function open(id) { $('#' + id).classList.add('open'); $('.drawer-scrim').classList.add('open'); }
  function closeAll() { $$('.drawer.open').forEach(d => d.classList.remove('open')); $('.drawer-scrim').classList.remove('open'); }
  let tt;
  function toast(msg) { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(tt); tt = setTimeout(() => t.classList.remove('show'), 2600); }

  function search(q) {
    const r = $('#searchResults'); q = q.trim().toLowerCase();
    if (!q) { r.innerHTML = ''; return; }
    const hits = C.filter(p => (p.n + ' ' + p.type + ' ' + p.vibes.join(' ')).toLowerCase().includes(q)).slice(0, 6);
    r.innerHTML = hits.length ? hits.map(p => `<a href="product.html?id=${p.id}"><img src="${img(p.img[0])}" alt=""><span><b>${esc(p.n)}</b><br><small class="muted">${SINGULAR[p.type]} · ${money(p.p)}${p.sold ? ' · sold out' : ''}</small></span></a>`).join('')
      : `<p class="muted" style="padding:8px">Nothing yet for “${esc(q)}” — <a href="custom.html">I can make it custom</a>.</p>`;
  }

  /* ---------- design notes ---------- */
  function notes() {
    const els = $$('[data-note]');
    els.forEach((el, i) => {
      if (getComputedStyle(el).position === 'static') el.style.position = 'relative';
      const pin = document.createElement('span');
      pin.className = 'note-pin'; pin.tabIndex = 0; pin.innerHTML = `${i + 1}<span class="tip">${esc(el.dataset.note)}</span>`;
      pin.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); pin.classList.toggle('open'); });
      el.prepend(pin);
    });
    const b = $('[data-notes]'); if (b) b.textContent = `Design notes (${els.length})`;
    const on = new URLSearchParams(location.search).get('notes') === '1' || store.get('inf-notes', false);
    document.body.classList.toggle('notes', on);
  }

  /* ---------- boot ---------- */
  function boot() {
    const page = document.body.dataset.page || '';
    const top = $('#site-top'), bottom = $('#site-bottom');
    if (top) top.outerHTML = header(page);
    if (bottom) bottom.outerHTML = footer();
    $$('[data-beads]').forEach(beadDivider);
    renderCart();
    document.addEventListener('click', e => {
      const a = e.target.closest('[data-add]'); if (a) { e.preventDefault(); add(a.dataset.add); toast('Added to your bag ✨'); return; }
      const o = e.target.closest('[data-open]'); if (o) { open(o.dataset.open); return; }
      if (e.target.closest('[data-close]')) { closeAll(); return; }
      const rm = e.target.closest('[data-rm]'); if (rm) { cart.splice(+rm.dataset.rm, 1); store.set('inf-cart', cart); renderCart(); return; }
      if (e.target.closest('[data-search]')) { const s = $('#searchPop'); s.classList.toggle('open'); if (s.classList.contains('open')) $('#q').focus(); return; }
      if (e.target.closest('[data-notes]')) { const on = !document.body.classList.contains('notes'); document.body.classList.toggle('notes', on); store.set('inf-notes', on); }
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeAll(); $('#searchPop')?.classList.remove('open'); } });
    $('#q')?.addEventListener('input', e => search(e.target.value));
    document.dispatchEvent(new Event('inf:ready'));
    notes();
  }

  window.INF = { C, byId, card, money, img, esc, add, toast, open, SINGULAR, VIBES, FREE_SHIP, LOGO_MARK, store };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
