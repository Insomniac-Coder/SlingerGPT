// SlingerGPT chat: chats live in this browser (localStorage); replies stream from chat_server.py.
const $ = (id) => document.getElementById(id);
const store = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem('modelchat.' + key)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem('modelchat.' + key, JSON.stringify(value)); } catch {} },
};
// Measured on SlingerGPT 1 pico Preview 1 (14 November): temperature 0.3 got 25 of 40 fact answers right against 18
// at 0.7; at 0.3 it loops more, and anti-repetition 0.8 halves that (15% -> 7% repeated four-word phrases) with
// facts unchanged; 300 words lets nearly all replies end by themselves.
const DEFAULTS = { temperature: 0.3, top_k: 50, words: 300, dry: 0.8 };
// Suggestions SlingerGPT 1 pico Preview 1 answered well in every try (3 tries each at temperature 0.7, 14 November);
// it got facts like the largest planet, who wrote Romeo and Juliet, and sums wrong, so those are left out.
const STARTERS = {
  chat: ['Can you give me a tip for studying for exams?', 'What is photosynthesis?', 'What does the heart do?', 'What is the internet?'],
  continue: ['Once upon a time, in a small village by the sea,', 'Photosynthesis is the process by which', 'A computer program is'],
};
const ICONS = {
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 8.5v-2a2 2 0 0 0-2-2h-7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  retry: '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4h-4"/>',
  trash: '<path d="M5 7h14M9.5 7V5a1 1 0 0 1 1-1h3a1 1 0 0 1 1 1v2M7 7l.8 11.5A1.5 1.5 0 0 0 9.3 20h5.4a1.5 1.5 0 0 0 1.5-1.5L17 7"/>',
};
const svg = (name) => `<svg viewBox="0 0 24 24">${ICONS[name]}</svg>`;

let chats = store.get('chats', []);   // [{id, title, updated, messages: [{role, text, model, meta, error, stopped}]}]
let currentId = null;
let models = [];
let model = store.get('chosen', null);   // a new key, so the newest SlingerGPT is picked the first time
let mode = store.get('mode', 'chat');
let lookupOn = store.get('lookup', false);   // "Look it up": answer from a Wikipedia passage found first (SlingerGPT, chat)
const canLookUp = () => false;   // off for now (owner, 16 Nov); was: mode === 'chat' && info(model)?.family === 'SlingerGPT'
let settings = { ...DEFAULTS, ...store.get('settings', {}) };
let controller = null;   // set while a reply is streaming

const current = () => chats.find((c) => c.id === currentId);
const save = () => store.set('chats', chats);
const info = (name) => models.find((m) => m.name === name);
const titleOf = (name) => info(name)?.title ?? name;

// --- Models -----------------------------------------------------------------

async function loadModels() {
  try {
    models = await (await fetch('/api/models')).json();   // SlingerGPT first, newest first; then the earlier experiments
  } catch {
    $('model-name').textContent = 'Server not running';
    return;
  }
  if (!info(model)) model = models[0]?.name ?? null;
  renderModels();
  render();
}

function renderModels() {
  const m = info(model);
  $('model-name').textContent = m ? m.title : 'No models found';
  $('model-size').textContent = m ? m.params : '';
  const menu = $('model-menu');
  menu.replaceChildren();
  let family = null;
  for (const item of models) {
    if (item.family !== family) {
      family = item.family;
      menu.append(Object.assign(document.createElement('div'), { className: 'menu-label', textContent: family }));
    }
    const b = document.createElement('button');
    b.className = 'menu-item';
    b.setAttribute('role', 'menuitemradio');
    b.setAttribute('aria-checked', item.name === model);
    b.innerHTML = `<span><b></b><small></small></span>${svg('check').replace('<svg', '<svg class="tick"')}`;
    b.querySelector('b').textContent = item.title;
    b.querySelector('small').textContent = `${item.params} weights, remembers the last ${item.context.toLocaleString()} ${item.unit}`;
    b.onclick = () => { model = item.name; store.set('chosen', model); menu.hidePopover(); renderModels(); render(); $('input').focus(); };
    menu.append(b);
  }
}

// Menus open under the button that opens them.
for (const [menuId, buttonId, alignRight] of [['model-menu', 'model-btn', false], ['settings', 'settings-btn', true]]) {
  $(menuId).addEventListener('beforetoggle', (e) => {
    if (e.newState !== 'open') return;
    const r = $(buttonId).getBoundingClientRect();
    const menu = $(menuId);
    menu.style.top = `${r.bottom + 6}px`;
    menu.style.left = alignRight ? 'auto' : `${Math.max(8, r.left)}px`;
    menu.style.right = alignRight ? `${Math.max(8, innerWidth - r.right)}px` : 'auto';
  });
}

// --- Rendering --------------------------------------------------------------

function render() {
  const chat = current();
  const empty = !chat || chat.messages.length === 0;
  $('main').classList.toggle('empty', empty);
  renderSidebar();
  renderMode();
  renderContext();
  if (empty) {
    const m = info(model);
    $('hero-title').textContent = mode === 'chat' ? 'Take your shot.' : 'Start a sentence. It carries on.';
    $('hero-sub').textContent = m ? `${m.title}: ${m.params} weights, trained on this laptop` : '';
    $('starters').replaceChildren(...STARTERS[mode].map((text) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = text;
      b.onclick = () => send(text);
      return b;
    }));
    return;
  }
  $('thread').replaceChildren(...chat.messages.map((msg, i) => messageView(msg, i === chat.messages.length - 1)));
}

function messageView(msg, last) {
  const el = document.createElement('div');
  if (msg.role === 'user') {
    el.className = 'msg user';
    el.textContent = msg.text;
    return el;
  }
  el.className = 'msg assistant';
  const body = document.createElement('div');
  body.className = 'body';
  el.append(body);
  if (msg.error) {
    el.classList.add('error');
    body.innerHTML = msg.error;   // only ever our own fixed messages (see generate), never model or server text
    return el;
  }
  body.innerHTML = rich(msg.text);
  if (msg.streaming) {
    body.append(Object.assign(document.createElement('span'), { className: msg.text ? 'stone' : 'stone whirl' }));
    return el;
  }
  if (!msg.text.trim()) {
    el.classList.add('empty');
    body.textContent = msg.stopped ? 'Stopped before it wrote anything.' : 'It ended its turn without writing anything.';
  }
  if (msg.source) {   // the passage it answered from: its article, with the passage itself one click away
    const source = document.createElement('details');
    source.className = 'source';
    source.innerHTML = '<summary>Looked up: <b></b> <span></span></summary><p></p>';
    source.querySelector('b').textContent = msg.source.title;
    source.querySelector('span').textContent = `(${msg.source.where})`;
    source.querySelector('p').textContent = msg.source.passage;
    el.append(source);
  }
  const actions = document.createElement('div');
  actions.className = 'actions';
  const copy = iconButton('copy', 'Copy', () => {
    navigator.clipboard?.writeText(msg.text);
    copy.innerHTML = svg('check');
    setTimeout(() => (copy.innerHTML = svg('copy')), 1200);
  });
  actions.append(copy);
  if (last) actions.append(iconButton('retry', 'Write another reply', regenerate));
  const meta = document.createElement('span');
  meta.className = 'meta';
  const parts = [titleOf(msg.model)];
  if (msg.meta) {
    parts.push(`${msg.meta.tokens} tokens in ${msg.meta.seconds.toFixed(msg.meta.seconds < 1 ? 2 : 1)} s`);
    if (msg.meta.load_seconds > 0.2) parts.push(`loaded in ${msg.meta.load_seconds.toFixed(1)} s`);
  }
  if (msg.stopped) parts.push('stopped');
  if (msg.meta?.cut) parts.push(`cut off at the ${msg.words}-word limit (Reply length in settings)`);
  meta.textContent = parts.join(', ');
  actions.append(meta);
  el.append(actions);
  return el;
}

function iconButton(icon, label, onclick) {
  const b = document.createElement('button');
  b.className = 'icon-btn';
  b.type = 'button';
  b.title = label;
  b.setAttribute('aria-label', label);
  b.innerHTML = svg(icon);
  b.onclick = onclick;
  return b;
}

function renderSidebar() {
  const list = $('chat-list');
  if (chats.length === 0) {
    list.innerHTML = '<p class="chats-empty">Your chats appear here.</p>';
    return;
  }
  list.replaceChildren(...[...chats].sort((a, b) => b.updated - a.updated).map((chat) => {
    const row = document.createElement('div');
    row.className = 'chat-item' + (chat.id === currentId ? ' active' : '');
    const open = document.createElement('button');
    open.textContent = chat.title;
    open.title = chat.title;
    open.onclick = () => { if (controller) return; currentId = chat.id; closeSidebarOnPhone(); render(); scrollToEnd(); };
    const del = iconButton('trash', 'Delete chat', () => {
      if (controller && chat.id === currentId) return;
      if (!confirm(`Delete "${chat.title}"? This can't be undone.`)) return;
      chats = chats.filter((c) => c !== chat);
      if (currentId === chat.id) currentId = null;
      save();
      render();
    });
    del.classList.add('del');
    row.append(open, del);
    return row;
  }));
}

// The context window bar: how much of what the model can see at once this chat fills. The count comes from the
// server with each reply, in the model's own pieces, so it shows only for the model that wrote the last reply.
function renderContext() {
  const m = info(model), chat = current(), box = $('context');
  box.hidden = !m || !chat?.used || chat.usedBy !== model;
  if (box.hidden) return;
  const used = Math.min(chat.used, m.context), full = chat.used >= m.context;
  Object.assign($('context-meter'), { max: m.context, low: Math.round(m.context * 0.85), high: m.context, value: used });
  $('context-label').textContent = `${used.toLocaleString()} / ${m.context.toLocaleString()}`;
  box.title = full
    ? `Context window full: this chat no longer fits in the ${m.context.toLocaleString()} ${m.unit} ${m.title} can see at once, so it can't see the start of it.`
    : `Context window: this chat fills ${used.toLocaleString()} of the ${m.context.toLocaleString()} ${m.unit} ${m.title} can see at once.`;
}

function renderMode() {
  for (const b of document.querySelectorAll('.mode button')) b.setAttribute('aria-checked', b.dataset.mode === mode);
  $('lookup').hidden = !canLookUp();
  $('lookup').setAttribute('aria-pressed', lookupOn);
  $('input').placeholder = mode === 'continue' ? 'Type the start of a text' : info(model)?.family === 'SlingerGPT' ? 'Ask SlingerGPT' : 'Ask anything';
}

const nearEnd = () => { const s = $('scroll'); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; };
const scrollToEnd = () => { const s = $('scroll'); s.scrollTop = s.scrollHeight; };

// --- Sending ----------------------------------------------------------------

function send(text) {
  text = text.trim();
  if (controller || !text || !model) return;
  let chat = current();
  if (!chat) {
    chat = { id: crypto.randomUUID?.() ?? String(Date.now()), title: text.split('\n')[0].slice(0, 48), updated: Date.now(), messages: [] };
    chats.push(chat);
    currentId = chat.id;
  }
  chat.messages.push({ role: 'user', text });
  $('input').value = '';
  grow();
  generate(chat);
}

function regenerate() {
  const chat = current();
  if (controller || !chat || chat.messages.at(-1)?.role !== 'assistant') return;
  chat.messages.pop();
  generate(chat);
}

async function generate(chat) {
  const history = chat.messages.filter((m) => !m.error && m.text.trim()).map(({ role, text }) => ({ role, text }));
  const reply = { role: 'assistant', text: '', model, words: settings.words, streaming: true };
  chat.messages.push(reply);
  chat.updated = Date.now();
  render();
  scrollToEnd();
  const body = $('thread').lastElementChild.querySelector('.body');
  const stone = body.querySelector('.stone');
  controller = new AbortController();
  setBusy(true);
  let loading = null;   // the "Loading ..." notice's clock, while the model gets onto the graphics card
  try {
    const res = await fetch('/api/reply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, mode, messages: history, lookup: canLookUp() && lookupOn, ...settings }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `The server answered ${res.status}.`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let at;
      while ((at = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, at);
        buffer = buffer.slice(at + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.error) throw new Error(event.error);
        if (event.loading) {   // the server has to load the model first (a switch, or just restarted): say so
          const note = Object.assign(document.createElement('span'), { className: 'loading-note' }), began = Date.now();
          const tick = () => (note.textContent = `Loading ${titleOf(reply.model)} onto the graphics card… ${Math.round((Date.now() - began) / 1000)} s`);
          tick();
          loading = setInterval(tick, 1000);
          body.prepend(note);
        }
        if (event.done) reply.meta = event;
        if (event.source) reply.source = event.source;
        if (event.used) {
          Object.assign(chat, { used: event.used, usedBy: reply.model });
          renderContext();
        }
        if (event.text) {
          clearInterval(loading);   // the notice goes with the old contents of the reply
          const follow = nearEnd();
          reply.text += event.text;
          body.innerHTML = rich(reply.text);
          stone.classList.remove('whirl');   // the first words are out: the stone rests where the text ends
          body.append(stone);
          if (follow) scrollToEnd();
        }
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') reply.stopped = true;
    else if (e instanceof TypeError) reply.error = 'Can\'t reach the chat server. In the llm-demo folder, run <code>python -P chat_server.py</code>, then try again.';
    else reply.error = escapeHtml(e.message);
  } finally {
    clearInterval(loading);
    delete reply.streaming;
    controller = null;
    setBusy(false);
    chat.updated = Date.now();
    save();
    const follow = nearEnd();
    render();
    if (follow) scrollToEnd();
  }
}

const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
// The markdown SlingerGPT picked up from its conversations (bold, headings, inline code); the rest stays plain text.
const rich = (s) => escapeHtml(s)
  .replace(/^#{1,4} +(.+)$/gm, '<strong>$1</strong>')
  .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  .replace(/`([^`\n]+)`/g, '<code>$1</code>');

function setBusy(busy) {
  $('composer').classList.toggle('busy', busy);
  $('send').setAttribute('aria-label', busy ? 'Stop' : 'Send');
  $('send').title = busy ? 'Stop' : 'Send';
  updateSend();
}

function updateSend() { $('send').disabled = !controller && !$('input').value.trim(); }

function grow() {
  const box = $('input');
  box.style.height = 'auto';
  box.style.height = `${box.scrollHeight}px`;
  updateSend();
}

// --- Controls ---------------------------------------------------------------

$('composer').addEventListener('submit', (e) => {
  e.preventDefault();
  if (controller) controller.abort();
  else send($('input').value);
});
$('composer').addEventListener('click', (e) => { if (e.target === e.currentTarget) $('input').focus(); });
$('input').addEventListener('input', grow);
$('input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!controller) send($('input').value);
  }
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && controller) controller.abort(); });
$('lookup').onclick = () => { lookupOn = !lookupOn; store.set('lookup', lookupOn); renderMode(); $('input').focus(); };
for (const b of document.querySelectorAll('.mode button')) {
  b.onclick = () => { mode = b.dataset.mode; store.set('mode', mode); render(); $('input').focus(); };
}

function newChat() {
  if (controller) return;
  currentId = null;
  closeSidebarOnPhone();
  render();
  $('input').focus();
}
$('new-chat').onclick = newChat;

const phone = matchMedia('(max-width: 760px)');
function setSidebar(open) { $('app').classList.toggle('side-closed', !open); if (!phone.matches) store.set('sidebar', open); }
function closeSidebarOnPhone() { if (phone.matches) setSidebar(false); }
$('close-side').onclick = () => setSidebar(false);
$('open-side').onclick = () => setSidebar(true);
$('scrim').onclick = () => setSidebar(false);
const sidebarForWidth = () => setSidebar(phone.matches ? false : store.get('sidebar', true));
phone.addEventListener('change', sidebarForWidth);
sidebarForWidth();

$('theme').onclick = () => {
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const now = document.documentElement.dataset.theme || (dark ? 'dark' : 'light');
  document.documentElement.dataset.theme = now === 'dark' ? 'light' : 'dark';
  store.set('theme', document.documentElement.dataset.theme);
};

// Settings: each slider shows its value and is remembered.
const form = $('settings');
function showSettings() {
  for (const key of Object.keys(DEFAULTS)) {
    form.elements[key].value = settings[key];
    const v = settings[key];
    $(`${key}-out`).textContent = key === 'dry' && v === 0 ? 'off' : key === 'top_k' && v === 0 ? 'no limit' : key === 'words' ? `${v} words` : v;
  }
}
form.addEventListener('input', (e) => {
  settings[e.target.name] = Number(e.target.value);
  store.set('settings', settings);
  showSettings();
});
$('reset-settings').onclick = () => { settings = { ...DEFAULTS }; store.set('settings', settings); showSettings(); };

showSettings();
render();
updateSend();
loadModels();
$('input').focus();
