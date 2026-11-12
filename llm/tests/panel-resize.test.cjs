// Test the public splitter API with deterministic events, layout and animation frames.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../panel-resize.js'), 'utf8');
assert.equal(typeof require('../panel-resize.js').create, 'function');

class Events {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, listener) { if (!this.listeners.has(type)) this.listeners.set(type, new Set()); this.listeners.get(type).add(listener); }
  removeEventListener(type, listener) { this.listeners.get(type)?.delete(listener); }
  emit(type, values = {}) {
    const event = { type, pointerId: 1, pointerType: 'mouse', button: 0, isPrimary: true, clientX: 0, clientY: 0,
      preventDefault() { this.prevented = true; }, stopPropagation() { this.stopped = true; }, ...values };
    for (const listener of Array.from(this.listeners.get(type) || [])) listener(event);
    return event;
  }
  listenerCount() { return Array.from(this.listeners.values()).reduce((n, set) => n + set.size, 0); }
}
class Handle extends Events {
  constructor() { super(); this.attributes = new Map(); this.captures = new Set(); this.hidden = false; this.disabled = false; this.isConnected = true; this.visible = true; this.computed = {}; }
  getAttribute(name) { return this.attributes.has(name) ? this.attributes.get(name) : null; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  removeAttribute(name) { this.attributes.delete(name); }
  closest() { return this.ancestorHidden ? {} : null; }
  getClientRects() { return this.visible ? [{}] : []; }
  setPointerCapture(id) { this.captures.add(id); }
  hasPointerCapture(id) { return this.captures.has(id); }
  releasePointerCapture(id) { this.captures.delete(id); this.emit('lostpointercapture', { pointerId: id }); }
  focus() { this.focused = true; }
}
function environment(stored = {}) {
  const win = new Events(), frames = new Map(), data = new Map(Object.entries(stored)), classes = new Set(), styles = new Map();
  let frameId = 0, writes = 0, removes = 0;
  const body = {
    classList: { add: key => classes.add(key), remove: key => classes.delete(key), contains: key => classes.has(key) },
    style: { getPropertyValue: key => styles.get(key)?.value || '', getPropertyPriority: key => styles.get(key)?.priority || '',
      setProperty: (key, value, priority = '') => styles.set(key, { value, priority }), removeProperty: key => styles.delete(key) }
  };
  const context = { document: { body }, localStorage: {
    getItem: key => data.get(key) ?? null,
    setItem: (key, value) => { writes++; data.set(key, value); },
    removeItem: key => { removes++; data.delete(key); }
  }, getComputedStyle: handle => handle.computed,
  requestAnimationFrame: callback => { const id = ++frameId; frames.set(id, callback); return id; },
  cancelAnimationFrame: id => frames.delete(id),
  addEventListener: win.addEventListener.bind(win), removeEventListener: win.removeEventListener.bind(win) };
  context.window = context;
  vm.runInNewContext(source, context);
  function runFrames() {
    const batch = Array.from(frames.entries());
    for (const [id, callback] of batch) if (frames.delete(id)) callback();
  }
  return { api: context.SquarePanelResize, context, win, body, frames, data, runFrames,
    get writes() { return writes; }, get removes() { return removes; } };
}
function panel(controller, options = {}) {
  let value = options.value ?? 200;
  const applied = [], handle = options.handle || new Handle(), bounds = options.bounds || { min: 100, max: 500 };
  const unbind = controller.attach({ id: options.id || 'left', handle, axis: options.axis || 'x', measure: () => value,
    limits: () => bounds, apply: size => { value = size; applied.push(size); options.onApply?.(size); },
    ...(options.direction !== undefined ? { direction: options.direction } : {}),
    ...(options.defaultValue !== undefined ? { defaultValue: options.defaultValue } : {}) });
  return { handle, bounds, applied, unbind, get value() { return value; }, set value(next) { value = next; } };
}

// Restore and window clamping never overwrite the user's wider saved preference.
const e = environment({ layout: '{"left":500,"bad":1.5,"negative":-4}', unrelated: 'keep' }), resized = [], committed = [];
const controller = e.api.create({ storageKey: 'layout', onResize: event => resized.push(event), onCommit: event => committed.push(event) });
assert.equal(controller.restore('left', 250, { min: 100, max: 400 }), 400);
assert.equal(controller.get('left', 250), 500); assert.equal(controller.get('bad', 250), 250);
assert.equal(controller.get('negative', 250), 250); assert.equal(e.writes, 0);
assert.equal(controller.restore('missing', () => 211.4, () => ({ min: 100, max: 400 })), 211);
const handle = new Handle(); handle.setAttribute('aria-label', '左侧宽度');
const p = panel(controller, { handle, bounds: { min: 100, max: 400 }, defaultValue: 220 });
assert.equal(p.applied.length, 0); assert.equal(p.value, 200);
assert.equal(handle.getAttribute('role'), 'separator'); assert.equal(handle.getAttribute('aria-orientation'), 'vertical');
assert.equal(handle.getAttribute('tabindex'), '0'); assert.equal(handle.getAttribute('aria-label'), '左侧宽度');
assert.equal(handle.getAttribute('aria-valuemin'), '100'); assert.equal(handle.getAttribute('aria-valuemax'), '400');
assert.equal(handle.getAttribute('aria-valuenow'), '200');
e.body.style.setProperty('cursor', 'crosshair');
const down = handle.emit('pointerdown', { clientX: 100 });
assert.ok(down.prevented && down.stopped && handle.hasPointerCapture(1) && handle.focused);
assert.ok(e.body.classList.contains('is-resizing-panels')); assert.equal(e.body.style.getPropertyValue('cursor'), 'col-resize');
handle.emit('pointermove', { clientX: 120 }); handle.emit('pointermove', { clientX: 130 });
assert.equal(e.frames.size, 1); assert.equal(p.value, 200); assert.equal(controller.get('left'), 500);
e.runFrames(); assert.equal(p.value, 230); assert.equal(p.applied.length, 1); assert.equal(e.writes, 0);
handle.emit('pointermove', { clientX: 180 }); handle.emit('pointerup', { clientX: 185 });
assert.equal(p.value, 285); assert.equal(controller.get('left'), 285); assert.equal(e.frames.size, 0);
assert.equal(committed.length, 1); assert.equal(committed[0].reason, 'pointer'); assert.equal(resized.at(-1).value, 285);
assert.equal(handle.getAttribute('aria-valuenow'), '285'); assert.equal(handle.hasPointerCapture(1), false);
assert.equal(e.body.classList.contains('is-resizing-panels'), false); assert.equal(e.body.style.getPropertyValue('cursor'), 'crosshair');
assert.equal(JSON.parse(e.data.get('layout')).left, 285); assert.equal(e.data.get('unrelated'), 'keep');
handle.emit('pointerdown', { clientX: 30 }); handle.emit('pointerup', { clientX: 30 });
assert.equal(committed.length, 1, 'A click without a resize must not replace preferences');
p.bounds.max = 240; p.value = 240; e.win.emit('resize'); e.runFrames();
assert.equal(handle.getAttribute('aria-valuemax'), '240'); assert.equal(handle.getAttribute('aria-valuenow'), '240');
assert.equal(controller.get('left'), 285); assert.equal(e.writes, 1);

// Pointer direction may be reversed; keyboard arrows retain numeric size semantics.
const reverse = panel(controller, { id: 'right', direction: -1, value: 300 });
reverse.handle.emit('pointerdown', { clientX: 200 }); reverse.handle.emit('pointermove', { clientX: 250 }); e.runFrames();
reverse.handle.emit('pointerup', { clientX: 250 }); assert.equal(reverse.value, 250);
reverse.handle.emit('keydown', { key: 'ArrowRight' }); assert.equal(reverse.value, 260);
reverse.handle.emit('keydown', { key: 'ArrowLeft', shiftKey: true }); assert.equal(reverse.value, 210);
reverse.handle.emit('keydown', { key: 'Home' }); assert.equal(reverse.value, 100);
reverse.handle.emit('keydown', { key: 'End' }); assert.equal(reverse.value, 500);
assert.equal(reverse.handle.emit('keydown', { key: 'ArrowUp' }).prevented, undefined);
reverse.handle.emit('keydown', { key: 'ArrowLeft', ctrlKey: true }); assert.equal(reverse.value, 500);
const y = panel(controller, { id: 'height', axis: 'y', value: 200, bounds: { min: 120.2, max: 399.9 } });
assert.equal(y.handle.getAttribute('aria-orientation'), 'horizontal');
y.handle.emit('pointerdown', { clientX: 300, clientY: 10, pointerType: 'touch' });
assert.equal(e.body.style.getPropertyValue('cursor'), 'row-resize');
y.handle.emit('pointermove', { clientX: 999, clientY: 11.6, pointerType: 'touch' }); e.runFrames();
assert.equal(y.value, 202); y.handle.emit('pointerup', { clientY: 11.6 });
y.handle.emit('keydown', { key: 'ArrowDown' }); assert.equal(y.value, 212);
y.handle.emit('keydown', { key: 'ArrowUp', shiftKey: true }); assert.equal(y.value, 162);
y.handle.emit('keydown', { key: 'Home' }); assert.equal(y.value, 121);
y.handle.emit('keydown', { key: 'End' }); assert.equal(y.value, 399);

// Cancel/lost capture/window blur clean up and roll back without persisting a gesture.
const cancel = panel(controller, { id: 'cancel', value: 200 }), beforeWrites = e.writes, beforeCommits = committed.length;
cancel.handle.emit('pointerdown', { clientX: 10 }); cancel.handle.emit('pointermove', { clientX: 60 }); e.runFrames();
assert.equal(cancel.value, 250); cancel.handle.emit('pointercancel', { clientX: 60 });
assert.equal(cancel.value, 200); assert.equal(e.writes, beforeWrites); assert.equal(committed.length, beforeCommits);
assert.equal(cancel.handle.hasPointerCapture(1), false); assert.equal(e.body.classList.contains('is-resizing-panels'), false);
cancel.handle.emit('pointerdown', { clientX: 10 }); cancel.handle.emit('pointermove', { clientX: 70 });
cancel.handle.emit('pointercancel'); e.runFrames(); assert.equal(cancel.value, 200);
for (const ending of ['lostpointercapture', 'blur']) {
  cancel.handle.emit('pointerdown', { clientX: 10 }); cancel.handle.emit('pointermove', { clientX: 70 }); e.runFrames();
  if (ending === 'blur') e.win.emit('blur'); else cancel.handle.emit(ending);
  assert.equal(cancel.value, 200); assert.equal(e.body.classList.contains('is-resizing-panels'), false);
}
cancel.handle.emit('pointerdown', { clientX: 10 }); cancel.handle.emit('pointermove', { clientX: 60, pointerId: 2 });
assert.equal(e.frames.size, 0); e.win.emit('pointerup', { clientX: 40 }); assert.equal(cancel.value, 230);

// Disabled or hidden handles ignore all input; pending work cannot resize a hidden pane.
for (const disable of [h => { h.hidden = true; }, h => { h.disabled = true; }, h => h.setAttribute('aria-disabled', 'true'),
  h => { h.ancestorHidden = true; }, h => { h.visible = false; }, h => { h.isConnected = false; },
  h => { h.computed.visibility = 'hidden'; }, h => { h.computed.display = 'none'; }]) {
  const blocked = panel(controller, { id: 'blocked' }); disable(blocked.handle);
  blocked.handle.emit('pointerdown'); blocked.handle.emit('keydown', { key: 'ArrowRight' }); blocked.handle.emit('dblclick');
  assert.equal(blocked.applied.length, 0); assert.equal(blocked.handle.hasPointerCapture(1), false);
}
const hideDuring = panel(controller, { id: 'hide-during' });
hideDuring.handle.emit('pointerdown'); hideDuring.handle.emit('pointermove', { clientX: 50 }); hideDuring.handle.hidden = true; e.runFrames();
assert.equal(hideDuring.applied.length, 0); assert.equal(e.body.classList.contains('is-resizing-panels'), false);

// Double click evaluates a live default; reset clears only its independent storage.
let defaultWidth = 220;
const defaults = panel(controller, { id: 'defaults', value: 300, defaultValue: () => defaultWidth });
defaultWidth = 260; defaults.handle.emit('dblclick'); assert.equal(defaults.value, 260); assert.equal(controller.get('defaults'), 260);
defaults.handle.emit('keydown', { key: 'End' }); defaultWidth = 180;
controller.reset(); assert.equal(defaults.value, 180); assert.equal(p.value, 220);
assert.equal(controller.get('defaults', 999), 999); assert.equal(e.data.has('layout'), false); assert.equal(e.data.get('unrelated'), 'keep');
assert.equal(committed.at(-1).reason, 'reset');

// Reattaching the same ID replaces listeners; an old unbind cannot remove the new binding.
const originalHandle = new Handle(); originalHandle.setAttribute('role', 'button'); originalHandle.setAttribute('tabindex', '-1');
const old = panel(controller, { id: 'replace', handle: originalHandle });
const replacement = panel(controller, { id: 'replace' });
assert.equal(originalHandle.listenerCount(), 0); assert.equal(originalHandle.getAttribute('role'), 'button'); assert.equal(originalHandle.getAttribute('tabindex'), '-1');
old.unbind(); replacement.handle.emit('keydown', { key: 'ArrowRight' }); assert.equal(replacement.value, 210);
replacement.handle.emit('pointerdown'); replacement.handle.emit('pointermove', { clientX: 50 });
controller.detach('replace'); e.runFrames(); assert.equal(replacement.value, 210); assert.equal(replacement.handle.hasPointerCapture(1), false);
assert.equal(e.body.classList.contains('is-resizing-panels'), false); assert.equal(replacement.handle.getAttribute('role'), null);
controller.destroy(); controller.destroy(); assert.equal(e.win.listenerCount(), 0); assert.equal(e.frames.size, 0);
assert.throws(() => panel(controller), /已销毁/);

// Multiple instances share the body cursor safely and restore the previous cursor priority.
const shared = environment(), first = shared.api.create(), secondController = shared.api.create();
shared.body.style.setProperty('cursor', 'help', 'important');
const firstPanel = panel(first), secondPanel = panel(secondController, { axis: 'y' });
firstPanel.handle.emit('pointerdown', { pointerId: 1 }); secondPanel.handle.emit('pointerdown', { pointerId: 2 });
assert.equal(shared.body.style.getPropertyValue('cursor'), 'row-resize'); first.destroy();
assert.ok(shared.body.classList.contains('is-resizing-panels')); assert.equal(shared.body.style.getPropertyValue('cursor'), 'row-resize');
secondController.destroy(); assert.equal(shared.body.style.getPropertyValue('cursor'), 'help');
assert.equal(shared.body.style.getPropertyPriority('cursor'), 'important'); assert.equal(shared.body.classList.contains('is-resizing-panels'), false);

// Storage denial and malformed JSON do not break resizing; apply errors still release capture.
for (const text of ['broken json', 'null', '[]', '42']) {
  const bad = environment({ layout: text }), api = bad.api.create({ storageKey: 'layout' });
  assert.equal(api.restore('left', 240), 240); api.destroy();
}
const failure = environment(); failure.context.localStorage.setItem = () => { throw Error('storage denied'); };
const failsafe = failure.api.create({ storageKey: 'layout' }), sessionPanel = panel(failsafe);
sessionPanel.handle.emit('keydown', { key: 'ArrowRight' }); assert.equal(failsafe.get('left'), 210);
const throwing = panel(failsafe, { id: 'throw', onApply: () => { throw Error('layout error'); } });
throwing.handle.emit('pointerdown'); throwing.handle.emit('pointermove', { clientX: 40 });
assert.throws(() => failure.runFrames(), /layout error/); assert.equal(throwing.handle.hasPointerCapture(1), false);
assert.equal(failure.body.classList.contains('is-resizing-panels'), false); failsafe.destroy();

console.log('panel resize tests passed: pointer capture/RAF, keyboard, persistence, cancellation, ARIA and lifecycle');
