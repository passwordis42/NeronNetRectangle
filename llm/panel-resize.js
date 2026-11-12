/* Pointer/keyboard splitters. Pixel preferences are independent of model state. */
(function (root, factory) {
  const api = factory(root);
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SquarePanelResize = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';
  const bodySessions = new WeakMap();
  const ownedAttributes = ['role', 'aria-orientation', 'aria-valuemin', 'aria-valuemax', 'aria-valuenow', 'tabindex'];
  const number = value => typeof value === 'number' && Number.isFinite(value);
  const resolve = value => typeof value === 'function' ? value() : value;
  const requestFrame = callback => root.requestAnimationFrame ? root.requestAnimationFrame(callback) : root.setTimeout(callback, 16);
  const cancelFrame = id => { if (id !== null) (root.cancelAnimationFrame || root.clearTimeout).call(root, id); };

  function range(limits) {
    const value = resolve(limits) || {};
    const min = number(value.min) ? Math.max(0, Math.ceil(value.min)) : 0;
    const max = number(value.max) ? Math.max(min, Math.floor(value.max)) : Number.MAX_SAFE_INTEGER;
    return { min, max };
  }
  function clamp(value, limits) {
    const bounds = range(limits), resolved = resolve(value);
    return Math.max(bounds.min, Math.min(bounds.max, number(resolved) ? Math.round(resolved) : bounds.min));
  }
  function available(handle) {
    if (handle.hidden || handle.disabled || handle.getAttribute('aria-disabled') === 'true' || handle.isConnected === false) return false;
    if (handle.closest?.('[hidden], [inert], [aria-hidden="true"]')) return false;
    if (handle.getClientRects && handle.getClientRects().length === 0) return false;
    const style = root.getComputedStyle?.(handle);
    return !style || style.display !== 'none' && style.visibility !== 'hidden' && style.visibility !== 'collapse';
  }
  function stop(event) { event.preventDefault?.(); event.stopPropagation?.(); }
  function beginBody(token, axis) {
    const body = root.document?.body;
    if (!body) return null;
    let state = bodySessions.get(body);
    if (!state) {
      state = { owners: new Map(), hadClass: body.classList.contains('is-resizing-panels'),
        cursor: body.style.getPropertyValue('cursor'), priority: body.style.getPropertyPriority('cursor') };
      bodySessions.set(body, state);
    }
    state.owners.set(token, axis === 'x' ? 'col-resize' : 'row-resize');
    body.classList.add('is-resizing-panels');
    body.style.setProperty('cursor', state.owners.get(token), 'important');
    return body;
  }
  function endBody(body, token) {
    const state = body && bodySessions.get(body);
    if (!state) return;
    state.owners.delete(token);
    if (state.owners.size) {
      body.style.setProperty('cursor', Array.from(state.owners.values()).at(-1), 'important');
    } else {
      if (state.cursor) body.style.setProperty('cursor', state.cursor, state.priority);
      else body.style.removeProperty('cursor');
      if (!state.hadClass) body.classList.remove('is-resizing-panels');
      bodySessions.delete(body);
    }
  }

  function create({ storageKey, onResize, onCommit } = {}) {
    const saved = new Map(), attachments = new Map();
    let active = null, dragFrame = null, ariaFrame = null, destroyed = false;
    if (storageKey) {
      try {
        const data = JSON.parse(root.localStorage?.getItem(storageKey) || '{}');
        if (data && typeof data === 'object' && !Array.isArray(data)) {
          for (const [id, value] of Object.entries(data)) if (Number.isSafeInteger(value) && value >= 0) saved.set(id, value);
        }
      } catch (_) { /* Missing or malformed preferences do not disable resizing. */ }
    }
    function persist() {
      if (!storageKey) return;
      try { root.localStorage?.setItem(storageKey, JSON.stringify(Object.fromEntries(saved))); } catch (_) { /* Session sizes still work when storage is unavailable. */ }
    }
    function syncAria(entry, explicitValue) {
      const bounds = range(entry.limits), value = clamp(explicitValue ?? entry.measure(), bounds);
      entry.handle.setAttribute('aria-valuemin', String(bounds.min));
      entry.handle.setAttribute('aria-valuemax', String(bounds.max));
      entry.handle.setAttribute('aria-valuenow', String(value));
      return value;
    }
    function apply(entry, proposed, reason) {
      const value = clamp(proposed, entry.limits);
      entry.apply(value);
      syncAria(entry, value);
      onResize?.({ id: entry.id, value, axis: entry.axis, reason });
      return value;
    }
    function commit(entry, value, reason) {
      saved.set(entry.id, value); persist();
      onCommit?.({ id: entry.id, value, axis: entry.axis, reason });
    }
    function release(transaction) {
      try {
        if (transaction.entry.handle.hasPointerCapture?.(transaction.pointerId)) transaction.entry.handle.releasePointerCapture(transaction.pointerId);
      } catch (_) { /* The browser may already have released capture. */ }
      endBody(transaction.body, transaction);
    }
    function finish(shouldCommit, rollback = false) {
      const transaction = active;
      if (!transaction) return;
      active = null;
      cancelFrame(dragFrame); dragFrame = null;
      try {
        if (shouldCommit && transaction.moved && available(transaction.entry.handle)) {
          const value = apply(transaction.entry, transaction.pending ?? transaction.lastValue ?? transaction.startValue, 'pointer');
          commit(transaction.entry, value, 'pointer');
        } else if (rollback && transaction.lastValue !== null && available(transaction.entry.handle)) {
          apply(transaction.entry, transaction.startValue, 'cancel');
        }
      } finally { release(transaction); }
    }
    function queue(transaction, value) {
      transaction.pending = clamp(value, transaction.entry.limits);
      if (dragFrame !== null) return;
      dragFrame = requestFrame(() => {
        dragFrame = null;
        if (active !== transaction) return;
        if (!available(transaction.entry.handle)) { finish(false); return; }
        const pending = transaction.pending; transaction.pending = null;
        try { transaction.lastValue = apply(transaction.entry, pending, 'pointer'); }
        catch (error) { finish(false); throw error; }
      });
    }
    function pointerMove(event) {
      if (!active || event.pointerId !== active.pointerId) return;
      stop(event);
      if (!available(active.entry.handle)) { finish(false); return; }
      const position = active.entry.axis === 'x' ? event.clientX : event.clientY;
      if (!number(position)) return;
      const delta = position - active.startPosition;
      if (delta !== 0) active.moved = true;
      if (active.moved) queue(active, active.startValue + delta * active.entry.direction);
    }
    function pointerUp(event) {
      if (!active || event.pointerId !== active.pointerId) return;
      pointerMove(event); finish(true);
    }
    function pointerCancel(event) {
      if (!active || event.pointerId !== active.pointerId) return;
      stop(event); finish(false, true);
    }
    function refreshAria() {
      if (destroyed || ariaFrame !== null) return;
      ariaFrame = requestFrame(() => {
        ariaFrame = null;
        if (!destroyed) for (const entry of attachments.values()) syncAria(entry);
      });
    }
    // Fallbacks also clean up when capture is unavailable or a window loses focus.
    root.addEventListener?.('pointerup', pointerUp);
    root.addEventListener?.('pointercancel', pointerCancel);
    const blur = () => finish(false, true);
    root.addEventListener?.('blur', blur);
    root.addEventListener?.('resize', refreshAria);

    function detach(id) {
      const entry = attachments.get(String(id));
      if (!entry) return;
      if (active?.entry === entry) finish(false);
      for (const [type, listener] of Object.entries(entry.listeners)) entry.handle.removeEventListener(type, listener);
      for (const [name, value] of entry.originalAttributes) {
        if (value === null) entry.handle.removeAttribute(name); else entry.handle.setAttribute(name, value);
      }
      attachments.delete(entry.id);
    }
    function attach(options) {
      if (destroyed) throw Error('面板尺寸控制器已销毁');
      const { id, handle, axis, measure, limits, apply: applySize, direction = 1, defaultValue } = options || {};
      if (typeof id !== 'string' || !id || !handle?.addEventListener || !['x', 'y'].includes(axis) ||
        typeof measure !== 'function' || typeof limits !== 'function' || typeof applySize !== 'function' || ![1, -1].includes(direction)) throw TypeError('面板尺寸手柄配置无效');
      detach(id);
      const initialDefault = measure(), entry = { id, handle, axis, measure, limits, apply: applySize, direction,
        defaultValue: defaultValue ?? initialDefault, originalAttributes: ownedAttributes.map(name => [name, handle.getAttribute(name)]) };
      const pointerDown = event => {
        if (active || !available(handle) || event.isPrimary === false || event.button !== undefined && event.button !== 0 || !number(event.pointerId)) return;
        const position = axis === 'x' ? event.clientX : event.clientY;
        if (!number(position)) return;
        stop(event);
        active = { entry, pointerId: event.pointerId, startPosition: position, startValue: clamp(measure(), limits), pending: null, lastValue: null, moved: false };
        active.body = beginBody(active, axis);
        try { handle.setPointerCapture(event.pointerId); } catch (_) { /* Window pointerup remains available. */ }
        handle.focus?.({ preventScroll: true });
      };
      const lostCapture = event => { if (active?.entry === entry && event.pointerId === active.pointerId) finish(false, true); };
      const keyDown = event => {
        if (active || !available(handle) || event.altKey || event.ctrlKey || event.metaKey) return;
        const bounds = range(limits), step = event.shiftKey ? 50 : 10;
        const negativeKey = axis === 'x' ? 'ArrowLeft' : 'ArrowUp', positiveKey = axis === 'x' ? 'ArrowRight' : 'ArrowDown';
        let target;
        if (event.key === 'Home') target = bounds.min;
        else if (event.key === 'End') target = bounds.max;
        else if (event.key === negativeKey || event.key === positiveKey) target = measure() + (event.key === positiveKey ? step : -step);
        else return;
        stop(event); commit(entry, apply(entry, target, 'keyboard'), 'keyboard');
      };
      const doubleClick = event => {
        if (active || !available(handle)) return;
        stop(event); commit(entry, apply(entry, resolve(entry.defaultValue), 'doubleclick'), 'doubleclick');
      };
      entry.listeners = { pointerdown: pointerDown, pointermove: pointerMove, pointerup: pointerUp, pointercancel: pointerCancel,
        lostpointercapture: lostCapture, keydown: keyDown, dblclick: doubleClick };
      attachments.set(id, entry);
      handle.setAttribute('role', 'separator'); handle.setAttribute('aria-orientation', axis === 'x' ? 'vertical' : 'horizontal');
      handle.setAttribute('tabindex', '0'); syncAria(entry);
      for (const [type, listener] of Object.entries(entry.listeners)) handle.addEventListener(type, listener);
      return () => { if (attachments.get(id) === entry) detach(id); };
    }
    function reset() {
      if (destroyed) return;
      finish(false); saved.clear();
      try { if (storageKey) root.localStorage?.removeItem(storageKey); } catch (_) { /* In-memory reset still succeeds. */ }
      for (const entry of Array.from(attachments.values())) {
        const value = apply(entry, resolve(entry.defaultValue), 'reset');
        onCommit?.({ id: entry.id, value, axis: entry.axis, reason: 'reset' });
      }
    }
    function destroy() {
      if (destroyed) return;
      finish(false); cancelFrame(ariaFrame); ariaFrame = null;
      for (const id of Array.from(attachments.keys())) detach(id);
      root.removeEventListener?.('pointerup', pointerUp); root.removeEventListener?.('pointercancel', pointerCancel);
      root.removeEventListener?.('blur', blur); root.removeEventListener?.('resize', refreshAria);
      destroyed = true;
    }
    return { attach, detach,
      restore: (id, defaultValue, limits) => clamp(saved.has(String(id)) ? saved.get(String(id)) : resolve(defaultValue), limits),
      get: (id, fallback) => saved.has(String(id)) ? saved.get(String(id)) : fallback,
      reset, destroy };
  }
  return { create };
});
