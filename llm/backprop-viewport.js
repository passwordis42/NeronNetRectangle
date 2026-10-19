/* Presentation only: zoom/pan never changes historical parameters or gradients. */
const SquareReplayViewport = (() => {
  function create({frame,scroll,canvas,replay,onRedraw}) {
    const surface=document.createElement('div');
    surface.className='replay-network-surface';
    canvas.before(surface);surface.append(canvas);
    const controls=document.createElement('div');
    controls.className='replay-zoom-controls';
    controls.setAttribute('role','group');controls.setAttribute('aria-label','观察框缩放');
    controls.innerHTML='<button type="button" data-network-zoom="out" aria-label="缩小网络" title="缩小网络">−</button><select data-replay-view="zoom" aria-label="网络缩放比例"></select><button type="button" data-network-zoom="in" aria-label="放大网络" title="放大网络">＋</button><button type="button" data-network-zoom="fit" title="将全部神经元放入观察框">适应窗口</button><button type="button" data-network-zoom="reset" title="恢复 100% 并回到网络起点">重置</button>';
    frame.append(controls);
    let anchor=null,drag=null,raf=0,lastLayout=null;
    const percentages=[.1,.25,.5,.75,1,1.25,1.5,2,3];
    function capture(clientX,clientY){
      const rect=scroll.getBoundingClientRect();
      const x=clientX===undefined?scroll.clientWidth/2:clientX-rect.left;
      const y=clientY===undefined?scroll.clientHeight/2:clientY-rect.top;
      return {x,y,worldX:(scroll.scrollLeft+x-canvas.offsetLeft)/replay.zoom,worldY:(scroll.scrollTop+y-canvas.offsetTop)/replay.zoom};
    }
    function update(){
      if(!replay.active||!replay.model||scroll.hidden)return;
      canvas.dataset.replayLayout=replay.layout;
      const width=canvas.offsetWidth,height=canvas.offsetHeight;
      if(!width||!height||!scroll.clientWidth||!scroll.clientHeight)return;
      if(replay.zoomMode==='fit')replay.updateFitZoom(Math.min((scroll.clientWidth-16)/width,(scroll.clientHeight-16)/height,3));
      const zoom=replay.zoom,scaledWidth=width*zoom,scaledHeight=height*zoom;
      surface.style.width=`${Math.max(scroll.clientWidth,scaledWidth)}px`;
      surface.style.height=`${Math.max(scroll.clientHeight,scaledHeight)}px`;
      canvas.style.left=`${Math.max(0,(scroll.clientWidth-scaledWidth)/2)}px`;
      canvas.style.top=`${Math.max(0,(scroll.clientHeight-scaledHeight)/2)}px`;
      canvas.style.transform=`scale(${zoom})`;
      if(lastLayout!==replay.layout||replay.zoomMode==='fit'){scroll.scrollLeft=0;scroll.scrollTop=0;}
      else if(anchor){scroll.scrollLeft=canvas.offsetLeft+anchor.worldX*zoom-anchor.x;scroll.scrollTop=canvas.offsetTop+anchor.worldY*zoom-anchor.y;}
      anchor=null;lastLayout=replay.layout;
      const values=[...new Set([...percentages,zoom])].sort((a,b)=>a-b);
      controls.querySelector('select').innerHTML=values.map(v=>`<option value="${v}" ${v===zoom?'selected':''}>${Math.round(v*100)}%</option>`).join('');
      controls.querySelector('[data-network-zoom="out"]').disabled=zoom<=.1;
      controls.querySelector('[data-network-zoom="in"]').disabled=zoom>=3;
      controls.querySelector('[data-network-zoom="fit"]').setAttribute('aria-pressed',String(replay.zoomMode==='fit'));
      onRedraw();
    }
    function schedule(){cancelAnimationFrame(raf);raf=requestAnimationFrame(update);}
    function onClick(e){
      const button=e.target.closest('[data-network-zoom]');if(!button)return;
      anchor=capture();
      switch(button.dataset.networkZoom){
        case'in':replay.setView('zoom',Math.round(replay.zoom*1.25*1000)/1000);break;
        case'out':replay.setView('zoom',Math.round(replay.zoom/1.25*1000)/1000);break;
        case'fit':anchor=null;replay.setView('fit');break;
        case'reset':anchor=null;replay.setView('zoom',1);scroll.scrollLeft=0;scroll.scrollTop=0;break;
      }
    }
    function onChange(e){if(e.target.dataset.replayView==='zoom'){e.stopPropagation();anchor=capture();replay.setView('zoom',e.target.value);}}
    function onWheel(e){
      if(!e.ctrlKey&&!e.metaKey)return;
      e.preventDefault();anchor=capture(e.clientX,e.clientY);
      const pixels=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?scroll.clientHeight:1);
      replay.setView('zoom',Math.round(replay.zoom*Math.exp(-Math.max(-200,Math.min(200,pixels))*.005)*1000)/1000);
    }
    function onDown(e){
      if(e.pointerType!=='mouse'||e.button!==0||e.target.closest('.neuron,line,button,select'))return;
      drag={id:e.pointerId,x:e.clientX,y:e.clientY,left:scroll.scrollLeft,top:scroll.scrollTop};
      scroll.setPointerCapture(e.pointerId);scroll.classList.add('is-panning');e.preventDefault();
    }
    function onMove(e){if(drag&&drag.id===e.pointerId){scroll.scrollLeft=drag.left+drag.x-e.clientX;scroll.scrollTop=drag.top+drag.y-e.clientY;}}
    function onUp(e){if(drag&&drag.id===e.pointerId){drag=null;scroll.classList.remove('is-panning');if(scroll.hasPointerCapture(e.pointerId))scroll.releasePointerCapture(e.pointerId);}}
    controls.addEventListener('click',onClick);controls.addEventListener('change',onChange);
    scroll.addEventListener('wheel',onWheel,{passive:false});
    scroll.addEventListener('pointerdown',onDown);scroll.addEventListener('pointermove',onMove);scroll.addEventListener('pointerup',onUp);scroll.addEventListener('pointercancel',onUp);scroll.addEventListener('lostpointercapture',onUp);
    const observer=new ResizeObserver(schedule);observer.observe(scroll);observer.observe(canvas);
    return {update,schedule,destroy(){
      observer.disconnect();cancelAnimationFrame(raf);
      scroll.removeEventListener('wheel',onWheel);scroll.removeEventListener('pointerdown',onDown);scroll.removeEventListener('pointermove',onMove);scroll.removeEventListener('pointerup',onUp);scroll.removeEventListener('pointercancel',onUp);scroll.removeEventListener('lostpointercapture',onUp);
      controls.remove();surface.replaceWith(canvas);canvas.style.left='';canvas.style.top='';delete canvas.dataset.replayLayout;scroll.classList.remove('is-panning');
    }};
  }
  return {create};
})();
