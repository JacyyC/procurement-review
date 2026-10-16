/* Local presentation assets: deterministic content, no services or credentials.
 * Basic line icons adapted from Feather/Lucide; see ICON-LICENSES.txt. */
const paths = {
  spark:'m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z',
  plus:'M12 5v14M5 12h14',
  chevron:'m9 5 7 7-7 7',
  down:'m6 9 6 6 6-6',
  arrow:'M7 17 17 7M7 7h10v10',
  send:'M12 19V5m-6 6 6-6 6 6',
  search:'m21 21-4.5-4.5M18 10.5a7.5 7.5 0 1 1-15 0 7.5 7.5 0 0 1 15 0',
  message:'M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z',
  check:'m5 12 4 4L19 6',
  checkCircle:'M22 11.1V12a10 10 0 1 1-5.9-9.1M22 4 12 14l-3-3',
  shield:'M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Zm-3-11 2 2 4-4',
  book:'M4 3h7a3 3 0 0 1 3 3v15a4 4 0 0 0-4-2H4V3Zm16 0h-3a3 3 0 0 0-3 3v15a4 4 0 0 1 4-2h2V3Z',
  brain:'M9.5 4a3.5 3.5 0 0 0-6 2.5 3.5 3.5 0 0 0-.5 6 4 4 0 0 0 3 6.5 3.5 3.5 0 0 0 6 2V4a2.5 2.5 0 0 0-2.5-2M14.5 4a3.5 3.5 0 0 1 6 2.5 3.5 3.5 0 0 1 .5 6 4 4 0 0 1-3 6.5 3.5 3.5 0 0 1-6 2M6 10h2m8 0h2M8 15h1m6 0h1',
  grid:'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
  list:'m3 6 1 1 2-2m-3 8 1 1 2-2m-3 8 1 1 2-2M10 6h11M10 13h11M10 20h11',
  layers:'m12 2 10 5-10 5L2 7l10-5Zm-10 10 10 5 10-5M2 17l10 5 10-5',
  folder:'M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2Z',
  file:'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Zm0 0v6h6M8 13h8M8 17h5',
  chart:'M3 3v18h18M7 14l4-4 4 2 6-8',
  activity:'M2 12h4l3-9 6 18 3-9h4',
  clock:'M12 8v4l3 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
  settings:'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M9 3H5v4L2 10v4l3 3v4h4l3 2 3-2h4v-4l3-3v-4l-3-3V3h-4l-3-2Z',
  bell:'M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4',
  dots:'M5 12h.01M12 12h.01M19 12h.01',
  pin:'m16 3 5 5-4 1-4 6-4-4 6-4 1-4Zm-7 8-6 10',
  download:'M12 3v12m-5-5 5 5 5-5M3 16v5h18v-5',
  link:'M10 13a5 5 0 0 0 7 .5l3-3a5 5 0 0 0-7-7l-2 2M14 11a5 5 0 0 0-7-.5l-3 3a5 5 0 0 0 7 7l2-2',
  copy:'M9 9h12v12H9zM15 5V3H3v12h2',
  thumb:'M7 10v11H3V10Zm0 0 5-8c2 0 3 1 2 4l-1 4h6a2 2 0 0 1 2 2l-2 8a2 2 0 0 1-2 1H7',
  tag:'M20 13 11 22 2 13V2h11l9 9-2 2ZM7 7h.01',
  calendar:'M4 5h16v16H4zM8 2v6m8-6v6M4 11h16',
  bolt:'m13 2-9 12h7l-1 8 10-13h-7l1-7Z',
  info:'M12 11v6m0-10h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
  alert:'m12 3 10 18H2L12 3Zm0 6v5m0 3h.01',
  route:'M4 4h5v5H4zM15 15h5v5h-5zM9 6h7a3 3 0 0 1 0 6H8a3 3 0 0 0 0 6h7',
  eye:'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Zm13 0a3 3 0 1 1-6 0 3 3 0 0 1 6 0',
  globe:'M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0ZM2 12h20M12 2a18 18 0 0 1 0 20 18 18 0 0 1 0-20',
  lock:'M5 10h14v12H5zM8 10V6a4 4 0 0 1 8 0v4',
  menu:'M3 5h18M3 12h18M3 19h18',
  mic:'M9 3h6v10H9zM5 10v3a7 7 0 0 0 14 0v-3M12 20v3',
  briefcase:'M3 7h18v14H3zM8 7V3h8v4M3 12h18M10 12v3h4v-3',
  node:'M9 3h6v6H9zM2 15h6v6H2zM16 15h6v6h-6zM12 9v3M5 15v-3h14v3',
  play:'m9 4 12 8-12 8V4Z',
  refresh:'M20 7V2l-3 3a9 9 0 1 0 3 11M20 2h-5',
  moon:'M21 13a9 9 0 0 1-10-10A9 9 0 1 0 21 13Z',
  external:'M15 3h6v6M21 3 10 14M11 3H3v18h18v-8',
  heart:'M20 4a5 5 0 0 1 0 8l-8 9-8-9a5 5 0 0 1 8-7 5 5 0 0 1 8-1Z',
};
function icon(name,size=18,cls='') {return `<svg class="icon ${cls}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[name]||paths.spark}"/></svg>`}
function badge(text,tone='green',dot=false){return `<span class="badge ${tone}">${dot?'<i class="status-dot"></i>':''}${text}</span>`}
function btn(text,name='',cls='',attrs=''){return `<button class="button ${cls}" ${attrs}>${name?icon(name,15):''}${text}</button>`}
function ib(name,label,attrs=''){return `<button class="icon-button" aria-label="${label}" title="${label}" ${attrs}>${icon(name)}</button>`}
function navItem(text,name,scene,active,extra=''){return `<button class="nav-item ${active?'active':''}" data-scene="${scene}">${icon(name,18)}<span>${text}</span>${extra?`<small>${extra}</small>`:''}</button>`}
function metric(label,value,delta,tone='green',extra='',spark=''){return `<div class="metric"><span class="metric-label">${label}${spark?icon(spark,15):''}</span><div class="metric-value">${value}</div><div class="metric-bottom"><span class="delta ${tone}">${delta}</span><span>${extra}</span></div></div>`}
function title(eyebrow,heading,sub='',actions=''){return `<div class="page-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${heading}</h1>${sub?`<p>${sub}</p>`:''}</div><div class="heading-actions">${actions}</div></div>`}
function person(initial='林',small=false){return `<span class="person ${small?'small':''}">${initial}</span>`}
function sourceChip(title,n=1){return `<button class="source-chip" data-notice="已定位到：${title}"><span>${n}</span>${title}${icon('external',12)}</button>`}
function lineChart({width=600,height=190,color='#168765',labels=['周一','周二','周三','周四','周五','周六','周日'],values=[52,60,44,70,58,92,80],previous=[72,88,65,100,94,108,98],max=120,fill=true}={}){
 const left=34,right=10,top=13,bottom=31,w=width-left-right,h=height-top-bottom;
 const pts=arr=>arr.map((v,i)=>[left+i*w/(arr.length-1),top+h-v/max*h]);
 const smooth=arr=>{const p=pts(arr);return p.map((c,i)=>i===0?`M${c[0]},${c[1]}`:`C${p[i-1][0]+w/(arr.length-1)*.4},${p[i-1][1]} ${c[0]-w/(arr.length-1)*.4},${c[1]} ${c[0]},${c[1]}`).join(' ')};
 const id='g'+Math.random().toString(36).slice(2,7);
 return `<svg class="line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${labels[0]}至${labels.at(-1)}的数据趋势"><defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${color}" stop-opacity=".17"/><stop offset="100%" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>${[0,.33,.66,1].map(v=>`<path d="M${left} ${top+h*v}H${width-right}" stroke="#e9edea" stroke-dasharray="3 4"/><text x="${left-9}" y="${top+h*v+4}" text-anchor="end" fill="#929b96" font-size="10">${Math.round(max*(1-v))}</text>`).join('')}${previous?`<path d="${smooth(previous)}" fill="none" stroke="#bcc7c2" stroke-width="2" stroke-dasharray="5 5"/>`:''}${fill?`<path d="${smooth(values)} L${width-right} ${top+h}H${left}Z" fill="url(#${id})"/>`:''}<path d="${smooth(values)}" fill="none" stroke="${color}" stroke-width="2.7"/>${pts(values).map((p,i)=>i===values.length-1?`<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="${color}" stroke="white" stroke-width="2"/>`:'').join('')}${labels.map((l,i)=>`<text x="${left+i*w/(labels.length-1)}" y="${height-8}" text-anchor="middle" fill="#88948e" font-size="10">${l}</text>`).join('')}</svg>`;
}
function toast(message){const el=document.getElementById('toast');el.innerHTML=icon('checkCircle',17)+message;el.classList.add('visible');clearTimeout(window.toastTimer);window.toastTimer=setTimeout(()=>el.classList.remove('visible'),2400)}
function fit(){const scale=Math.min(1,window.innerWidth/1680);const app=document.getElementById('app');app.style.transform=`scale(${scale})`;document.getElementById('viewport').style.height=1060*scale+'px'}
function bindScenes(render){document.addEventListener('click',event=>{const el=event.target.closest('[data-scene],[data-notice],[data-copy]');if(!el)return;if(el.dataset.scene){history.replaceState(null,'',`?scene=${el.dataset.scene}`);render(el.dataset.scene);fit()}else if(el.dataset.copy){navigator.clipboard?.writeText(el.dataset.copy);toast('已复制到剪贴板')}else toast(el.dataset.notice)});window.addEventListener('resize',fit)}
