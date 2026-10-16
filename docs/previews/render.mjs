/** Export the standalone design preview. Requires a local Playwright install. */
import {createRequire} from 'node:module'
import {fileURLToPath,pathToFileURL} from 'node:url'
import {dirname,join} from 'node:path'
import {writeFileSync} from 'node:fs'

const require=createRequire(import.meta.url)
const {chromium}=process.env.PLAYWRIGHT_MODULE_PATH
  ?require(process.env.PLAYWRIGHT_MODULE_PATH)
  :await import('playwright')
const root=dirname(fileURLToPath(import.meta.url))
const system=process.argv.includes('--review')?'review':'agent'
const scenes=system==='agent'?['diagnosis','decision','knowledge']:['overview','evidence','trace']
const browser=await chromium.launch({headless:true})
const page=await browser.newPage({viewport:{width:1680,height:1060},deviceScaleFactor:2,locale:'zh-CN',timezoneId:'Asia/Shanghai',colorScheme:'light',reducedMotion:'reduce'})
const errors=[]
page.on('pageerror',e=>errors.push(e.message))
const records=[]
for(const scene of scenes){
 await page.goto(pathToFileURL(join(root,'index.html')).href+'?scene='+scene)
 await page.evaluate(()=>document.fonts.ready)
 await page.screenshot({path:join(root,`${system}-${scene}.png`),animations:'disabled'})
 const qa=await page.evaluate(()=>{
  const app=document.querySelector('#app').getBoundingClientRect()
  const escaped=[...document.querySelectorAll('#app h1,#app h2,#app h3,#app p,#app .card,#app .answer-footer')].filter(el=>{const r=el.getBoundingClientRect();return r.bottom>app.bottom+1||r.right>app.right+1}).map(el=>({tag:el.tagName,class:el.className,text:el.textContent.trim().slice(0,30)}))
  const composer=document.querySelector('.composer-area')?.getBoundingClientRect()
  const answer=document.querySelector('.answer')?.getBoundingClientRect()
  return{canvas:[app.width,app.height],escaped,answerComposerGap:answer&&composer?Math.round(composer.top-answer.bottom):null}
 })
 records.push({scene,...qa})
 console.log(scene,JSON.stringify(qa))
 if(qa.escaped.length||(qa.answerComposerGap!==null&&qa.answerComposerGap<12)){
  throw new Error(`Layout needs attention before exporting ${scene}`)
 }
}
if(errors.length)throw new Error(errors.join('\n'))
if(process.argv.includes('--cover')){
 await page.setViewportSize({width:1920,height:1280})
 await page.goto(pathToFileURL(join(root,'cover.html')).href)
 await page.evaluate(()=>document.fonts.ready)
 await page.screenshot({path:join(root,`${system}-cover.png`),animations:'disabled'})
}
writeFileSync(join(root,'render-checks.json'),JSON.stringify({viewport:[1680,1060],pixelRatio:2,scenes:records,pageErrors:errors},null,2)+'\n')
await browser.close()
