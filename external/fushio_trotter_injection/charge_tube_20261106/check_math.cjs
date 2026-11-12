const fs=require('fs'),vm=require('vm');
const extension='/Applications/Visual Studio Code.app/Contents/Resources/app/extensions/markdown-math/dist/extension.js';
const fakeVSCode={workspace:{getConfiguration:()=>({get:(_,def)=>def}),onDidChangeConfiguration:()=>({dispose(){}})},commands:{executeCommand(){}}};
const sandbox={module:{exports:{}},require:n=>n==='vscode'?fakeVSCode:require(n),console};
vm.runInNewContext(fs.readFileSync(extension,'utf8'),sandbox,{filename:extension});
const ruler={after(){},before(){},push(){}};
const md={inline:{ruler},block:{ruler},core:{ruler},renderer:{rules:{}},use(fn,opts){fn(this,opts);return this}};
sandbox.module.exports.activate({subscriptions:[]}).extendMarkdownIt(md);
const formulas=JSON.parse(fs.readFileSync(require('path').join(__dirname,'control/formulas.json')));
for(const f of formulas){
  const html=md.renderer.rules[f.display?'math_block':'math_inline']([{content:f.tex}],0);
  if(html.includes('katex-error')||!html.includes('class="katex"'))throw Error(f.tex+'\n'+html);
}
console.log(JSON.stringify({renderer:extension,checked:formulas.length,katex_errors:0}));
