// node tests/river_curve.test.cjs — execute the viewer's actual geometry helper.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync(require('node:path').join(__dirname,'../skyisle_gen/web/static/island.html'),'utf8');
const code = html.split('// SOURCE_CURVE_BEGIN')[1].split('\n').slice(1).join('\n').split('// SOURCE_CURVE_END')[0];
const point=(r,c,q=1)=>[r,c,1,10,q,5,8,1,5,.5,101,100,.01,10,2,2,0,10,10,2,1];
const seg={pts:[point(1,1),point(1,2,2),point(2,2,3),point(3,3,4)],down:-1};
const child={pts:[point(0,0,.1),point(0,1,.2)],down:0,join:1};
const context={D:{rivers:{segments:[seg,child]},island:{raster:{res_m:100}}}};
vm.createContext(context); vm.runInContext(code,context);
const curve=context.riverEdges(seg);
assert.equal(context.riverEdges(seg),curve,'draw and picking reuse cached geometry');
for(const p of seg.pts) assert(curve.some(q=>q.every((v,i)=>Math.abs(v-p[i])<1e-10)),'preserve every source anchor and hydraulic value');
let maxTurn=0;
for(let i=1;i+1<curve.length;i++) {
  const a=curve[i-1],b=curve[i],c=curve[i+1];
  const u=[b[0]-a[0],b[1]-a[1]],v=[c[0]-b[0],c[1]-b[1]];
  maxTurn=Math.max(maxTurn,Math.acos(Math.min(1,(u[0]*v[0]+u[1]*v[1])/(Math.hypot(...u)*Math.hypot(...v)))));
}
assert(maxTurn<.35,`no grid corner: ${maxTurn}`);
const join=context.riverEdges(child).at(-1);
assert.equal(join[0],seg.pts[1][0]); assert.equal(join[1],seg.pts[1][1]);
assert.equal(join[4],.2,'tributary must not borrow trunk discharge');
assert.equal(join.sourceIndex,1);
assert(html.includes('best={seg,p,k:a.sourceIndex}'),'inspector reports source indices');
console.log(`river curve PASS: ${curve.length} samples, max turn ${maxTurn.toFixed(3)} rad`);
