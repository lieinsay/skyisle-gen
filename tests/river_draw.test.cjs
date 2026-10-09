// 运行：node tests/river_draw.test.cjs；验证真实比例在远景下不丢连接段。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../skyisle_gen/web/static/island.html'), 'utf8');
const strokes = [];
const canvas = {canvas:{width:200,height:200},lineWidth:1,strokeStyle:'',
  beginPath(){this.points=[];},moveTo(x,y){this.points.push([x,y]);},
  lineTo(x,y){this.points.push([x,y]);},stroke(){strokes.push({width:this.lineWidth,points:this.points});}};
const c = vm.createContext({Math,day:0,dayTab:null,liveDraw:()=>false,
  $:()=>({checked:true}),view:{s:1,tx:10,ty:10},
  R:{step:1,res_m:100,lines:[{lvl:1,amax:20,bb:[0,0,0,20],
    pts:[[0,0,10,1,10],[0,10,15,1,20],[0,20,20,1,30]]}]}});
vm.runInContext(source.slice(source.indexOf('function riverMinPx('),source.indexOf('// 日水面与点击探针')),c);
const savedWidth = c.R.lines[0].pts.map(p=>p[2]);
c.drawRivers(canvas);
assert.ok(strokes.length > 0);
assert.ok(strokes.every(s=>s.width>0 && s.points.length>=2));
assert.equal(strokes[0].points[0][0],10);
assert.equal(strokes.at(-1).points.at(-1)[0],30);
assert.deepEqual(c.R.lines[0].pts.map(p=>p[2]),savedWidth);
assert.equal(strokes[0].width,.15); // 远景抽稀跨整个段，平均 15 m；修复前取整到 0.25。
strokes.length=0;
c.view.s=.5;
c.drawRivers(canvas);
assert.ok(strokes.length>0 && strokes.every(s=>s.width>0)); // 修复前 0.12 取整到 0。
console.log('真实比例河道：亚像素连接保留，宽度不量化为零，物理宽度未修改');
