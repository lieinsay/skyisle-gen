// 运行：node tests/river_probe.test.cjs；测试实际页面函数，不启动浏览器。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../skyisle_gen/web/static/island.html'), 'utf8');
new vm.Script(source.split('<script>')[1].split('</script>')[0]);
const panel = {textContent:'',innerHTML:'',scrollIntoView(){}};
const c = vm.createContext({Math, Number, Infinity,
  D:{rivers:{basins:[{index:[1,4]},{rest:true,index:[0.5,2]}]},island_cfg:{hydro:{at_station_width_b:0.26}}},
  day:0,dayTab:{aMin:[1,1],frozen:[false,false],frozenS:[false,false]},
  riverProbe:null,R:{step:2,res_m:200,lines:[]},view:{s:4,tx:0,ty:0},
  $:()=>panel,fmt:(x,n)=>x.toFixed(n),vecOn:()=>true,draw(){}});
vm.runInContext(source.slice(source.indexOf('function smooth(t)'),source.indexOf('\n',source.indexOf('function smooth(t)'))),c);
vm.runInContext(source.slice(source.indexOf('// 日水面与点击探针'),source.indexOf('function flyToCell')),c);
const p=[10,10,15,1,8,10,2], L={basin:0,island:1,lvl:1,amax:8,bb:[10,10,10,20],pts:[p,[10,20,15,1,9,10,2]]};
assert.equal(c.riverWaterAt(L,p).width,10);
c.day=1;
assert.ok(c.riverWaterAt(L,p).width>10);
assert.equal(c.riverWaterAt(L,p).q,8);
assert.equal(p[2],15); // 日期不改变河槽。
assert.ok(c.riverWaterAt({basin:-1},p).width>10); // 小流域回退。
assert.equal(c.riverWaterAt(L,[10,10,1,0,0.3,1,0.1]).width,0);
assert.equal(c.riverWaterAt(L,[10,10,1,0,0.3,1,0.1]).q,0);
c.R.lines=[L]; c.pickRiver(30,20);
assert.equal(c.riverProbe.L,L);
assert.equal(c.riverProbe.p[1],15); // 抽稀栅格上的中心线中点。
assert.equal(c.riverProbe.p[4],8.5);
c.pickRiver(500,500);
assert.equal(c.riverProbe.p[1],15); // 空白处不误选。
c.renderRiverProbe(); assert.match(panel.innerHTML,/第 1 日/);
c.riverProbe=null; c.renderRiverProbe(); assert.match(panel.textContent,/点击地图/);
console.log('河道探针：日期变化、固定河槽、断流、rest 流域、抽稀坐标命中与清除均通过');
