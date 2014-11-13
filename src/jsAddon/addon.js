var addon = require('bindings')('addon');

var obj = new addon.AnnWrapper(24,1,100,0.12,"hana","DING_TECO","DINGteco1234!");
var i;
for(i = 0; i < 10;i++){
  obj.train(0.3,0,3);
  obj.predict();
}
buff = obj.getNeuronLayer();
console.log("Buffer read");
console.log(buff.readDoubleLE(0));
console.log(buff.readDoubleLE(8));
console.log("Last buffer read");
console.log(buff.readDoubleLE(2499*8));
