var addon = require('bindings')('addon');

var obj = new addon.AnnWrapper(10);
console.log( obj.plusOne() ); // 11
console.log( obj.plusOne() ); // 12
console.log( obj.plusOne() ); // 13
