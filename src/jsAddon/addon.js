var addon = require('bindings')('addon');

var obj = new addon.AnnWrapper(24,1,100,0.12);
obj.predict();
obj.train();
obj.predict();
obj.train();
obj.predict();
obj.train();
obj.predict();
obj.train();
obj.predict();
obj.train();
