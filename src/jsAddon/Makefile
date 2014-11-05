phony: build/Release/addon.node

build/Release/addon.node:
	node-gyp configure
	node-gyp build

test:
	node addon.js
