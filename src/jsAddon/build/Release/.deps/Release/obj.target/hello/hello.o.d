cmd_Release/obj.target/hello/hello.o := g++ '-D_LARGEFILE_SOURCE' '-D_FILE_OFFSET_BITS=64' '-DBUILDING_NODE_EXTENSION' -I/usr/include -I/usr/include/node -I../node_modules/nan  -fPIC -Wall -Wextra -Wno-unused-parameter -pthread -m64 -O2 -fno-strict-aliasing -fno-tree-vrp -fno-omit-frame-pointer  -fno-rtti -fno-exceptions  -MMD -MF ./Release/.deps/Release/obj.target/hello/hello.o.d.raw  -c -o Release/obj.target/hello/hello.o ../hello.cc
Release/obj.target/hello/hello.o: ../hello.cc ../node_modules/nan/nan.h \
 /usr/include/node/node.h /usr/include/node/node_object_wrap.h \
 /usr/include/node/node.h /usr/include/node/node_buffer.h \
 /usr/include/node/node_version.h /usr/include/node/node_object_wrap.h
../hello.cc:
../node_modules/nan/nan.h:
/usr/include/node/node.h:
/usr/include/node/node_object_wrap.h:
/usr/include/node/node.h:
/usr/include/node/node_buffer.h:
/usr/include/node/node_version.h:
/usr/include/node/node_object_wrap.h:
