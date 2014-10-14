cmd_Release/hello.node := ln -f "Release/obj.target/hello.node" "Release/hello.node" 2>/dev/null || (rm -rf "Release/hello.node" && cp -af "Release/obj.target/hello.node" "Release/hello.node")
