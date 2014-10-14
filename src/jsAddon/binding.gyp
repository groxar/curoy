{
  "targets": [
    {
      "target_name": "addon",
      "sources": [ "addon.cpp", "AnnWrapper.cpp" ],
      "cflags": ["-std=c++11","-I/usr/local/cuda/include"],
      "include_dirs": [
        "<!(node -e \"require('nan')\")"
      ]
    }
  ]
}
