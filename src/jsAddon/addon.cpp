#include <nan.h>
#include "AnnWrapper.hpp"

using namespace v8;

void InitAll(Handle<Object> exports) {
  AnnWrapper::Init(exports);
}

NODE_MODULE(addon, InitAll)
