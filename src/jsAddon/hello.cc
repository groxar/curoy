#include <nan.h>

using namespace v8;

NAN_METHOD(Method) {
  NanScope();
  NanReturnValue(String::New("world"));
}

void Init(Handle<Object> exports) {
  exports->Set(NanNew("hello"), FunctionTemplate::New(Method)->GetFunction());
}

NODE_MODULE(hello, Init)
