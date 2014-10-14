#pragma once

#include <nan.h>

class AnnWrapper : public node::ObjectWrap {
 public:
  static void Init(v8::Handle<v8::Object> exports);

 private:
  explicit AnnWrapper(double value = 0);
  ~AnnWrapper();

  static NAN_METHOD(New);
  static NAN_METHOD(PlusOne);
  static v8::Persistent<v8::Function> constructor;
  double value_;
};
