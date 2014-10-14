#include "AnnWrapper.hpp"
#include "../ml/ann.hpp"

using namespace v8;

Persistent<Function> AnnWrapper::constructor;

AnnWrapper::AnnWrapper(double value) : value_(value) {
}

AnnWrapper::~AnnWrapper() {
}

void AnnWrapper::Init(Handle<Object> exports) {
  NanScope();

  // Prepare constructor template
  Local<FunctionTemplate> tpl = NanNew<FunctionTemplate>(New);
  tpl->SetClassName(NanNew("AnnWrapper"));
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  // Prototype
  NODE_SET_PROTOTYPE_METHOD(tpl, "plusOne", PlusOne);

  NanAssignPersistent(constructor, tpl->GetFunction());
  exports->Set(NanNew("AnnWrapper"), tpl->GetFunction());
}

NAN_METHOD(AnnWrapper::New) {
  NanScope();

  if (args.IsConstructCall()) {
    // Invoked as constructor: `new AnnWrapper(...)`
    double value = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
    AnnWrapper* obj = new AnnWrapper(value);
    obj->Wrap(args.This());
    NanReturnValue(args.This());
  } else {
    // Invoked as plain function `AnnWrapper(...)`, turn into construct call.
    const int argc = 1;
    Local<Value> argv[argc] = { args[0] };
    Local<Function> cons = NanNew<Function>(constructor);
    NanReturnValue(cons->NewInstance(argc, argv));
  }
}

NAN_METHOD(AnnWrapper::PlusOne) {
  NanScope();

  AnnWrapper* obj = ObjectWrap::Unwrap<AnnWrapper>(args.Holder());
  obj->value_ += 1;

  NanReturnValue(NanNew(obj->value_));
}
