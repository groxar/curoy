#include "AnnWrapper.hpp"
#include "nanodbc.h"
#include "../wt/WaveletTransformator.hpp"
#include "../wt/WaveletReturn.hpp"
#include "../wt/Filter.hpp"
#include <string>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <node/node_buffer.h>

using namespace v8;
using namespace curoy;
using namespace node;

Persistent<Function> AnnWrapper::constructor;

xMatrix<double> getColumn(nanodbc::result input, size_t column){
  vector<double> vec;
  double* data;
  double prevValue;
  double crntValue;

  input.next();
  prevValue = stod(input.get<string>(column));
  while(input.next()){
    crntValue = stod(input.get<string>(column));
    vec.push_back(crntValue-prevValue);
    prevValue = crntValue;
  }

  data = (double*) malloc(sizeof(double)* vec.size());

  memcpy(data,&vec[0],vec.size()*sizeof(double));
  return xMatrix<double>(data,{1,vec.size()},memPermission::owner);
}

AnnWrapper::AnnWrapper(size_t numFeatures, size_t numPossibleOutputs, size_t numNeurons, double epsilon) {
  net = ann(numFeatures, numPossibleOutputs, vector<size_t>({numNeurons}), epsilon);
  value_ = 0;
}

AnnWrapper::~AnnWrapper() {
}

void AnnWrapper::Init(Handle<Object> exports) {
  NanScope();

  // Prepare constructor template
  Local<FunctionTemplate> tpl = NanNew<FunctionTemplate>(New);
  tpl->SetClassName(NanNew("AnnWrapper"));
  tpl->InstanceTemplate()->SetInternalFieldCount(4);

  // Prototype
  NODE_SET_PROTOTYPE_METHOD(tpl, "train", Train);
  NODE_SET_PROTOTYPE_METHOD(tpl, "predict", Predict);

  NanAssignPersistent(constructor, tpl->GetFunction());
  exports->Set(NanNew("AnnWrapper"), tpl->GetFunction());
}

NAN_METHOD(AnnWrapper::New) {
  NanScope();

  if (args.IsConstructCall()) {
    // Invoked as constructor: `new AnnWrapper(...)`
    AnnWrapper* obj = new AnnWrapper( args[0]->IntegerValue(),
                                      args[1]->IntegerValue(),
                                      args[2]->IntegerValue(),
                                      args[3]->NumberValue());
    obj->Wrap(args.This());

    //get Data
    nanodbc::connection connection("hana", "");
    cout << "Connected with driver " << connection.driver_name() << endl;
    nanodbc::result results;
    results = execute(connection, "select * from \"DING_TECO\".\"nobel2\" where deviceid = 12486 order by time");
    auto x = getColumn(results,2);
    x >> obj->x;
    obj->position = 0;
    NanReturnValue(args.This());
  } else {
    // Invoked as plain function `AnnWrapper(...)`, turn into construct call.
    const int argc = 4;
    Local<Value> argv[4] = {  args[0],
                              args[1],
                              args[2],
                              args[3]};
    Local<Function> cons = NanNew<Function>(constructor);
    NanReturnValue(cons->NewInstance(argc, argv));
  }
}


NAN_METHOD(AnnWrapper::Predict) {
  NanScope();

  AnnWrapper* obj = ObjectWrap::Unwrap<AnnWrapper>(args.Holder());

  cout <<"Prediction Input: "<< cuMatrix<double>((obj->x.m_data)+obj->position,{1,24},memPermission::user) << endl;
  cout <<"Prediction Output: "<< obj->net.predict(cuMatrix<double>((obj->x.m_data)+obj->position,{1,24},memPermission::user)) << endl;

  NanReturnValue(NanNew(obj->value_));
}

NAN_METHOD(AnnWrapper::Train) {
  NanScope();

  AnnWrapper* obj = ObjectWrap::Unwrap<AnnWrapper>(args.Holder());
  xMatrix<double> XTrain;
  xMatrix<double> YTrain;
  cuMatrix<double> cuXTrain;
  cuMatrix<double> cuYTrain;

  WaveletReturn* transformedData;
  WaveletTransformator transformator;

  XTrain << cuMatrix<double>((obj->x.m_data)+obj->position,{1,24},memPermission::user);
  cuYTrain = cuMatrix<double>((obj->x.m_data)+obj->position+24,{1,1},memPermission::user);

  transformedData = transformator.waveletDecomposition(XTrain.m_data, 23, 3, "sym4");
  XTrain = xMatrix<double>(transformedData->data,{1,24},memPermission::owner);
  XTrain >> cuXTrain;

  cout <<"cuXTrain: "<< cuXTrain << endl;
  cout <<"cuYTrain: "<< cuYTrain << endl;

  obj->net.gradientDescent(cuXTrain,cuYTrain,0.2,0,3);

  obj->position = (obj->position + 1 ) % (obj->x.dim(1)-25);

  NanReturnValue(NanNew(obj->value_));
}
