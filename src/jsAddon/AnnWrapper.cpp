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
  NODE_SET_PROTOTYPE_METHOD(tpl, "predict", Predict);
  NODE_SET_PROTOTYPE_METHOD(tpl, "train", Train);
  NODE_SET_PROTOTYPE_METHOD(tpl, "nextValue", NextValue);
  NODE_SET_PROTOTYPE_METHOD(tpl, "getNeuronLayer", GetNeuronLayer);

  NanAssignPersistent(constructor, tpl->GetFunction());
  exports->Set(NanNew("AnnWrapper"), tpl->GetFunction());
}

NAN_METHOD(AnnWrapper::New) {
  NanScope();

  if (args.IsConstructCall()) {
    AnnWrapper* obj = new AnnWrapper( args[0]->IntegerValue(),
                                      args[1]->IntegerValue(),
                                      args[2]->IntegerValue(),
                                      args[3]->NumberValue());
    obj->Wrap(args.This());
    cout << "#Inputs"<< (obj->net.hiddenLayerVec[0]).dim(0)-1<<endl;
    //get Data
    nanodbc::connection connection( string(*v8::String::Utf8Value(args[4]->ToString())),
                                    string(*v8::String::Utf8Value(args[5]->ToString())),
                                    string(*v8::String::Utf8Value(args[6]->ToString()))
                                  );
    cout << "Connected with driver " << connection.driver_name() << endl;
    nanodbc::result results;
    results = execute(connection, "select * from \"nobel2\" where deviceid = 15789 order by time");
    auto x = getColumn(results,2);
    x >> obj->x;
    obj->position = 0;
    NanReturnValue(args.This());
  } else {
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
  xMatrix<double> XTrain;
  cuMatrix<double> cuXTrain;

  WaveletReturn* transformedData;
  WaveletTransformator transformator;

  AnnWrapper* obj = ObjectWrap::Unwrap<AnnWrapper>(args.Holder());

  XTrain << cuMatrix<double>((obj->x.m_data)+obj->position,{1,((obj->net.hiddenLayerVec[0]).dim(0)-1)},memPermission::user);

  transformedData = transformator.waveletDecomposition(XTrain.m_data, (((obj->net.hiddenLayerVec)[0].dim(0))-1), 3, "sym4");
  XTrain = xMatrix<double>(transformedData->data,{1,((obj->net.hiddenLayerVec[0]).dim(0)-1)},memPermission::owner);
  XTrain >> cuXTrain;

  double prediction = ~(obj->net.predict(cuXTrain));
  cout <<"Prediction Input: "<< cuMatrix<double>((obj->x.m_data)+obj->position,{1,((obj->net.hiddenLayerVec[0]).dim(0)-1)},memPermission::user) << endl;
  cout <<"Prediction Output: "<< prediction << endl;
  cout <<"Real       Output: "<< cuMatrix<double>((obj->x.m_data)+obj->position+((obj->net.hiddenLayerVec[0]).dim(0)-1),{1,1},memPermission::user)<<endl;

  NanReturnValue(NanNew(prediction));
}

NAN_METHOD(AnnWrapper::NextValue) {
  NanScope();

  AnnWrapper* obj = ObjectWrap::Unwrap<AnnWrapper>(args.Holder());
  double result = ~(cuMatrix<double>((obj->x.m_data)+obj->position+((obj->net.hiddenLayerVec[0]).dim(0)-1),{1,1},memPermission::user));
  obj->position = (obj->position + 1 ) % (obj->x.dim(1)-(((obj->net.hiddenLayerVec[0]).dim(0)-1)+1));

  NanReturnValue(NanNew(result));
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

  XTrain << cuMatrix<double>((obj->x.m_data)+obj->position,{1,((obj->net.hiddenLayerVec[0]).dim(0)-1)},memPermission::user);
  cuYTrain = cuMatrix<double>((obj->x.m_data)+obj->position+((obj->net.hiddenLayerVec[0]).dim(0)-1),{1,1},memPermission::user);

  transformedData = transformator.waveletDecomposition(XTrain.m_data, ((obj->net.hiddenLayerVec)[0].dim(0)-1), 3, "sym4");
  XTrain = xMatrix<double>(transformedData->data,{1,((obj->net.hiddenLayerVec[0]).dim(0)-1)},memPermission::owner);
  XTrain >> cuXTrain;

  obj->net.gradientDescent(cuXTrain,cuYTrain,args[0]->NumberValue(),args[1]->NumberValue(),args[2]->IntegerValue());


  NanReturnValue(NanNew(obj->value_));
}

NAN_METHOD(AnnWrapper::GetNeuronLayer) {
  NanScope();

  AnnWrapper* obj = ObjectWrap::Unwrap<AnnWrapper>(args.Holder());
  xMatrix<double> layer;
  layer << obj->net.hiddenLayerVec[0];
  const double* data = layer.m_data;
  cout << layer.m_data[0]<<endl;
  cout << layer.m_data[1]<<endl;
  cout <<"size: "<< layer.size() << endl;
  cout << layer.m_data[layer.size()-1]<<endl;

  size_t length = sizeof(double) * layer.size();

  node::Buffer *buffer = node::Buffer::New(length);

  memcpy(node::Buffer::Data(buffer), data, length);
  NanReturnValue(buffer->handle_);
}
