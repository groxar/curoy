#pragma once

#include "../ml/ann.hpp"
#include "../lib/cuMatrix.hpp"
#include <nan.h>
#include <v8.h>
#include <node.h>


class AnnWrapper : public node::ObjectWrap {
 public:
  static void Init(v8::Handle<v8::Object> exports);

 private:
  explicit AnnWrapper(size_t numFeatures, size_t numPossibleOutputs, size_t numNeurons, double epsilon);
  ~AnnWrapper();

  static NAN_METHOD(New);
  static NAN_METHOD(Predict);
  static NAN_METHOD(Train);
  static v8::Persistent<v8::Function> constructor;
  size_t position;
  double value_;
  curoy::ann net;
  curoy::cuMatrix<double> x;
};
