/*
Implementation for class Base Model, the abstract class for all models.
All models should be inherited from this class to guarantee
the operation of loading model, predicting model and other basics.
*/

#include "daisykitsdk/models/base_model.h"

using namespace daisykit::models;

template <typename TInput, typename TOutput>
BaseModel<TInput, TOutput>::~BaseModel() {
  if (model_) {
    model_.clear();
  }
}

template <typename TInput, typename TOutput>
int BaseModel<TInput, TOutput>::LoadModel(const char* param_buffer,
                                          const unsigned char* weight_buffer) {
  if (model_) {
    model_.clear();
  }
  if (model_.load_param_mem(param_buffer) != 0 ||
      model_.load_model(weight_buffer) != 0) {
    // Return -1 allows developers to use backup solution
    // when model file not found or ruined.
    // exit(1) crash the application before the developer can use any backup.
    return -1;
  }
  return 0;
}