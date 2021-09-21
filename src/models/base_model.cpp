// Copyright 2021 The DaisyKit Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
Implementation for class Base Model, the abstract class for all models.
All models should be inherited from this class to guarantee
the operation of loading model, predicting model and other basics.
*/

#include "daisykitsdk/models/base_model.h"

#include <string>

namespace daisykit {
namespace models {

template <typename TInput, typename TOutput>
BaseModel<TInput, TOutput>::~BaseModel() {
  // https://github.com/VNOpenAI/daisykit/commit/89fbf2fcf34c75662c23d5f48abc3a538fae7e93#r56348806
  model_.clear();
}

template <typename TInput, typename TOutput>
int BaseModel<TInput, TOutput>::LoadModel(const char* param_buffer,
                                          const unsigned char* weight_buffer) {
  // https://github.com/VNOpenAI/daisykit/commit/89fbf2fcf34c75662c23d5f48abc3a538fae7e93#r56348806
  model_.clear();
  if (model_.load_param_mem(param_buffer) != 0 ||
      model_.load_model(weight_buffer) != 0) {
    // Return -1 allows developers to use backup solution
    // when model file not found or ruined.
    // Don't use exit(1) which crash the application
    // before the developer can use any backup.
    return -1;
  }
  return 0;
}

// Will be deleted when IO module is supported. Keep for old compatibility.
template <typename TInput, typename TOutput>
int BaseModel<TInput, TOutput>::LoadModel(const std::string& param_file,
                                          const std::string& weight_file) {
  model_.clear();
  if (model_.load_param(param_file.c_str()) != 0 ||
      model_.load_model(weight_file.c_str()) != 0) {
    return -1;
  }
  return 0;
}

// Subclasses need to overrid this function
template <typename TInput, typename TOutput>
TOutput BaseModel<TInput, TOutput>::Predict(const TInput& input) {}

}  // namespace models
}  // namespace daisykit
