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
Header for class Base Model, the abstract class for all models.
All models should be inherited from this class to guarantee
the operation of loading model, predicting model and other basics.
*/

#ifndef DAISYKIT_MODELS_BASE_MODEL_H_
#define DAISYKIT_MODELS_BASE_MODEL_H_

#include <net.h>

#include <string>

namespace daisykit {
namespace models {

template <typename TInput, typename TOutput>
class BaseModel {
 public:
  // No abstract the constructor function for arbitary initialization
  virtual ~BaseModel();

  // Only load model from buffer to avoid depending on any format
  // in different platforms. Use module io to convert model to appropriate
  // format.
  virtual int LoadModel(const char* param_buffer,
                        const unsigned char* weight_buffer);

  // Will be deleted when IO module is supported. Keep for old compatibility.
  virtual int LoadModel(const std::string& param_file,
                        const std::string& weight_file);

  // Because there are many kinds of data (exp: image, voice,...).
  // Inherited models should decide what data type they process by themselves.
  virtual TOutput Predict(const TInput& input);

 protected:
  ncnn::Net model_;
  ncnn::Mutex lock_;
};

}  // namespace models
}  // namespace daisykit

#endif
