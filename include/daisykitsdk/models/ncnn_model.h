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

#ifndef DAISYKIT_MODELS_NCNN_MODEL_H_
#define DAISYKIT_MODELS_NCNN_MODEL_H_

#include <net.h>

#include <string>

namespace daisykit {
namespace models {

/// Base class for NCNN models.
class NCNNModel {
 public:
  /// Initialize a NCNN model.
  NCNNModel(bool use_gpu = false);

  /// Initialize a NCNN from buffers.
  NCNNModel(const char* param_buffer, const unsigned char* weight_buffer,
            bool use_gpu = false);

  /// Initialize a NCNN model from files.
  NCNNModel(const std::string& param_file, const std::string& weight_file,
            bool use_gpu = false);

  /// Load model from buffers.
  /// Module IO should be used to provide inferfaces for loading assets and do
  /// some middleware processing.
  int LoadModel(const char* param_buffer, const unsigned char* weight_buffer);

  /// Load model from param and weight file.
  /// This function only works if file access is supported. Be careful when use
  /// it for multiplatform applications. Instead, use IO module for loading
  /// models from different sources.
  int LoadModel(const std::string& param_file, const std::string& weight_file);

  /// Prediction function for NCNN model with 1 input and 1 output.
  /// Return 0 on success, otherwise return error code.
  int Predict(const ncnn::Mat& in, ncnn::Mat& out,
              const std::string& input_name, const std::string& output_name);

 protected:
  ncnn::Net model_;
  ncnn::Mutex model_mutex_;  // Mutex for limiting to only 1 inference stream.
  bool use_gpu_;             // Turn on GPU support
};

}  // namespace models
}  // namespace daisykit

#endif
