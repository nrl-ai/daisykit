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

#include "daisykitsdk/models/image_model.h"

#include <string>
#include <vector>

namespace daisykit {
namespace models {

ImageModel::ImageModel(int input_width, int input_height) {
  input_height_ = input_height;
  input_width_ = input_width;
}

int ImageModel::InputWidth() { return input_width_; }

int ImageModel::InputHeight() { return input_height_; }

}  // namespace models
}  // namespace daisykit
