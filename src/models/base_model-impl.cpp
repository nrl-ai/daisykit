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
Follow
https://isocpp.org/wiki/faq/templates#separate-template-class-defn-from-decl to
enable implement codes of templated abstract class be put in a .cpp file.
*/

/*
New implemented BaseModel need to be listed here
*/

#include "base_model.cpp"
#include "daisykitsdk/common/types.h"

#include <opencv2/opencv.hpp>

namespace daisykit {
namespace models {

// Belows for template implementation
template class BaseModel<cv::Mat, std::vector<types::Object>>;
template class BaseModel<cv::Mat, std::vector<types::Face>>;
template class BaseModel<cv::Mat, std::vector<types::Keypoint>>;
template class BaseModel<cv::Mat, cv::Mat>;

}  // namespace models
}  // namespace daisykit
