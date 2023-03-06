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

#include "daisykit/models/face_recognition/config.h"

namespace daisykit {
namespace models {
AnchorCfg ::AnchorCfg() {}
AnchorCfg::AnchorCfg(const std::vector<float> s, const std::vector<float> r,
                     int size) {
  scales_ = s;
  ratios_ = r;
  base_size_ = size;
}
}  // namespace models
}  // namespace daisykit
