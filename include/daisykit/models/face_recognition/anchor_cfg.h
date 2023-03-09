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

#ifndef DAISYKIT_MODELS_FACE_RECOGNITION_ANCHOR_CFG_H_
#define DAISYKIT_MODELS_FACE_RECOGNITION_ANCHOR_CFG_H_

#include <stdio.h>
#include <map>
#include <vector>
namespace daisykit {
namespace models {

class AnchorCfg {
 public:
  std::vector<float> scales_;
  std::vector<float> ratios_;
  int base_size_;

  AnchorCfg();
  AnchorCfg(const std::vector<float> s, const std::vector<float> r, int size);
};
extern std::vector<int> feat_stride_fpn_;
extern std::map<int, AnchorCfg> anchor_cfg_;
extern bool dense_anchor_;

}  // namespace models
}  // namespace daisykit
#endif
