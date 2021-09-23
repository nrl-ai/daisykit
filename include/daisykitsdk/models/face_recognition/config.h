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

#ifndef FD_CONFIG_
#define FD_CONFIG_

#include <stdio.h>
#include <map>
#include <vector>

extern float pixel_mean[3];
extern float pixel_std[3];
extern float pixel_scale;

class AnchorCfg {
 public:
  std::vector<float> SCALES;
  std::vector<float> RATIOS;
  int BASE_SIZE;

  AnchorCfg() {}
  ~AnchorCfg() {}
  AnchorCfg(const std::vector<float> s, const std::vector<float> r, int size) {
    SCALES = s;
    RATIOS = r;
    BASE_SIZE = size;
  }
};

#define fmc 3
extern std::vector<int> _feat_stride_fpn;
extern std::map<int, AnchorCfg> anchor_cfg;

extern bool dense_anchor;
extern float cls_threshold;

#endif  // FD_CONFIG
