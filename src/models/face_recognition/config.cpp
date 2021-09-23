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

#include "daisykitsdk/models/face_recognition/config.h"

float pixel_mean[3] = {0, 0, 0};
float pixel_std[3] = {1, 1, 1};
float pixel_scale = 1.0;

#if fmc == 3
std::vector<int> _feat_stride_fpn = {32, 16, 8};
std::map<int, AnchorCfg> anchor_cfg = {
    {32, AnchorCfg(std::vector<float>{32, 16}, std::vector<float>{1}, 16)},
    {16, AnchorCfg(std::vector<float>{8, 4}, std::vector<float>{1}, 16)},
    {8, AnchorCfg(std::vector<float>{2, 1}, std::vector<float>{1}, 16)}};
#endif

bool dense_anchor = false;
float cls_threshold = 0.8;
