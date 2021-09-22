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

#ifndef DAISYKIT_PROCESSORS_FITNESS_PUSHUPS_PUSHUP_ANALYZER_H_
#define DAISYKIT_PROCESSORS_FITNESS_PUSHUPS_PUSHUP_ANALYZER_H_

#include "daisykitsdk/common/logging/mjpeg_server.h"

#include <memory>
#include <opencv2/opencv.hpp>

namespace daisykit {
namespace processors {

/// Push-up analyzer using image and signal processing
class PushupAnalyzer {
 public:
  PushupAnalyzer();
  int CountPushups(const cv::Mat& rgb, bool is_pushing_up);

 private:
  int CountWithNewPoint(double data, bool is_pushing_up);
  double CalcOpticalFlow(const cv::Mat& img);
  std::shared_ptr<logging::MJPEGServer> debug_img_server_;
  cv::Mat prvs_;
  std::vector<double> input_ = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<bool> pushing_;
  bool is_first_frame_ = true;
  int count_interval_ = 50;  // ms
  int current_count_ = 0;
  long long int last_count_time_ = 0;
};

}  // namespace processors
}  // namespace daisykit

#endif
