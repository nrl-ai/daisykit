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

#ifndef DAISYKIT_EXAMPLES_PUSHUPS_PUSHUP_ANALYZER_H_
#define DAISYKIT_EXAMPLES_PUSHUPS_PUSHUP_ANALYZER_H_

#include "daisykitsdk/utils/logging/mjpeg_server.h"
#include "daisykitsdk/utils/signal_proc/signal_smoothing.h"
#include "daisykitsdk/utils/signal_proc/z_score_filter.h"
#include "daisykitsdk/utils/visualizers/base_visualizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace daisykit {
namespace examples {

class PushupAnalyzer {
 public:
  PushupAnalyzer();
  int CountPushups(const cv::Mat& rgb, bool is_pushing_up);

 private:
  int CountWithNewPoint(double data, bool is_pushing_up);
  double CalcOpticalFlow(const cv::Mat& img);
  std::shared_ptr<daisykit::utils::logging::MJPEGServer> debug_img_server_;
  cv::Mat prvs_;
  std::vector<daisykit::utils::signal_proc::ld> input_ = {0, 0, 0, 0,
                                                          0, 0, 0, 0};
  std::vector<bool> pushing_;
  bool is_first_frame_ = true;
  int count_interval_ = 50;  // ms
  int current_count_ = 0;
  long long int last_count_time_ = 0;
};

}  // namespace examples
}  // namespace daisykit

#endif