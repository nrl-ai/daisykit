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

#include "daisykitsdk/processors/fitness/pushup_analyzer.h"
#include "daisykitsdk/common/visualizers/base_visualizer.h"
#include "daisykitsdk/processors/signal_processors/signal_smoothing.h"
#include "daisykitsdk/processors/signal_processors/z_score_filter.h"

namespace daisykit {
namespace processors {

PushupAnalyzer::PushupAnalyzer() {
  debug_img_server_ = std::make_shared<logging::MJPEGServer>(8080);
}

int PushupAnalyzer::CountWithNewPoint(double data, bool is_pushing_up) {
  input_.push_back(data);
  pushing_.push_back(is_pushing_up);
  std::vector<double> signal =
      processors::SignalSmoothing::MeanFilter1D(input_);

  int range[2] = {0, 300};

  std::vector<int> filtered_signals = processors::ZScoreFilter::Filter(signal);

  cv::Mat line_graph =
      visualizers::BaseVisualizer::LineGraph(signal, range, 500);

  std::vector<int> countPostions;
  for (int i = 0; i < (int)filtered_signals.size() - 1; ++i) {
    if (filtered_signals[i] == 1 && filtered_signals[i + 1] == 0 &&
        i < pushing_.size() && pushing_[i]) {
      countPostions.push_back((int)i);
    }
  }

  if (is_first_frame_) {
    debug_img_server_->Start();
    is_first_frame_ = false;
  } else {
    debug_img_server_->WriteFrame(line_graph);
  }

  return (int)countPostions.size();
}

int PushupAnalyzer::CountPushups(const cv::Mat& rgb, bool is_pushing_up) {
  long long int current_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (current_time - last_count_time_ < count_interval_) {
    // Skip counting
    return current_count_;
  } else {
    last_count_time_ = current_time;
  }

  double x = CalcOpticalFlow(rgb);

  int count = CountWithNewPoint(x, is_pushing_up);
  current_count_ = count;

  return current_count_;
}

double PushupAnalyzer::CalcOpticalFlow(const cv::Mat& frame2) {
  cv::Mat next;
  cv::resize(frame2, next, cv::Size(224, 224));
  cv::cvtColor(next, next, cv::COLOR_BGR2GRAY);
  if (prvs_.empty()) {
    prvs_ = next;
  }
  cv::Mat flow(prvs_.size(), CV_32FC2);
  calcOpticalFlowFarneback(prvs_, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

  cv::Mat flow_parts[2];
  cv::split(flow, flow_parts);
  cv::Mat magnitude, angle;
  cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

  angle = angle * magnitude;

  double avg_angle = cv::sum(angle)[0];
  avg_angle /= angle.rows * angle.cols;
  prvs_ = next;

  return avg_angle;
}

}  // namespace processors
}  // namespace daisykit
