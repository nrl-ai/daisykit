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

#ifndef DAISYKIT_GRAPHS_CORE_UTILS_H_
#define DAISYKIT_GRAPHS_CORE_UTILS_H_

#include "daisykitsdk/graphs/core/node.h"
#include "daisykitsdk/utils/timer.h"

#include <chrono>
#include <memory>

namespace daisykit {
namespace graphs {

void Packet2CvMat(PacketPtr packet, cv::Mat& output) {
  if (packet == nullptr) {
    output = cv::Mat();
    return;
  }
  std::shared_ptr<void> data;
  std::time_t timestamp;
  packet->GetData(data, timestamp);
  output = *std::static_pointer_cast<cv::Mat>(data);
}

void CvMat2Packet(const cv::Mat& input, PacketPtr& packet, bool copy = false) {
  utils::TimePoint timestamp = utils::Timer::GetCurrentTime();
  std::shared_ptr<cv::Mat> output_mat = std::make_shared<cv::Mat>();
  *output_mat = copy ? input.clone() : input;
  packet = std::make_shared<Packet>(std::static_pointer_cast<void>(output_mat),
                                    timestamp);
}

}  // namespace graphs
}  // namespace daisykit

#endif