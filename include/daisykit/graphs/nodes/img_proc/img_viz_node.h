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

#ifndef DAISYKIT_GRAPHS_NODES_IMG_PROC_IMG_VIZ_NODE_H_
#define DAISYKIT_GRAPHS_NODES_IMG_PROC_IMG_VIZ_NODE_H_

#include "daisykit/graphs/core/node.h"

#include <memory>
#include <mutex>
#include <thread>

namespace daisykit {
namespace graphs {
namespace nodes {

/// Image visualizer node.
/// Used for graph API development
class ImgVizNode : public Node {
 public:
  using Node::Node;  // For constructor inheritance

  void Tick() {
    // Wait for data
    WaitForData();

    // Prepare input packets
    std::map<std::string, PacketPtr> inputs;
    PrepareInputs(inputs);

    if (inputs.count("binary") > 0) {
      PacketPtr binary_input = inputs["binary"];
      cv::Mat binary = *binary_input->GetData<cv::Mat>();

      // Write to output
      std::lock_guard<std::mutex> lock(binary_draw_mutex_);
      binary_draw_ = binary;
      binary_draw_ready_ = true;
    }

    if (inputs.count("gray") > 0) {
      PacketPtr gray_input = inputs["gray"];
      cv::Mat gray = *gray_input->GetData<cv::Mat>();

      // Write to output
      std::lock_guard<std::mutex> lock(gray_draw_mutex_);
      gray_draw_ = gray;
      gray_draw_ready_ = true;
    }
  }

  bool GetOutputBinary(cv::Mat& img) {
    std::lock_guard<std::mutex> lock(binary_draw_mutex_);
    if (!binary_draw_ready_) return false;
    img = binary_draw_.clone();
    binary_draw_ready_ = false;
    return true;
  }

  bool GetOutputGray(cv::Mat& img) {
    std::lock_guard<std::mutex> lock(gray_draw_mutex_);
    if (!gray_draw_ready_) return false;
    img = gray_draw_.clone();
    gray_draw_ready_ = false;
    return true;
  }

 private:
  cv::Mat binary_draw_;
  bool binary_draw_ready_;
  std::mutex binary_draw_mutex_;

  cv::Mat gray_draw_;
  bool gray_draw_ready_;
  std::mutex gray_draw_mutex_;
};

}  // namespace nodes
}  // namespace graphs
}  // namespace daisykit

#endif
