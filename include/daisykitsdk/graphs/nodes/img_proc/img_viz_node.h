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

#include "daisykitsdk/graphs/core/node.h"

#include <memory>

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
      cv::imshow("Binary", binary);
      cv::waitKey(1);
    }

    if (inputs.count("gray") > 0) {
      PacketPtr gray_input = inputs["gray"];
      cv::Mat gray = *gray_input->GetData<cv::Mat>();
      cv::imshow("Gray", gray);
      cv::waitKey(1);
    }
  }
};

}  // namespace nodes
}  // namespace graphs
}  // namespace daisykit

#endif