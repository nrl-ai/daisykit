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

#ifndef DAISYKIT_GRAPHS_NODES_IMG_PROC_GRAYSCALE_NODE_H_
#define DAISYKIT_GRAPHS_NODES_IMG_PROC_GRAYSCALE_NODE_H_

#include "daisykitsdk/graphs/core/node.h"
#include "daisykitsdk/graphs/core/utils.h"

#include <memory>

namespace daisykit {
namespace graphs {

class GrayScaleNode : public Node {
 public:
  using Node::Node;  // For constructor inheritance

  void Process(PacketPtr in_packet, PacketPtr& out_packet) {
    // Convert packet to processing format: cv::Mat
    cv::Mat img;
    Packet2CvMat(in_packet, img);

    // Process
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Convert & assign output to the output packet
    CvMat2Packet(gray, out_packet);
  }

  void Tick() {
    WaitForData();

    std::map<std::string, PacketPtr> inputs;
    if (PrepareInputs(inputs) != 0) {
      std::cerr << GetNodeName() << ": Error on preparing inputs." << std::endl;
      return;
    }

    PacketPtr input = inputs["input"];
    PacketPtr output;
    Process(input, output);

    std::map<std::string, PacketPtr> outputs;
    outputs.insert(std::make_pair("output", output));

    Publish(outputs);
  }
};

}  // namespace graphs
}  // namespace daisykit

#endif