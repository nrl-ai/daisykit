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

#ifndef DAISYKIT_GRAPHS_NODES_MODELS_FACE_DETECTOR_NODE_H_
#define DAISYKIT_GRAPHS_NODES_MODELS_FACE_DETECTOR_NODE_H_

#include "daisykitsdk/graphs/core/node.h"

#include "daisykitsdk/models/face_detector.h"

#include <chrono>
#include <memory>

namespace daisykit {
namespace graphs {

namespace nodes {

/// Face detector node.
/// Receives an image from "input" connection, detects all faces in that image
/// and pushes the result through "output".
class FaceDetectorNode : public Node {
 public:
  FaceDetectorNode(const std::string& node_name, const std::string& param_file,
                   const std::string& weight_file,
                   NodeType node_type = NodeType::kSyncNode)
      : Node(node_name, node_type) {
    // Init model
    face_detector_ =
        std::make_shared<models::FaceDetector>(param_file, weight_file);
  }
  void Process(std::shared_ptr<Packet> in_packet,
               std::shared_ptr<Packet>& out_packet) {
    // Convert packet to processing format: cv::Mat
    cv::Mat img = *in_packet->GetData<cv::Mat>();

    // Process
    std::shared_ptr<std::vector<daisykit::types::Face>> result =
        std::make_shared<std::vector<daisykit::types::Face>>();
    face_detector_->Detect(img, *result);

    // Convert to output packet
    utils::TimePoint timestamp = utils::Timer::GetCurrentTime();
    out_packet = Packet::MakePacket<std::vector<daisykit::types::Face>>(result);
  }

  void Tick() {
    // Wait for data
    WaitForData();

    // Prepare input packets
    std::map<std::string, PacketPtr> inputs;
    PrepareInputs(inputs);

    PacketPtr input = inputs["input"];
    PacketPtr output;
    Process(input, output);

    std::map<std::string, PacketPtr> outputs;
    outputs.insert(std::make_pair("output", output));
    Publish(outputs);
  }

 private:
  std::shared_ptr<models::FaceDetector> face_detector_;
};

}  // namespace nodes
}  // namespace graphs
}  // namespace daisykit

#endif
