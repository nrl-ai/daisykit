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

#ifndef DAISYKIT_GRAPHS_NODES_MODELS_FACIAL_LANDMARK_ESTIMATOR_NODE_H_
#define DAISYKIT_GRAPHS_NODES_MODELS_FACIAL_LANDMARK_ESTIMATOR_NODE_H_

#include "daisykitsdk/graphs/core/node.h"
#include "daisykitsdk/graphs/core/utils.h"

#include "daisykitsdk/models/facial_landmark_estimator.h"

#include <chrono>
#include <memory>

namespace daisykit {
namespace graphs {

class FacialLandmarkEstimatorNode : public Node {
 public:
  FacialLandmarkEstimatorNode(const std::string& node_name,
                              const std::string& param_file,
                              const std::string& weight_file,
                              NodeType node_type = NodeType::kSyncNode)
      : Node(node_name, node_type) {
    // Init model
    facial_landmark_estimator_ =
        std::make_shared<models::FacialLandmarkEstimator>(param_file,
                                                          weight_file);
  }
  void Process(std::shared_ptr<Packet> in_packet,
               std::shared_ptr<Packet>& out_packet) {}

  void Tick() {
    WaitForData();

    std::map<std::string, PacketPtr> inputs;
    if (PrepareInputs(inputs) != 0) {
      std::cerr << GetNodeName() << ": Error on preparing inputs." << std::endl;
      return;
    }

    // Get faces result
    std::shared_ptr<std::vector<daisykit::common::Face>> faces;
    faces = ParseFacePacket(inputs["faces"]);

    // Get image
    cv::Mat img;
    Packet2CvMat(inputs["image"], img);

    // Modify faces to add landmark info
    facial_landmark_estimator_->DetectMulti(img, *faces);

    // Convert to output packet
    PacketPtr output;
    utils::TimePoint timestamp = utils::Timer::GetCurrentTime();
    output = std::make_shared<Packet>(std::static_pointer_cast<void>(faces),
                                      timestamp);

    std::map<std::string, PacketPtr> outputs;
    outputs.insert(std::make_pair("output", output));
    Publish(outputs);
  }

  std::shared_ptr<std::vector<daisykit::common::Face>> ParseFacePacket(
      PacketPtr packet) {
    // Get data
    std::shared_ptr<void> data;
    utils::TimePoint timestamp;
    packet->GetData(data, timestamp);

    // Cast to faces
    std::shared_ptr<std::vector<daisykit::common::Face>> faces =
        std::static_pointer_cast<std::vector<daisykit::common::Face>>(data);

    return faces;
  }

 private:
  std::shared_ptr<models::FacialLandmarkEstimator> facial_landmark_estimator_;
};

}  // namespace graphs
}  // namespace daisykit

#endif