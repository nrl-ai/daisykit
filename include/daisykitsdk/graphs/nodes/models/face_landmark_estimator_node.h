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

#include "daisykitsdk/models/facial_landmark_estimator.h"

#include <chrono>
#include <memory>

namespace daisykit {
namespace graphs {

namespace nodes {

/// Face landmark estimator node.
/// Receives an image from "image" and face bounding boxes from "faces", add
/// landmark info to "faces" packet and push the output through "output".
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

  void Tick() {
    // Wait for data
    WaitForData();

    // Prepare input packets
    std::map<std::string, PacketPtr> inputs;
    PrepareInputs(inputs);

    // Get faces result
    std::shared_ptr<std::vector<daisykit::types::Face>> faces;
    faces = inputs["faces"]->GetData<std::vector<daisykit::types::Face>>();

    // Get image
    cv::Mat img = *inputs["image"]->GetData<cv::Mat>();

    // Modify faces to add landmark info
    facial_landmark_estimator_->PredictMulti(img, *faces);

    // Convert to output packet
    PacketPtr output;
    utils::TimePoint timestamp = utils::Timer::GetCurrentTime();
    output = std::make_shared<Packet>(std::static_pointer_cast<void>(faces),
                                      timestamp);

    std::map<std::string, PacketPtr> outputs;
    outputs.insert(std::make_pair("output", output));
    Publish(outputs);
  }

 private:
  std::shared_ptr<models::FacialLandmarkEstimator> facial_landmark_estimator_;
};

}  // namespace nodes
}  // namespace graphs
}  // namespace daisykit

#endif
