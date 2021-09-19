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

#ifndef DAISYKIT_GRAPHS_NODES_FACE_VISUALIZER_NODE_H_
#define DAISYKIT_GRAPHS_NODES_FACE_VISUALIZER_NODE_H_

#include "daisykitsdk/common/profiler.h"
#include "daisykitsdk/common/types.h"
#include "daisykitsdk/common/visualizers/face_visualizer.h"
#include "daisykitsdk/graphs/core/node.h"

#include <chrono>
#include <memory>

namespace daisykit {
namespace graphs {

namespace nodes {

// Face visualizer.
// Receives "image" and "faces" as inputs. Draw faces and landmarks to the image
// and show the final result in an OpenCV imshow window.
class FaceVisualizerNode : public Node {
 public:
  FaceVisualizerNode(const std::string& node_name,
                     NodeType node_type = NodeType::kSyncNode,
                     bool with_landmark = false)
      : Node(node_name, node_type) {
    with_landmark_ = with_landmark;
  }

  void Process(std::shared_ptr<Packet> in_packet,
               std::shared_ptr<Packet>& out_packet) {}

  void Tick() {
    WaitForData();

    std::map<std::string, PacketPtr> inputs;
    PrepareInputs(inputs);

    // Get new faces result
    // or take the last result
    std::shared_ptr<std::vector<daisykit::types::Face>> faces;
    if (inputs.count("faces") > 0) {
      faces = inputs["faces"]->GetData<std::vector<daisykit::types::Face>>();
      faces_ = faces;
    } else {
      faces = faces_;
    }

    // Get image
    cv::Mat img = *inputs["image"]->GetData<cv::Mat>();

    // Clone image to draw on
    cv::Mat draw = img.clone();

    // Draw face to image
    if (faces != nullptr) {
      visualizers::FaceVisualizer::DrawFace(draw, *faces, with_landmark_);
      double fps = profiler.Tick();
      visualizers::BaseVisualizer::PutText(
          draw, std::string("FPS: ") + std::to_string(fps), cv::Point(100, 100),
          cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, 10, cv::Scalar(0, 0, 0),
          cv::Scalar(0, 255, 0));
    }

    cv::cvtColor(draw, draw, cv::COLOR_RGB2BGR);
    cv::imshow("Face detection", draw);
    cv::waitKey(1);
  }

 private:
  bool with_landmark_;
  std::shared_ptr<std::vector<daisykit::types::Face>> faces_;
  Profiler profiler;
};

}  // namespace nodes
}  // namespace graphs
}  // namespace daisykit

#endif