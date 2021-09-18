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

#include "daisykitsdk/graphs/core/graph.h"
#include "daisykitsdk/graphs/core/node.h"
#include "daisykitsdk/graphs/nodes/models/face_detector_node.h"
#include "daisykitsdk/graphs/nodes/models/face_landmark_estimator_node.h"
#include "daisykitsdk/graphs/nodes/packet_distributor_node.h"
#include "daisykitsdk/graphs/nodes/visualizers/face_visualizer_node.h"
#include "daisykitsdk/thirdparties/json.hpp"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <streambuf>
#include <string>

using namespace cv;
using namespace std;
using namespace daisykit;
using namespace daisykit::graphs;

int main(int, char**) {
  // Create processing nodes
  std::shared_ptr<nodes::PacketDistributorNode> packet_distributor_node =
      std::make_shared<nodes::PacketDistributorNode>("packet_distributor",
                                                     NodeType::kAsyncNode);
  std::shared_ptr<nodes::FaceDetectorNode> face_detector_node =
      std::make_shared<nodes::FaceDetectorNode>(
          "face_detector",
          "../assets/models/face_detection/yolo_fastest_with_mask/"
          "yolo-fastest-opt.param",
          "../assets/models/face_detection/yolo_fastest_with_mask/"
          "yolo-fastest-opt.bin",
          NodeType::kAsyncNode);
  std::shared_ptr<nodes::FacialLandmarkEstimatorNode>
      facial_landmark_estimator_node =
          std::make_shared<nodes::FacialLandmarkEstimatorNode>(
              "facial_landmark_estimator",
              "../assets/models/facial_landmark/pfld-sim.param",
              "../assets/models/facial_landmark/pfld-sim.bin",
              NodeType::kAsyncNode);
  std::shared_ptr<nodes::FaceVisualizerNode> face_visualizer_node =
      std::make_shared<nodes::FaceVisualizerNode>("face_visualizer",
                                                  NodeType::kAsyncNode, true);

  // Create connections between nodes
  Graph::Connect(nullptr, "", packet_distributor_node.get(), "input",
                 TransmissionProfile(2, true), true);

  Graph::Connect(packet_distributor_node.get(), "output",
                 face_detector_node.get(), "input",
                 TransmissionProfile(2, true), true);

  Graph::Connect(packet_distributor_node.get(), "output",
                 facial_landmark_estimator_node.get(), "image",
                 TransmissionProfile(2, true), true);
  Graph::Connect(face_detector_node.get(), "output",
                 facial_landmark_estimator_node.get(), "faces",
                 TransmissionProfile(2, true), true);

  Graph::Connect(packet_distributor_node.get(), "output",
                 face_visualizer_node.get(), "image",
                 TransmissionProfile(2, true), true);
  Graph::Connect(facial_landmark_estimator_node.get(), "output",
                 face_visualizer_node.get(), "faces",
                 TransmissionProfile(2, true), true);

  // Need to init these nodes before use
  // This method also start worker threads of asynchronous node
  packet_distributor_node->Activate();
  face_detector_node->Activate();
  facial_landmark_estimator_node->Activate();
  face_visualizer_node->Activate();

  VideoCapture cap(0);

  while (1) {
    Mat frame;
    cap >> frame;
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    std::shared_ptr<Packet> in_packet = Packet::MakePacket<cv::Mat>(frame);
    packet_distributor_node->Input("input", in_packet);
  }

  return 0;
}