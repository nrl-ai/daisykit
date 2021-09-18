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
#include "daisykitsdk/graphs/nodes/img_proc/binarize_node.h"
#include "daisykitsdk/graphs/nodes/img_proc/grayscale_node.h"
#include "daisykitsdk/graphs/nodes/img_proc/img_viz_node.h"
#include "daisykitsdk/thirdparties/json.hpp"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <streambuf>
#include <string>

using namespace cv;
using namespace std;
using json = nlohmann::json;
using namespace daisykit;
using namespace daisykit::graphs;

int main(int, char**) {
  // Create processing nodes
  std::shared_ptr<nodes::GrayScaleNode> grayscale_node =
      std::make_shared<nodes::GrayScaleNode>("grayscale", NodeType::kAsyncNode);
  std::shared_ptr<nodes::BinarizeNode> binarize_node =
      std::make_shared<nodes::BinarizeNode>("binary", NodeType::kAsyncNode);
  std::shared_ptr<nodes::ImgVizNode> visualize_node =
      std::make_shared<nodes::ImgVizNode>("binary", NodeType::kAsyncNode);

  // Create connections between nodes
  Graph::Connect(nullptr, "", grayscale_node.get(), "input",
                 TransmissionProfile(2, true), true);
  Graph::Connect(grayscale_node.get(), "output", binarize_node.get(), "input",
                 TransmissionProfile(2, true), true);
  Graph::Connect(binarize_node.get(), "output", visualize_node.get(), "binary",
                 TransmissionProfile(2, true), false);
  Graph::Connect(grayscale_node.get(), "output", visualize_node.get(), "gray",
                 TransmissionProfile(2, true), false);

  // Need to init these nodes before use
  // This method also start worker threads of asynchronous node
  grayscale_node->Activate();
  binarize_node->Activate();
  visualize_node->Activate();

  VideoCapture cap(0);
  while (1) {
    Mat frame;
    cap >> frame;
    std::shared_ptr<Packet> in_packet = Packet::MakePacket<cv::Mat>(frame);
    grayscale_node->Input("input", in_packet);
  }

  return 0;
}