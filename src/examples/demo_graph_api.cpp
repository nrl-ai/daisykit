#include <stdio.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <streambuf>
#include <string>

#include <daisykitsdk/graphs/core/graph.h>
#include <daisykitsdk/graphs/core/node.h>
#include <daisykitsdk/graphs/core/utils.h>
#include <daisykitsdk/graphs/img_proc/binarize_node.h>
#include <daisykitsdk/graphs/img_proc/grayscale_node.h>
#include <daisykitsdk/graphs/img_proc/img_viz_node.h>
#include <daisykitsdk/thirdparties/json.hpp>

using namespace cv;
using namespace std;
using json = nlohmann::json;
using namespace daisykit::common;
using namespace daisykit::graphs;

int main(int, char**) {
  // std::shared_ptr<GrayScaleNode> grayscale_node =
  // std::make_shared<GrayScaleNode>("grayscale", NodeType::kSyncNode);
  // std::shared_ptr<BinarizeNode> binarize_node =
  // std::make_shared<BinarizeNode>("binary", NodeType::kSyncNode);
  // std::shared_ptr<ImgVizNode> visualize_node =
  // std::make_shared<ImgVizNode>("binary", NodeType::kSyncNode);

  // Create processing nodes
  std::shared_ptr<GrayScaleNode> grayscale_node =
      std::make_shared<GrayScaleNode>("grayscale", NodeType::kAsyncNode);
  std::shared_ptr<BinarizeNode> binarize_node =
      std::make_shared<BinarizeNode>("binary", NodeType::kAsyncNode);
  std::shared_ptr<ImgVizNode> visualize_node =
      std::make_shared<ImgVizNode>("binary", NodeType::kAsyncNode);

  // Create connections between nodes
  Graph::Connect(nullptr, "", grayscale_node.get(), "input",
                 TransmissionProfile(5, true), true);
  Graph::Connect(grayscale_node.get(), "output", binarize_node.get(), "input",
                 TransmissionProfile(5, true), true);
  Graph::Connect(binarize_node.get(), "output", visualize_node.get(), "binary",
                 TransmissionProfile(5, true), false);
  Graph::Connect(grayscale_node.get(), "output", visualize_node.get(), "gray",
                 TransmissionProfile(5, true), false);

  // Need to init these nodes before use
  // This method also start worker threads of asynchronous node
  grayscale_node->Activate();
  binarize_node->Activate();
  visualize_node->Activate();

  Mat frame;
  VideoCapture cap(0);

  std::shared_ptr<Packet> in_packet;
  while (1) {
    cap >> frame;
    CvMat2Packet(frame, in_packet);
    grayscale_node->Input("input", in_packet);
  }

  return 0;
}