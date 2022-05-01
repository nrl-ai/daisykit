# C++: Face Detection

Face detector flow in Daisykit contains a face detection model based on
[*YOLO Fastest*](https://github.com/dog-qiuqiu/Yolo-Fastest) and a
facial landmark detection model based on
[*PFLD*](https://github.com/polarisZhao/PFLD-pytorch). In addition, to
encourage makers to join hands in the fighting with COVID-19, we
selected a face detection model that can recognize people wearing face
masks or not.

![](/images/python/image5.png)

![](/images/python/image14.gif)

## 1. Sequential flow

Source code: `src/examples/demo_face_detector.cpp`.

```cpp
#include "daisykit/common/types.h"
#include "daisykit/flows/face_detector_flow.h"
#include "third_party/json.hpp"

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
using namespace daisykit::flows;

int main(int, char**) {
  std::ifstream t("configs/face_detector_config.json");
  std::string config_str((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());

  FaceDetectorFlow flow(config_str, true);

  Mat frame;
  VideoCapture cap(0);

  while (1) {
    cap >> frame;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    std::vector<types::Face> faces = flow.Process(rgb);
    flow.DrawResult(rgb, faces);

    cv::Mat draw;
    cv::cvtColor(rgb, draw, cv::COLOR_RGB2BGR);
    imshow("Image", draw);
    waitKey(1);
  }

  return 0;
}
```

Update the configurations by modifying config files in `assets/configs`.


## 2. Multithreading mode with graph

In order to use multithreading mode (graph mode) to improve the FPS, try the example in `src/examples/demo_face_detector_graph.cpp`.


```cpp
#include "daisykit/graphs/core/graph.h"
#include "daisykit/graphs/core/node.h"
#include "daisykit/graphs/nodes/models/face_detector_node.h"
#include "daisykit/graphs/nodes/models/face_landmark_detector_node.h"
#include "daisykit/graphs/nodes/packet_distributor_node.h"
#include "daisykit/graphs/nodes/visualizers/face_visualizer_node.h"
#include "third_party/json.hpp"

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
          "models/face_detection/yolo_fastest_with_mask/"
          "yolo-fastest-opt.param",
          "models/face_detection/yolo_fastest_with_mask/"
          "yolo-fastest-opt.bin",
          NodeType::kAsyncNode);
  std::shared_ptr<nodes::FacialLandmarkDetectorNode>
      facial_landmark_detector_node =
          std::make_shared<nodes::FacialLandmarkDetectorNode>(
              "facial_landmark_detector",
              "models/facial_landmark/pfld-sim.param",
              "models/facial_landmark/pfld-sim.bin", NodeType::kAsyncNode);
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
                 facial_landmark_detector_node.get(), "image",
                 TransmissionProfile(2, true), true);
  Graph::Connect(face_detector_node.get(), "output",
                 facial_landmark_detector_node.get(), "faces",
                 TransmissionProfile(2, true), true);

  Graph::Connect(packet_distributor_node.get(), "output",
                 face_visualizer_node.get(), "image",
                 TransmissionProfile(2, true), true);
  Graph::Connect(facial_landmark_detector_node.get(), "output",
                 face_visualizer_node.get(), "faces",
                 TransmissionProfile(2, true), true);

  // Need to init these nodes before use
  // This method also start worker threads of asynchronous node
  packet_distributor_node->Activate();
  face_detector_node->Activate();
  facial_landmark_detector_node->Activate();
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
```
