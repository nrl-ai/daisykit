# C++: Hand Pose Detection

The hand pose detection flow comprises two models: a hand detection
model based on YOLOX and a 3D hand pose detection model released by
Google this November. Thanks to
[*FeiGeChuanShu*](https://github.com/FeiGeChuanShu) for the effort in
early model conversion.

![](/images/python/image1.png)

This hand pose flow can be used in AR games, hand gesture control, and
many cool DIY projects.

![](/images/python/image11.gif)

Source code: `src/examples/demo_hand_pose_detector.cpp`.

```cpp
#include "daisykit/common/types.h"
#include "daisykit/flows/hand_pose_detector_flow.h"
#include "third_party/json.hpp"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <streambuf>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
using json = nlohmann::json;
using namespace daisykit::types;
using namespace daisykit::flows;

int main(int, char**) {
  std::ifstream t("configs/hand_pose_yolox_mp_config.json");
  std::string config_str((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());

  HandPoseDetectorFlow flow(config_str);

  Mat frame;
  VideoCapture cap(0);

  while (1) {
    cap >> frame;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    std::vector<ObjectWithKeypointsXYZ> hands = flow.Process(rgb);
    flow.DrawResult(rgb, hands);

    cv::Mat draw;
    cv::cvtColor(rgb, draw, cv::COLOR_RGB2BGR);
    imshow("Image", draw);
    waitKey(1);
  }

  return 0;
}
```

Update the configurations by modifying config files in `assets/configs`. In the configuration file, `input_width` and `input_height` of the `hand_detection_model` can be adjusted for speed/accuracy trade-off.
