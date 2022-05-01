# C++: Human Pose

The human pose detector module contains an SSD-MobileNetV2 body detector
and a ported Google MoveNet model for human keypoints. This module can
be applied in fitness applications and AR games.

![](/images/python/image3.png)

![](/images/python/image12.gif)

Source code: `src/examples/demo_movenet.cpp`.

```cpp
#include "daisykit/common/types.h"
#include "daisykit/flows/human_pose_movenet_flow.h"
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
  std::ifstream t("configs/human_pose_movenet_config.json");
  std::string config_str((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());

  HumanPoseMoveNetFlow flow(config_str);

  Mat frame;
  VideoCapture cap(0);

  while (1) {
    cap >> frame;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    std::vector<ObjectWithKeypoints> poses = flow.Process(rgb);
    flow.DrawResult(rgb, poses);

    cv::Mat draw;
    cv::cvtColor(rgb, draw, cv::COLOR_RGB2BGR);
    imshow("Image", draw);
    waitKey(1);
  }

  return 0;
}
```

Update the configurations by modifying config files in `assets/configs`.
