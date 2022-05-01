# C++: Object Detection

A general-purpose object detector based on
[*YOLOX*](https://github.com/Megvii-BaseDetection/YOLOX) is integrated
with Daisykit. The models are trained on the COCO dataset using the
[*official repository of
YOLOX*](https://github.com/Megvii-BaseDetection/YOLOX). You can retrain
the model with your custom dataset and convert it to NCNN format, which
can be integrated into Daisykit easily.

![](/images/python/image7.gif)

Source code: `src/examples/demo_object_detector_yolox.cpp`.

```cpp
#include "daisykit/common/types.h"
#include "daisykit/flows/object_detector_flow.h"
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
  std::ifstream t("configs/object_detector_yolox_config.json");
  std::string config_str((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());

  ObjectDetectorFlow flow(config_str);

  Mat frame;
  VideoCapture cap(0);

  while (1) {
    cap >> frame;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    std::vector<Object> objects = flow.Process(rgb);
    flow.DrawResult(rgb, objects);

    cv::Mat draw;
    cv::cvtColor(rgb, draw, cv::COLOR_RGB2BGR);
    imshow("Image", draw);
    waitKey(1);
  }

  return 0;
}
```

Update the configurations by modifying config files in `assets/configs`.
