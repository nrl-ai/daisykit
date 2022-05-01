# C++: Background Matting

Background matting use only one segmentation model to generate a human
body mask. This mask can figure out which pixels belong to humans and
which belong to the background. This output can be used for background
replacement (like in the Google Meet app). The segmentation model was
taken from [*this
implementation*](https://github.com/nihui/ncnn-webassembly-portrait-segmentation)
by [*nihui*](https://github.com/nihui), the author of the NCNN
framework. The author also has a webpage for a live demo on web
browsers.

[*https://github.com/nihui/ncnn-webassembly-portrait-segmentation*](https://github.com/nihui/ncnn-webassembly-portrait-segmentation).

![](/images/python/image8.gif)

Source code: `src/examples/demo_background_matting.cpp`.

```cpp
#include "daisykit/flows/background_matting_flow.h"
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
using namespace daisykit::types;
using namespace daisykit::flows;

int main(int, char**) {
  std::ifstream t("configs/background_matting_config.json");
  std::string config_str((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());

  cv::Mat background = cv::imread("images/background.jpg");
  cv::cvtColor(background, background, cv::COLOR_BGR2RGB);
  BackgroundMattingFlow flow(config_str, background);

  Mat frame;
  VideoCapture cap(0);

  while (1) {
    cap >> frame;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    cv::Mat draw = rgb.clone();

    cv::Mat mask = flow.Process(rgb);
    flow.DrawResult(draw, mask);

    cv::cvtColor(draw, draw, cv::COLOR_RGB2BGR);
    imshow("Image", draw);
    waitKey(1);
  }

  return 0;
}
```

Update the configurations by modifying config files in `assets/configs`.
