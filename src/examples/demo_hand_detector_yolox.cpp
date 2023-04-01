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

#include "daisykit/common/types.h"
#include "daisykit/models/hand_detector_yolox.h"
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
using namespace daisykit::types;
using namespace daisykit::models;

int main(int, char**) {
  HandDetectorYOLOX model("models/hand_detection/yolox/yolox_hand_relu.param",
                          "models/hand_detection/yolox/yolox_hand_relu.bin",
                          0.3, 0.3, 256, 256);

  Mat frame;
  VideoCapture cap(0);

  while (1) {
    cap >> frame;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    std::vector<Object> hands;
    model.Predict(rgb, hands);

    cv::Mat draw;
    cv::cvtColor(rgb, draw, cv::COLOR_RGB2BGR);

    for (size_t i = 0; i < hands.size(); ++i) {
      cv::rectangle(draw,
                    cv::Rect(hands[i].x, hands[i].y, hands[i].w, hands[i].h),
                    cv::Scalar(0, 255, 0), 2);
    }

    imshow("Image", draw);
    waitKey(1);
  }

  return 0;
}
