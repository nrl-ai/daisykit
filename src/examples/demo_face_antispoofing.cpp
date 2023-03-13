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

#include "daisykit/models/face_anti_spoofing.h"
#include "daisykit/common/types.h"

#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

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
  FakeRealClassifiers model("models/face_anti_spoofing/detection.param", "models/face_anti_spoofing/detection.bin", 480,640, false);
  string image_path = samples::findFile("/Users/haiduong/Downloads/Frame/real/G_NT_ZTE_g_E_5_45_120.jpg");
  Mat img = imread(image_path, IMREAD_COLOR);
  
  cv::Mat rgb;
  cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

  std::vector<FaceBox> face_anti_spoof;
  model.Detect(rgb, face_anti_spoof);

  cv::Mat draw;
  cv::cvtColor(rgb, draw, cv::COLOR_RGB2BGR);

  for (size_t i = 0; i < face_anti_spoof.size(); ++i) {
    cv::rectangle(draw,
                  cv::Rect(face_anti_spoof[i].x1, face_anti_spoof[i].y1, face_anti_spoof[i].x2, face_anti_spoof[i].y2),
                  cv::Scalar(0, 255, 0), 2);
    string result = face_anti_spoof[i].confidence >= 0.5 ? "Real" : "Spoof";
    cout << result << endl;
  }
  imshow("Image", draw);
  waitKey(0);

  return 0;
}
