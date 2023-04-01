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
#include "daisykit/common/visualizers/face_visualizer.h"
#include "daisykit/models/face_recognition/face_detector_scrfd.h"
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
using namespace daisykit::models;

FaceDetectorSCRFD<types::Face>* face_detector =
    new FaceDetectorSCRFD<types::Face>(
        "models/face_detection/scrfd/scrfd_2.5g_1.param",
        "models/face_detection/scrfd/scrfd_2.5g_1.bin", 640, 0.7, 0.5, false);

int main(int, char**) {
  Mat frame;
  VideoCapture cap(0);

  std::vector<types::Face> faces;
  while (1) {
    cap >> frame;
    face_detector->Predict(frame, faces);
    cv::Mat draw = frame.clone();
    visualizers::FaceVisualizer<types::Face>::DrawFace(draw, faces, true);
    imshow("Image", draw);
    waitKey(1);
  }

  return 0;
}
