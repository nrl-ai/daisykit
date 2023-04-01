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
#include "daisykit/models/face_recognition/face_alignment.h"
#include "daisykit/models/face_recognition/face_detector_scrfd.h"
#include "daisykit/models/face_recognition/face_extractor.h"
#include "daisykit/models/face_recognition/face_manager.h"
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

FaceDetectorSCRFD<types::FaceExtended>* face_detector =
    new FaceDetectorSCRFD<types::FaceExtended>(
        "models/face_detection/scrfd/scrfd_2.5g_1.param",
        "models/face_detection/scrfd/scrfd_2.5g_1.bin", 640, 0.7, 0.5, true);
FaceAligner* face_aligner = new FaceAligner();

FaceExtractor* face_extractor = new FaceExtractor(
    "models/face_extraction/arcface/iresnet18_1.param",
    "models/face_extraction/arcface/iresnet18_1.bin", 112, true);

FaceManager* face_manager = new FaceManager("data.hnsw", 1000, 1, 512, 1.01);

int main(int argc, char** argv) {
  // Read faces and name to register
  if (argc < 2) {
    std::cout << "Please an image to register" << std::endl;
    std::cout << "Example: ./demo_face_sequential data/my_face.jpg"
              << std::endl;
    return 1;
  }

  if (!face_manager->LoadData())
    std::cout << "File was not found or not supported." << std::endl;

  // Read parameters
  std::string person_image_path = argv[1];
  std::string person_name;
  for (int i = 2; i < argc; i++) {
    person_name += argv[i];
    if (i < argc - 1) person_name += " ";
  }

  // Predict and check faces
  // Currently only support one face per image
  std::vector<types::FaceExtended> faces;
  std::vector<types::FaceSearchResult> result;
  cv::Mat regis_img = cv::imread(person_image_path);
  faces.clear();
  face_detector->Predict(regis_img, faces);
  std::cout << faces.size() << std::endl;
  if (faces.size() != 1) {
    std::cout << "Provide image with only one face to register";
    return 1;
  }

  // Align and extract face feature
  face_aligner->AlignMutipleFaces(regis_img, faces);
  face_extractor->Predict(faces);
  if (faces.size() == 1) {
    int id;
    if (face_manager->Insert(faces[0].feature, id))  // feature, name, id
      std::cout << "Inserted successfully with id : " << id << std::endl;
    else
      std::cout << "Insertion failed." << std::endl;
  }

  std::cout << "Number of loaded faces: " << face_manager->GetNumDatas()
            << std::endl;

  Mat frame;
  VideoCapture cap(0);
  bool draw_landmarks = false;

  while (1) {
    cap >> frame;
    face_detector->Predict(frame, faces);
    face_aligner->AlignMutipleFaces(frame, faces);
    face_extractor->Predict(faces);
    cv::Mat draw = frame.clone();
    visualizers::FaceVisualizer<types::FaceExtended>::DrawFace(draw, faces,
                                                               draw_landmarks);
    for (auto face : faces) {
      if (face_manager->Search(face.feature, result)) {
        for (types::FaceSearchResult& faceif : result) {
          // Format distance to 2 decimal places
          std::stringstream stream;
          stream << std::fixed << std::setprecision(2)
                 << faceif.min_distance * 100;
          std::string formated_distance = stream.str();
          visualizers::BaseVisualizer::PutText(
              draw, std::to_string(faceif.id) + ": " + formated_distance,
              cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, -1);
        }
      }
    }
    imshow("Image", draw);
    waitKey(1);
  }

  return 0;
}
