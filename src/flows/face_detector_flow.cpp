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

#include "daisykitsdk/flows/face_detector_flow.h"
#include "daisykitsdk/common/visualizers/face_visualizer.h"
#include "daisykitsdk/thirdparties/json.hpp"

namespace daisykit {
namespace flows {

FaceDetectorFlow::FaceDetectorFlow(const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  face_detector_ = new models::FaceDetector(
      config["face_detection_model"]["model"],
      config["face_detection_model"]["weights"],
      config["face_detection_model"]["input_width"],
      config["face_detection_model"]["input_height"],
      config["face_detection_model"]["score_threshold"],
      config["face_detection_model"]["iou_threshold"]);
  with_landmark_ = config["with_landmark"];
  if (with_landmark_) {
    facial_landmark_estimator_ = new models::FacialLandmarkEstimator(
        config["facial_landmark_model"]["model"],
        config["facial_landmark_model"]["weights"]);
  }
}

#ifdef __ANDROID__
FaceDetectorFlow::FaceDetectorFlow(AAssetManager* mgr,
                                   const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  face_detector_ = new models::FaceDetector(
      mgr, config["face_detection_model"]["model"],
      config["face_detection_model"]["weights"],
      config["face_detection_model"]["input_width"],
      config["face_detection_model"]["input_height"],
      config["face_detection_model"]["score_threshold"],
      config["face_detection_model"]["iou_threshold"]);
  with_landmark_ = config["with_landmark"];
  if (with_landmark_) {
    facial_landmark_estimator_ = new models::FacialLandmarkEstimator(
        config["facial_landmark_model"]["model"],
        config["facial_landmark_model"]["weights"]);
  }
}
#endif

FaceDetectorFlow::~FaceDetectorFlow() {
  delete face_detector_;
  face_detector_ = nullptr;
  delete facial_landmark_estimator_;
  facial_landmark_estimator_ = nullptr;
}

std::vector<types::Face> FaceDetectorFlow::Process(const cv::Mat& rgb) {
  // Detect faces
  std::vector<types::Face> faces = face_detector_->Predict(rgb);

  // Detect landmarks
  if (with_landmark_) {
    facial_landmark_estimator_->PredictMulti(rgb, faces);
  }

  return faces;
}

void FaceDetectorFlow::DrawResult(cv::Mat& rgb,
                                  std::vector<types::Face>& faces) {
  // Draw face bounding boxes and keypoints
  visualizers::FaceVisualizer::DrawFace(rgb, faces, true);
}

}  // namespace flows
}  // namespace daisykit
