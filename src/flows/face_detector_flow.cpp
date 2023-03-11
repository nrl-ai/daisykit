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

#include "daisykit/flows/face_detector_flow.h"
#include "daisykit/common/visualizers/face_visualizer.h"
#include "third_party/json.hpp"

namespace daisykit {
namespace flows {

FaceDetectorFlow::FaceDetectorFlow(const std::string& config_str,
                                   bool show_fps) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  face_detector_ = new models::FaceDetector(
      config["face_detection_model"]["model"],
      config["face_detection_model"]["weights"],
      config["face_detection_model"]["score_threshold"],
      config["face_detection_model"]["iou_threshold"],
      config["face_detection_model"]["input_width"],
      config["face_detection_model"]["input_height"],
      config["face_detection_model"]["use_gpu"]);
  with_landmark_ = config["with_landmark"];
  if (with_landmark_) {
    facial_landmark_detector_ = new models::FacialLandmarkDetector(
        config["facial_landmark_model"]["model"],
        config["facial_landmark_model"]["weights"],
        config["facial_landmark_model"]["input_width"],
        config["facial_landmark_model"]["input_height"],
        config["facial_landmark_model"]["use_gpu"]);
  }
  show_fps_ = show_fps;
}

#ifdef __ANDROID__
FaceDetectorFlow::FaceDetectorFlow(AAssetManager* mgr,
                                   const std::string& config_str,
                                   bool show_fps) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  face_detector_ = new models::FaceDetector(
      mgr, config["face_detection_model"]["model"],
      config["face_detection_model"]["weights"],
      config["face_detection_model"]["score_threshold"],
      config["face_detection_model"]["iou_threshold"],
      config["face_detection_model"]["input_width"],
      config["face_detection_model"]["input_height"],
      config["face_detection_model"]["use_gpu"]);
  with_landmark_ = config["with_landmark"];
  if (with_landmark_) {
    facial_landmark_detector_ = new models::FacialLandmarkDetector(
        mgr, config["facial_landmark_model"]["model"],
        config["facial_landmark_model"]["weights"],
        config["facial_landmark_model"]["input_width"],
        config["facial_landmark_model"]["input_height"],
        config["facial_landmark_model"]["use_gpu"]);
  }
  show_fps_ = show_fps;
}
#endif

FaceDetectorFlow::~FaceDetectorFlow() {
  delete face_detector_;
  face_detector_ = nullptr;
  delete facial_landmark_detector_;
  facial_landmark_detector_ = nullptr;
}

std::vector<types::Face> FaceDetectorFlow::Process(const cv::Mat& rgb) {
  // Detect faces
  std::vector<types::Face> faces;
  face_detector_->Predict(rgb, faces);

  // Detect landmarks
  if (with_landmark_) {
    facial_landmark_detector_->PredictMulti(rgb, faces);
  }

  // Tick profiler when processing is done
  profiler.Tick();

  return faces;
}

void FaceDetectorFlow::DrawResult(cv::Mat& rgb,
                                  std::vector<types::Face>& faces) {
  // Draw face bounding boxes and keypoints
  visualizers::FaceVisualizer<types::Face>::DrawFace(rgb, faces, true);
  if (show_fps_)
    visualizers::BaseVisualizer::PutText(
        rgb, std::string("FPS: ") + std::to_string(profiler.CurrentFPS()),
        cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, 10,
        cv::Scalar(0, 0, 0), cv::Scalar(0, 255, 0));
}

}  // namespace flows
}  // namespace daisykit
