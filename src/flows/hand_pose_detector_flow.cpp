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

#include "daisykit/flows/hand_pose_detector_flow.h"
#include "daisykit/common/visualizers/base_visualizer.h"
#include "third_party/json.hpp"

namespace daisykit {
namespace flows {

HandPoseDetectorFlow::HandPoseDetectorFlow(const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  hand_detector_ = new models::HandDetectorYOLOX(
      config["hand_detection_model"]["model"],
      config["hand_detection_model"]["weights"],
      config["hand_detection_model"]["score_threshold"],
      config["hand_detection_model"]["iou_threshold"],
      config["hand_detection_model"]["input_width"],
      config["hand_detection_model"]["input_height"],
      config["hand_detection_model"]["use_gpu"]);
  hand_pose_detector_ = new models::HandPoseDetector(
      config["hand_pose_model"]["model"], config["hand_pose_model"]["weights"],
      config["hand_pose_model"]["input_size"],
      config["hand_pose_model"]["use_gpu"]);
}

#ifdef __ANDROID__
HandPoseDetectorFlow::HandPoseDetectorFlow(AAssetManager* mgr,
                                           const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  hand_detector_ = new models::HandDetectorYOLOX(
      mgr, config["hand_detection_model"]["model"],
      config["hand_detection_model"]["weights"],
      config["hand_detection_model"]["score_threshold"],
      config["hand_detection_model"]["iou_threshold"],
      config["hand_detection_model"]["input_width"],
      config["hand_detection_model"]["input_height"],
      config["hand_detection_model"]["use_gpu"]);
  hand_pose_detector_ =
      new models::HandPoseDetector(mgr, config["hand_pose_model"]["model"],
                                   config["hand_pose_model"]["weights"],
                                   config["hand_pose_model"]["input_size"],
                                   config["hand_pose_model"]["use_gpu"]);
}
#endif

std::vector<types::ObjectWithKeypointsXYZ> HandPoseDetectorFlow::Process(
    cv::Mat& rgb) {
  // Detect hands
  std::vector<types::Object> hands;
  hand_detector_->Predict(rgb, hands);

  // Detect keypoints
  std::vector<std::vector<types::KeypointXYZ>> keypoints;
  std::vector<float> lr_scores;
  hand_pose_detector_->PredictMulti(rgb, hands, keypoints, lr_scores);

  // Prepare and return outputs
  std::vector<types::ObjectWithKeypointsXYZ> hands_with_pose;
  for (size_t i = 0; i < hands.size(); ++i) {
    types::ObjectWithKeypointsXYZ hand_with_pose(hands[i], keypoints[i]);
    hand_with_pose.class_id = lr_scores[i] > 0.3 ? 0 : 1;
    hands_with_pose.push_back(hand_with_pose);
  }
  return hands_with_pose;
}

void HandPoseDetectorFlow::DrawResult(
    cv::Mat& rgb, std::vector<types::ObjectWithKeypointsXYZ>& hands) {
  hand_pose_detector_->DrawHandPoses(rgb, hands);
}

}  // namespace flows
}  // namespace daisykit
