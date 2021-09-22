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

#include "daisykitsdk/flows/pushup_counter_flow.h"
#include "daisykitsdk/common/visualizers/base_visualizer.h"
#include "daisykitsdk/thirdparties/json.hpp"

namespace daisykit {
namespace flows {

PushupCounterFlow::PushupCounterFlow(const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  body_detector_ =
      new models::BodyDetector(config["person_detection_model"]["model"],
                               config["person_detection_model"]["weights"]);
  pose_detector_ =
      new models::PoseDetector(config["human_pose_model"]["model"],
                               config["human_pose_model"]["weights"]);
  action_classifier_ = new models::ActionClassifier(
      config["action_recognition_model"]["model"],
      config["action_recognition_model"]["weights"], true);
  pushup_analyzer_ = new processors::PushupAnalyzer();
  num_pushups_ = 0;
}

#ifdef __ANDROID__
PushupCounterFlow::PushupCounterFlow(AAssetManager* mgr,
                                     const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  body_detector_ =
      new models::BodyDetector(mgr, config["person_detection_model"]["model"],
                               config["person_detection_model"]["weights"]);
  pose_detector_ =
      new models::PoseDetector(mgr, config["background_pose_model"]["model"],
                               config["background_pose_model"]["weights"]);
  action_classifier_ = new models::ActionClassifier(
      mgr, config["action_recognition_model"]["model"],
      config["action_recognition_model"]["weights"], true);
  pushup_analyzer_ = new examples::PushupAnalyzer();
  num_pushups_ = 0;
}
#endif

void PushupCounterFlow::Process(cv::Mat& rgb) {
  // Detect background pose
  std::vector<types::Object> bodies = body_detector_->Predict(rgb);
  {
    const std::lock_guard<std::mutex> lock(bodies_lock_);
    bodies_ = bodies;
  }

  // Detect keypoints
  std::vector<std::vector<types::Keypoint>> keypoints =
      pose_detector_->PredictMulti(rgb, bodies);
  {
    const std::lock_guard<std::mutex> lock(keypoints_lock_);
    keypoints_ = keypoints;
  }

  // Recognize action and count pushups
  float score;
  types::Action action = action_classifier_->Classify(rgb, score);
  is_pushup_score_ = score;
  num_pushups_ =
      pushup_analyzer_->CountPushups(rgb, action == types::Action::kPushup);
}

int PushupCounterFlow::NumPushups() { return num_pushups_; }

void PushupCounterFlow::DrawResult(cv::Mat& rgb) {
  // Draw body bounding boxes
  {
    const std::lock_guard<std::mutex> lock(bodies_lock_);
    for (auto body : bodies_) {
      cv::rectangle(rgb, cv::Rect(body.x, body.y, body.w, body.h),
                    cv::Scalar(0, 255, 0), 2);
    }
  }

  // Draw body keypoints
  {
    const std::lock_guard<std::mutex> lock(keypoints_lock_);
    for (auto kp_single : keypoints_) {
      pose_detector_->DrawKeypoints(rgb, kp_single);
    }
  }

  // Draw pushups counting
  if (is_pushup_score_ > 0.5) {
    visualizers::BaseVisualizer::PutText(
        rgb, "Is pushing: " + std::to_string(is_pushup_score_),
        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, 10,
        cv::Scalar(0, 0, 0), cv::Scalar(0, 255, 0));
  } else {
    visualizers::BaseVisualizer::PutText(
        rgb, "Is pushing: " + std::to_string(is_pushup_score_),
        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, 10,
        cv::Scalar(0, 0, 0), cv::Scalar(220, 220, 220));
  }
  visualizers::BaseVisualizer::PutText(
      rgb, std::string("PushUps: ") + std::to_string(num_pushups_),
      cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, 10,
      cv::Scalar(255, 255, 255), cv::Scalar(255, 0, 0));
}

}  // namespace flows
}  // namespace daisykit
