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

#include "daisykitsdk/flows/human_pose_movenet_flow.h"
#include "daisykitsdk/common/visualizers/base_visualizer.h"
#include "third_party/json.hpp"

namespace daisykit {
namespace flows {

HumanPoseMoveNetFlow::HumanPoseMoveNetFlow(const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  body_detector_ =
      new models::BodyDetector(config["person_detection_model"]["model"],
                               config["person_detection_model"]["weights"]);
  pose_detector_ = new models::PoseDetectorMoveNet(
      config["human_pose_model"]["model"],
      config["human_pose_model"]["weights"],
      config["human_pose_model"]["input_width"],
      config["human_pose_model"]["input_height"]);
}

void HumanPoseMoveNetFlow::Process(cv::Mat& rgb) {
  // Detect background pose
  std::vector<types::Object> bodies;
  body_detector_->Predict(rgb, bodies);
  {
    const std::lock_guard<std::mutex> lock(bodies_lock_);
    bodies_ = bodies;
  }

  // Detect keypoints
  std::vector<std::vector<types::Keypoint>> keypoints;
  pose_detector_->PredictMulti(rgb, bodies, keypoints);
  {
    const std::lock_guard<std::mutex> lock(keypoints_lock_);
    keypoints_ = keypoints;
  }
}

void HumanPoseMoveNetFlow::DrawResult(cv::Mat& rgb) {
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
}

}  // namespace flows
}  // namespace daisykit
