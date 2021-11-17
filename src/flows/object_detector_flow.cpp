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

#include "daisykitsdk/flows/object_detector_flow.h"
#include "daisykitsdk/common/visualizers/base_visualizer.h"
#include "third_party/json.hpp"

namespace daisykit {
namespace flows {

ObjectDetectorFlow::ObjectDetectorFlow(const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  detector_ = new models::ObjectDetectorYOLOX(
      config["object_detection_model"]["model"],
      config["object_detection_model"]["weights"],
      config["object_detection_model"]["score_threshold"],
      config["object_detection_model"]["iou_threshold"],
      config["object_detection_model"]["input_width"],
      config["object_detection_model"]["input_height"],
      config["object_detection_model"]["use_gpu"]);
  detector_->SetClassNames(config["object_detection_model"]["class_names"]);
}

std::vector<types::Object> ObjectDetectorFlow::Process(cv::Mat& rgb) {
  // Detect
  std::vector<types::Object> objects;
  detector_->Predict(rgb, objects);

  return objects;
}

void ObjectDetectorFlow::DrawResult(cv::Mat& rgb,
                                    std::vector<types::Object>& objects) {
  for (auto object : objects) {
    daisykit::visualizers::BaseVisualizer::DrawBox(
        rgb, (types::Box)(object), detector_->GetClassNames()[object.class_id],
        cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0));
  }
}

}  // namespace flows
}  // namespace daisykit
