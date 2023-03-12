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

#ifndef DAISYKIT_COMMON_VISUALIZERS_FACE_VISUALIZER_H_
#define DAISYKIT_COMMON_VISUALIZERS_FACE_VISUALIZER_H_

#include "daisykit/common/types.h"
#include "daisykit/common/visualizers/base_visualizer.h"

#include <opencv2/opencv.hpp>

namespace daisykit {
namespace visualizers {

template <typename FaceT>
class FaceVisualizer {
 public:
  static void DrawFace(cv::Mat& img, const std::vector<FaceT>& faces,
                       bool with_landmark = false) {
    for (auto face : faces) {
      std::string mask;
      if (face.wearing_mask_prob >= 0.0) {
        if (face.wearing_mask_prob < 0.5) {
          mask = "No Mask";
        } else {
          mask = "Mask";
        }
      }
      BaseVisualizer::DrawRoundedBox(img, static_cast<types::Box>(face), mask);
      if (with_landmark) {
        DrawLandmark(img, face.landmark);
      }
    }
  }

  static void DrawLandmark(cv::Mat& img,
                           const std::vector<types::Keypoint>& keypoints,
                           float threshold = 0.2) {
    // Draw joint
    for (size_t i = 0; i < keypoints.size(); i++) {
      const types::Keypoint& keypoint = keypoints[i];
      if (keypoint.confidence < threshold) continue;
      cv::circle(img, cv::Point(keypoint.x, keypoint.y), 4,
                 cv::Scalar(0, 255, 255), -1);
    }
  }
};

}  // namespace visualizers
}  // namespace daisykit

#endif
