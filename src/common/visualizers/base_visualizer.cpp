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

#include "daisykitsdk/common/visualizers/base_visualizer.h"

namespace daisykit {
namespace visualizers {

void BaseVisualizer::PutText(cv::Mat& im, const std::string label,
                             const cv::Point& origin, int fontface,
                             double scale, int thickness, int baseline,
                             cv::Scalar text_color, cv::Scalar bg_color) {
  cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
  cv::rectangle(im, origin + cv::Point(-5, baseline),
                origin + cv::Point(text.width, -text.height - 5), bg_color,
                cv::FILLED);
  cv::putText(im, label, origin, fontface, scale, text_color, thickness, 8);
}

void BaseVisualizer::DrawBox(cv::Mat& img, const types::Box box,
                             const std::string& text, cv::Scalar box_color,
                             cv::Scalar text_color, int box_line_width,
                             int text_fontface, double text_scale,
                             int text_thickness, int text_baseline) {
  cv::rectangle(img, cv::Rect(box.x, box.y, box.w, box.h), box_color,
                box_line_width);
  if (!text.empty()) {
    visualizers::BaseVisualizer::PutText(
        img, text, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX,
        text_scale, text_thickness, text_baseline, text_color, box_color);
  }
}

}  // namespace visualizers
}  // namespace daisykit
