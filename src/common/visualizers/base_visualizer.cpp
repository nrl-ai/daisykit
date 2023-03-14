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

#include "daisykit/common/visualizers/base_visualizer.h"

namespace daisykit {
namespace visualizers {
void BaseVisualizer::PutText(cv::Mat& im, const std::string label,
                             const cv::Point& origin, int fontface,
                             double scale, int thickness, int baseline,
                             const cv::Scalar text_color,
                             const cv::Scalar bg_color) {
  // Auto scale text size
  if (scale < 0) {
    scale = 0.8 * im.rows / 1000.0;
    scale = std::max(scale, 0.4);
  }
  cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
  cv::rectangle(im, origin + cv::Point(-5, baseline),
                origin + cv::Point(text.width, -text.height - 5), bg_color,
                cv::FILLED);
  cv::putText(im, label, origin, fontface, scale, text_color, thickness, 8);
}

void BaseVisualizer::DrawBox(cv::Mat& img, const types::Box box,
                             const std::string& text,
                             const cv::Scalar line_color,
                             const cv::Scalar text_color, float box_line_width,
                             int text_fontface, double text_scale,
                             int text_thickness, int text_baseline) {
  int box_line_width_px;
  if (box_line_width < 1) {
    box_line_width_px = box_line_width * std::min(box.w, box.h);
  } else {
    box_line_width_px = box_line_width;
  }
  box_line_width_px = std::max(box_line_width_px, 3);
  cv::rectangle(img, cv::Rect(box.x, box.y, box.w, box.h), line_color,
                box_line_width);

  if (text != "") {
    visualizers::BaseVisualizer::PutText(
        img, text, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX,
        text_scale, text_thickness, text_baseline, text_color, line_color);
  }
}

void BaseVisualizer::DrawRoundedBox(cv::Mat& image, const types::Box box,
                                    const std::string& text,
                                    const cv::Scalar line_color,
                                    const cv::Scalar text_color,
                                    float box_line_width, int text_fontface,
                                    double text_scale, int text_thickness,
                                    int text_baseline, float border_radius) {
  /* corners:
   * p1 - p2
   * |     |
   * p4 - p3
   */
  cv::Point p1 = cv::Point(box.x, box.y);
  cv::Point p2 = cv::Point(box.x + box.w, box.y);
  cv::Point p3 = cv::Point(box.x + box.w, box.y + box.h);
  cv::Point p4 = cv::Point(box.x, box.y + box.h);

  int border_radius_px;
  if (border_radius < 1) {
    border_radius_px = border_radius * std::min(box.w, box.h);
  } else {
    border_radius_px = border_radius;
  }
  border_radius_px = std::max(border_radius_px, 10);

  int box_line_width_px;
  if (box_line_width < 1) {
    box_line_width_px = box_line_width * std::min(box.w, box.h);
  } else {
    box_line_width_px = box_line_width;
  }
  box_line_width_px = std::max(box_line_width_px, 3);

  auto line_type = cv::LINE_8;
  cv::line(image, cv::Point(p1.x + border_radius_px, p1.y),
           cv::Point(p2.x - border_radius_px, p2.y), line_color,
           box_line_width_px, line_type);
  cv::line(image, cv::Point(p2.x, p2.y + border_radius_px),
           cv::Point(p3.x, p3.y - border_radius_px), line_color,
           box_line_width_px, line_type);
  cv::line(image, cv::Point(p4.x + border_radius_px, p4.y),
           cv::Point(p3.x - border_radius_px, p3.y), line_color,
           box_line_width_px, line_type);
  cv::line(image, cv::Point(p1.x, p1.y + border_radius_px),
           cv::Point(p4.x, p4.y - border_radius_px), line_color,
           box_line_width_px, line_type);

  // Draw arcs
  cv::ellipse(image, p1 + cv::Point(border_radius_px, border_radius_px),
              cv::Size(border_radius_px, border_radius_px), 180.0, 0, 90,
              line_color, box_line_width_px, line_type);
  cv::ellipse(image, p2 + cv::Point(-border_radius_px, border_radius_px),
              cv::Size(border_radius_px, border_radius_px), 270.0, 0, 90,
              line_color, box_line_width_px, line_type);
  cv::ellipse(image, p3 + cv::Point(-border_radius_px, -border_radius_px),
              cv::Size(border_radius_px, border_radius_px), 0.0, 0, 90,
              line_color, box_line_width_px, line_type);
  cv::ellipse(image, p4 + cv::Point(border_radius_px, -border_radius_px),
              cv::Size(border_radius_px, border_radius_px), 90.0, 0, 90,
              line_color, box_line_width_px, line_type);

  if (text != "") {
    visualizers::BaseVisualizer::PutText(
        image, text, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX,
        text_scale, text_thickness, text_baseline, text_color, line_color);
  }
}

}  // namespace visualizers
}  // namespace daisykit
