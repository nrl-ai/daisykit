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

#ifndef DAISYKIT_COMMON_VISUALIZERS_BASE_VISUALIZER_H_
#define DAISYKIT_COMMON_VISUALIZERS_BASE_VISUALIZER_H_

#include "daisykitsdk/common/types.h"

#include <opencv2/opencv.hpp>

namespace daisykit {
namespace visualizers {

/// Basic visualization utilities
class BaseVisualizer {
 public:
  /// Draw text with background.
  static void PutText(
      cv::Mat& img,                                       /// Image to draw on
      const std::string label,                            /// Draw content
      const cv::Point& origin,                            /// Position
      int fontface = cv::FONT_HERSHEY_SIMPLEX,            /// Text font face
      double scale = 0.8,                                 /// Text scale
      int thickness = 1,                                  /// Text thickness
      int baseline = 0,                                   /// Y axis shift
      cv::Scalar text_color = cv::Scalar(255, 255, 255),  /// Text color
      cv::Scalar bg_color = cv::Scalar(255, 0, 0)         /// Background color
  );

  /// Draw a box with label.
  static void DrawBox(
      cv::Mat& img,                                  /// Image to draw on
      const types::Box box,                          /// Box to draw
      const std::string& text = "",                  /// Label text of the box
      cv::Scalar box_color = cv::Scalar(255, 0, 0),  /// Background color
      cv::Scalar text_color = cv::Scalar(255, 255, 255),  /// Text color
      int box_line_width = 2,                             /// Box line width
      int text_fontface = cv::FONT_HERSHEY_SIMPLEX,       /// Text font face
      double text_scale = 0.8,                            /// Text scale
      int text_thickness = 2,                             /// Text thickness
      int text_baseline = 0                               /// Y axis shift
  );

  /// Draw a line graph.
  template <typename T>
  static cv::Mat LineGraph(
      std::vector<T>& vals,   /// Values to draw
      int y_range[2],         /// Range of values - y-axis
      int n_last_values = -1  /// Number of values from the end of values
                              /// vector. Set to -1 to show all values.
  ) {
    auto begin_elm = n_last_values == -1 ? vals.begin()
                                         : vals.size() > n_last_values
                                               ? vals.end() - n_last_values
                                               : vals.begin();
    auto it = minmax_element(begin_elm, vals.end());
    float scale = 1. / ceil(*it.second - *it.first);
    float bias = *it.first;
    int rows = y_range[1] - y_range[0] + 1;

    int graph_y_size = n_last_values != -1
                           ? std::min(n_last_values, (int)vals.size())
                           : vals.size();
    cv::Mat image = cv::Mat::zeros(rows, graph_y_size, CV_8UC3);
    image.setTo(0);

    int k = 0;
    if (n_last_values != -1) {
      if (vals.size() > n_last_values) {
        k = vals.size() - n_last_values;
      }
    }

    for (int i = k; i < (int)vals.size() - 1; i++) {
      cv::line(
          image,
          cv::Point(i - k, rows - 1 - (vals[i] - bias) * scale * y_range[1]),
          cv::Point(i + 1 - k,
                    rows - 1 - (vals[i + 1] - bias) * scale * y_range[1]),
          cv::Scalar(255, 0, 0), 1);
    }

    return image;
  }
};  // namespace visualizers

}  // namespace visualizers
}  // namespace daisykit

#endif
