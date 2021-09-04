#ifndef VIZ_UTILS_
#define VIZ_UTILS_

#include <opencv2/opencv.hpp>

class VizUtils {
 public:
  static void draw_label(cv::Mat& im, const std::string label,
                         const cv::Point& origin,
                         int fontface = cv::FONT_HERSHEY_SIMPLEX,
                         double scale = 0.8, int thickness = 1,
                         int baseline = 0,
                         cv::Scalar text_color = cv::Scalar(255, 255, 255),
                         cv::Scalar bg_color = cv::Scalar(255, 0, 0));
};

#endif
