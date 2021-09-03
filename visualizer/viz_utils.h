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

void VizUtils::draw_label(cv::Mat& im, const std::string label,
                          const cv::Point& origin, int fontface, double scale,
                          int thickness, int baseline, cv::Scalar text_color,
                          cv::Scalar bg_color) {
  cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
  cv::rectangle(im, origin + cv::Point(-5, baseline),
                origin + cv::Point(text.width, -text.height - 5), bg_color,
                cv::FILLED);
  cv::putText(im, label, origin, fontface, scale, text_color, thickness, 8);
}

#endif
