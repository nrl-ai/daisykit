#include "daisykitsdk/utils/visualizer/viz_utils.h"

using namespace daisykit::utils::visualizer;

void VizUtils::DrawLabel(cv::Mat& im, const std::string label,
                         const cv::Point& origin, int fontface, double scale,
                         int thickness, int baseline, cv::Scalar text_color,
                         cv::Scalar bg_color) {
  cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
  cv::rectangle(im, origin + cv::Point(-5, baseline),
                origin + cv::Point(text.width, -text.height - 5), bg_color,
                cv::FILLED);
  cv::putText(im, label, origin, fontface, scale, text_color, thickness, 8);
}
