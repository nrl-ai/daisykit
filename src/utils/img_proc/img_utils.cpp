#include <daisykitsdk/utils/img_proc/img_utils.h>

using namespace daisykit::utils::img_proc;

cv::Mat ImgUtils::SquarePadding(const cv::Mat& img, int target_width) {
  int width = img.cols, height = img.rows;

  cv::Mat square = cv::Mat::zeros(target_width, target_width, img.type());

  int max_dim = (width >= height) ? width : height;
  float scale = ((float)target_width) / max_dim;
  cv::Rect roi;
  if (width >= height) {
    roi.width = target_width;
    roi.x = 0;
    roi.height = height * scale;
    roi.y = (target_width - roi.height) / 2;
  } else {
    roi.y = 0;
    roi.height = target_width;
    roi.width = width * scale;
    roi.x = (target_width - roi.width) / 2;
  }

  cv::resize(img, square(roi), roi.size());

  return square;
}