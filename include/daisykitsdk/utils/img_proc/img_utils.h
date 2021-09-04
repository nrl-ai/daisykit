#ifndef DAISYKIT_UTILS_IMG_PROC_IMG_UTILS_H_
#define DAISYKIT_UTILS_IMG_PROC_IMG_UTILS_H_

#include <opencv2/opencv.hpp>

namespace daisykit {
namespace utils {
namespace img_proc {

class ImgUtils {
 public:
  static cv::Mat SquarePadding(const cv::Mat& img, int target_width = 500);
};

}  // namespace img_proc
}  // namespace utils
}  // namespace daisykit

#endif
