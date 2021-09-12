#ifndef DAISYKIT_MODELS_BODY_DETECTOR_H_
#define DAISYKIT_MODELS_BODY_DETECTOR_H_

#include <daisykitsdk/common/types.h>
#include <daisykitsdk/models/base_model.h>
#include <opencv2/opencv.hpp>

namespace daisykit {
namespace models {

class BodyDetector
    : public BaseModel<cv::Mat, std::vector<daisykit::common::Object>> {
 public:
  BodyDetector(const char* param_buffer, const unsigned char* weight_buffer,
               const int& width = 320, const int& height = 320);

  // Overide abstract Predict
  virtual std::vector<daisykit::common::Object> Predict(cv::Mat& image);

 private:
  int input_width_;
  int input_height_;
};

}  // namespace models
}  // namespace daisykit

#endif