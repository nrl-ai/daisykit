#include <daisykitsdk/models/human_matting.h>

using namespace daisykit::common;
using namespace daisykit::models;

HumanMatting::HumanMatting(const std::string& param_file,
                           const std::string& weight_file) {
  LoadModel(param_file, weight_file);
}

void HumanMatting::LoadModel(const std::string& param_file,
                             const std::string& weight_file) {
  if (model_) {
    delete model_;
    model_ = nullptr;
  }
  model_ = new ncnn::Net;
  int ret_param = model_->load_param(param_file.c_str());
  int ret_model = model_->load_model(weight_file.c_str());
  if (ret_param != 0 || ret_model != 0) {
    exit(1);
  }
}

#ifdef __ANDROID__
HumanMatting::HumanMatting(AAssetManager* mgr, const std::string& param_file,
                           const std::string& weight_file) {
  LoadModel(mgr, param_file, weight_file);
}

void HumanMatting::LoadModel(AAssetManager* mgr, const std::string& param_file,
                             const std::string& weight_file) {
  if (model_) {
    delete model_;
    model_ = nullptr;
  }
  model_ = new ncnn::Net;
  int ret_param = model_->load_param(mgr, param_file.c_str());
  int ret_model = model_->load_model(mgr, weight_file.c_str());
  if (ret_param != 0 || ret_model != 0) {
    exit(1);
  }
}
#endif

void HumanMatting::Segmentation(cv::Mat& image, cv::Mat& mask) {
  cv::Mat rgb = image.clone();
  const int w = rgb.cols;
  const int h = rgb.rows;

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      rgb.data, ncnn::Mat::PIXEL_RGB2BGR, w, h, input_width_, input_height_);

  ncnn::Extractor ex = model_->create_extractor();
  ex.input("input_blob1", in);
  const float mean_vals[3] = {104.f, 112.f, 121.f};
  const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Mat out;
  ex.extract("sigmoid_blob1", out);

  const float denorm_vals[3] = {255.f, 255.f, 255.f};
  out.substract_mean_normalize(0, denorm_vals);

  mask = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);
  out.to_pixels_resize(mask.data, ncnn::Mat::PIXEL_GRAY, w, h);
}

void HumanMatting::BindWithBackground(cv::Mat& rgb, const cv::Mat& bg,
                                      const cv::Mat& mask) {
  const int w = rgb.cols;
  const int h = rgb.rows;

  for (int y = 0; y < h; y++) {
    uchar* rgbptr = rgb.ptr<uchar>(y);
    const uchar* bgptr = bg.ptr<const uchar>(y);
    const uchar* mptr = mask.ptr<const uchar>(y);

    for (int x = 0; x < w; x++) {
      const uchar alpha = mptr[0];

      rgbptr[0] = cv::saturate_cast<uchar>(
          (rgbptr[0] * alpha + bgptr[0] * (255 - alpha)) / 255);
      rgbptr[1] = cv::saturate_cast<uchar>(
          (rgbptr[1] * alpha + bgptr[1] * (255 - alpha)) / 255);
      rgbptr[2] = cv::saturate_cast<uchar>(
          (rgbptr[2] * alpha + bgptr[2] * (255 - alpha)) / 255);

      rgbptr += 3;
      bgptr += 3;
      mptr += 1;
    }
  }
}