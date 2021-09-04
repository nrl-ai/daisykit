#include <daisykitsdk/models/action_classifier.h>

using namespace daisykit::common;
using namespace daisykit::models;
using namespace daisykit::utils::img_proc;

ActionClassifier::ActionClassifier(const std::string& param_file,
                                   const std::string& weight_file,
                                   bool smooth) {
  _smooth = smooth;
  LoadModel(param_file, weight_file);
}

void ActionClassifier::LoadModel(const std::string& param_file,
                                  const std::string& weight_file) {
  if (model_) {
    delete model_;
    model_ = nullptr;
  }
  model_ = new ncnn::Net;
  model_->load_param(param_file.c_str());
  model_->load_model(weight_file.c_str());
}

#ifdef __ANDROID__
ActionClassifier::ActionClassifier(AAssetManager* mgr,
                                   const std::string& param_file,
                                   const std::string& weight_file,
                                   bool smooth) {
  _smooth = smooth;
  LoadModel(mgr, param_file, weight_file);
}

void ActionClassifier::LoadModel(AAssetManager* mgr,
                                  const std::string& param_file,
                                  const std::string& weight_file) {
  if (model_) {
    delete model_;
    model_ = nullptr;
  }
  model_ = new ncnn::Net;
  model_->load_param(mgr, param_file.c_str());
  model_->load_model(mgr, weight_file.c_str());
}
#endif

Action ActionClassifier::Classify(cv::Mat& image, float& confidence) {
  cv::Mat rgb = image.clone();
  rgb = ImgUtils::SquarePadding(rgb, input_width_).clone();
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows, input_width_,
      input_height_);

  ncnn::Mat out;
  { 
    ncnn::MutexLockGuard g(lock_);
    ncnn::Extractor ex = model_->create_extractor();
    ex.set_num_threads(4);
    ex.input("input_1_blob", in);
    ex.extract("dense_Softmax_blob", out);
  }
  
  out = out.reshape(out.w * out.h * out.c);
  confidence = out[1];
  bool is_pushup = confidence > 0.9;

  if (!_smooth) {
    return is_pushup ? Action::kPushup : Action::kUnknown;
  }

  // Check and recognize pushup
  long long int current_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (is_pushup) {
    last_pushup_time_ = current_time;
  }

  // Return smoothed result
  if (current_time - last_pushup_time_ < 2000) {
    return Action::kPushup;
  }

  return Action::kUnknown;
}

Action ActionClassifier::Classify(cv::Mat& image) {
  float confidence;
  return Classify(image, confidence);
}