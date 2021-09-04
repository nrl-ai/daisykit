#include <daisykitsdk/models/action_classifier.h>

using namespace daisykit::common;
using namespace daisykit::models;

ActionClassifier::ActionClassifier(const std::string& param_file,
                                   const std::string& weight_file,
                                   bool smooth) {
  _smooth = smooth;
  load_model(param_file, weight_file);
}

void ActionClassifier::load_model(const std::string& param_file,
                                  const std::string& weight_file) {
  if (_model) {
    delete _model;
  }
  _model = new ncnn::Net;
  _model->load_param(param_file.c_str());
  _model->load_model(weight_file.c_str());
}

#ifdef __ANDROID__
ActionClassifier::ActionClassifier(AAssetManager* mgr,
                                   const std::string& param_file,
                                   const std::string& weight_file,
                                   bool smooth) {
  _smooth = smooth;
  load_model(mgr, param_file, weight_file);
}

void ActionClassifier::load_model(AAssetManager* mgr,
                                  const std::string& param_file,
                                  const std::string& weight_file) {
  if (_model) {
    delete _model;
  }
  _model = new ncnn::Net;
  _model->load_param(mgr, param_file.c_str());
  _model->load_model(mgr, weight_file.c_str());
}
#endif

Action ActionClassifier::classify(cv::Mat& image, float& confidence) {
  cv::Mat rgb = image.clone();
  rgb = square_padding(rgb, _input_width).clone();
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows, _input_width,
      _input_height);

  ncnn::Mat out;
  { 
    ncnn::MutexLockGuard g(_lock);
    ncnn::Extractor ex = _model->create_extractor();
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
    _last_pushup_time = current_time;
  }

  // Return smoothed result
  if (current_time - _last_pushup_time < 2000) {
    return Action::kPushup;
  }

  return Action::kUnknown;
}

Action ActionClassifier::classify(cv::Mat& image) {
  float confidence;
  return classify(image, confidence);
}

cv::Mat ActionClassifier::square_padding(const cv::Mat& img, int target_width) {
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