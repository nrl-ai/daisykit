#include <daisykitsdk/models/body_detector.h>

BodyDetector::BodyDetector(const std::string& param_file,
                           const std::string& weight_file) {
  load_model(param_file, weight_file);
}

void BodyDetector::load_model(const std::string& param_file,
                              const std::string& weight_file) {
  if (_model) {
    delete _model;
  }
  _model = new ncnn::Net;
  _model->load_param(param_file.c_str());
  _model->load_model(weight_file.c_str());
}

#ifdef __ANDROID__
BodyDetector::BodyDetector(AAssetManager* mgr, const std::string& param_file,
                           const std::string& weight_file) {
  load_model(mgr, param_file, weight_file);
}

void BodyDetector::load_model(AAssetManager* mgr, const std::string& param_file,
                              const std::string& weight_file) {
  if (_model) {
    delete _model;
  }
  _model = new ncnn::Net;
  _model->load_param(mgr, param_file.c_str());
  _model->load_model(mgr, weight_file.c_str());
}
#endif

std::vector<Object> BodyDetector::detect(cv::Mat& image) {
  cv::Mat rgb = image.clone();
  int img_w = rgb.cols;
  int img_h = rgb.rows;

  ncnn::Mat in =
      ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols,
                                    rgb.rows, _input_width, _input_height);

  const float mean_vals[3] = {0.f, 0.f, 0.f};
  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Extractor ex = _model->create_extractor();
  ex.set_num_threads(4);
  ex.input("data", in);
  ncnn::Mat out;
  ex.extract("output", out);

  std::vector<Object> objects;
  for (int i = 0; i < out.h; i++) {
    Object object;
    float x1, y1, x2, y2, score, label;
    float pw, ph, cx, cy;
    const float* values = out.row(i);

    x1 = values[2] * img_w;
    y1 = values[3] * img_h;
    x2 = values[4] * img_w;
    y2 = values[5] * img_h;

    pw = x2 - x1;
    ph = y2 - y1;
    cx = x1 + 0.5 * pw;
    cy = y1 + 0.5 * ph;

    x1 = cx - 0.7 * pw;
    y1 = cy - 0.6 * ph;
    x2 = cx + 0.7 * pw;
    y2 = cy + 0.6 * ph;

    object.confidence = values[1];
    object.class_id = values[0];
    object.x = x1;
    object.y = y1;
    object.w = x2 - x1;
    object.h = y2 - y1;

    objects.push_back(object);
  }

  return objects;
}