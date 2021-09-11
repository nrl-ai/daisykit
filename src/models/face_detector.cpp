#include <daisykitsdk/models/face_detector.h>

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

using namespace daisykit::common;
using namespace daisykit::models;

FaceDetector::FaceDetector(const std::string& param_file,
                           const std::string& weight_file, int input_width,
                           int input_height, float score_threshold,
                           float iou_threshold) {
  InitParams(input_width, input_height, score_threshold, iou_threshold);
  LoadModel(param_file, weight_file);
}

void FaceDetector::LoadModel(const std::string& param_file,
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
FaceDetector::FaceDetector(AAssetManager* mgr, const std::string& param_file,
                           const std::string& weight_file, int input_width,
                           int input_height, float score_threshold,
                           float iou_threshold) {
  InitParams(input_width, input_height, score_threshold, iou_threshold);
  LoadModel(mgr, param_file, weight_file);
}

void FaceDetector::LoadModel(AAssetManager* mgr, const std::string& param_file,
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

void FaceDetector::InitParams(int input_width, int input_height,
                              float score_threshold, float iou_threshold) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  input_width_ = input_width;
  input_height_ = input_height;
  w_h_list_ = {input_width_, input_height_};

  for (auto size : w_h_list_) {
    std::vector<float> fm_item;
    for (float stride : strides) {
      fm_item.push_back(ceil(size / stride));
    }
    featuremap_size_.push_back(fm_item);
  }

  for (auto size : w_h_list_) {
    shrinkage_size_.push_back(strides);
  }

  /* generate prior anchors */
  for (int index = 0; index < kNumFeaturemaps; index++) {
    float scale_w = input_width_ / shrinkage_size_[0][index];
    float scale_h = input_height_ / shrinkage_size_[1][index];
    for (int j = 0; j < featuremap_size_[1][index]; j++) {
      for (int i = 0; i < featuremap_size_[0][index]; i++) {
        float x_center = (i + 0.5) / scale_w;
        float y_center = (j + 0.5) / scale_h;

        for (float k : min_boxes_[index]) {
          float w = k / input_width_;
          float h = k / input_height_;
          priors_.push_back(
              {clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
        }
      }
    }
  }
  num_anchors_ = priors_.size();
}

std::vector<Face> FaceDetector::Detect(cv::Mat& image) {
  cv::Mat rgb = image.clone();
  ncnn::Mat inmat = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB,
                                           rgb.cols, rgb.rows);
  int image_width = inmat.w;
  int image_height = inmat.h;

  ncnn::Mat in;
  ncnn::resize_bilinear(inmat, in, input_width_, input_height_);
  ncnn::Mat ncnn_img = in;
  ncnn_img.substract_mean_normalize(mean_vals_, norm_vals_);

  std::vector<Face> bbox_collection;
  std::vector<Face> face_list;

  ncnn::Extractor ex = model_->create_extractor();
  ex.input("input", ncnn_img);

  ncnn::Mat scores;
  ncnn::Mat boxes;
  ex.extract("scores", scores);
  ex.extract("boxes", boxes);
  GenerateBBox(bbox_collection, scores, boxes, score_threshold_, num_anchors_,
               image_width, image_height);
  Nms(bbox_collection, face_list);

  return face_list;
}

void FaceDetector::GenerateBBox(std::vector<Face>& bbox_collection,
                                ncnn::Mat scores, ncnn::Mat boxes,
                                float score_threshold, int num_anchors,
                                int image_width, int image_height) {
  for (int i = 0; i < num_anchors; i++) {
    if (scores.channel(0)[i * 2 + 1] > score_threshold) {
      Face face;
      float x_center =
          boxes.channel(0)[i * 4] * center_variance_ * priors_[i][2] +
          priors_[i][0];
      float y_center =
          boxes.channel(0)[i * 4 + 1] * center_variance_ * priors_[i][3] +
          priors_[i][1];
      float w =
          exp(boxes.channel(0)[i * 4 + 2] * size_variance_) * priors_[i][2];
      float h =
          exp(boxes.channel(0)[i * 4 + 3] * size_variance_) * priors_[i][3];

      face.x = clip(x_center - w / 2.0, 1) * image_width;
      face.y = clip(y_center - h / 2.0, 1) * image_height;
      int x2 = clip(x_center + w / 2.0, 1) * image_width;
      int y2 = clip(y_center + h / 2.0, 1) * image_height;
      face.w = x2 - face.x;
      face.h = y2 - face.y;

      face.confidence = clip(scores.channel(0)[i * 2 + 1], 1);
      bbox_collection.push_back(face);
    }
  }
}

void FaceDetector::Nms(std::vector<Face>& input, std::vector<Face>& output,
                       int type) {
  std::sort(input.begin(), input.end(), [](const Face& a, const Face& b) {
    return a.confidence > b.confidence;
  });

  int box_num = input.size();

  std::vector<int> merged(box_num, 0);

  for (int i = 0; i < box_num; i++) {
    if (merged[i]) continue;
    std::vector<Face> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    float h0 = input[i].h + 1;
    float w0 = input[i].w + 1;

    float area0 = h0 * w0;

    for (int j = i + 1; j < box_num; j++) {
      if (merged[j]) continue;

      float inner_x0 = std::max(input[i].x, input[j].x);
      float inner_y0 = std::max(input[i].y, input[j].y);

      float inner_x1 =
          std::min(input[i].x + input[i].w, input[j].x + input[j].w);
      float inner_y1 =
          std::min(input[i].y + input[i].h, input[j].y + input[j].h);

      float inner_h = inner_y1 - inner_y0 + 1;
      float inner_w = inner_x1 - inner_x0 + 1;

      if (inner_h <= 0 || inner_w <= 0) continue;

      float inner_area = inner_h * inner_w;

      float h1 = input[j].h + 1;
      float w1 = input[j].w + 1;

      float area1 = h1 * w1;

      float score;

      score = inner_area / (area0 + area1 - inner_area);

      if (score > iou_threshold_) {
        merged[j] = 1;
        buf.push_back(input[j]);
      }
    }
    switch (type) {
      case NmsMethod::kHardNms: {
        output.push_back(buf[0]);
        break;
      }
      case NmsMethod::kBlendingNms: {
        float total = 0;
        for (int i = 0; i < buf.size(); i++) {
          total += exp(buf[i].confidence);
        }
        Face rects;
        memset(&rects, 0, sizeof(rects));
        for (int i = 0; i < buf.size(); i++) {
          float rate = exp(buf[i].confidence) / total;
          rects.x += buf[i].x * rate;
          rects.y += buf[i].y * rate;
          rects.w += buf[i].w * rate;
          rects.h += buf[i].h * rate;
          rects.confidence += buf[i].confidence * rate;
        }
        output.push_back(rects);
        break;
      }
      default: {
        std::cerr << "Wrong NMS type." << std::endl;
        exit(-1);
      }
    }
  }
}