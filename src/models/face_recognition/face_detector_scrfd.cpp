// Copyright 2021 The DaisyKit Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License

#include "daisykit/models/face_recognition/face_detector_scrfd.h"

namespace daisykit {
namespace models {

FaceDetectorSCRFD::FaceDetectorSCRFD(const char* param_buffer,
                                     const unsigned char* weight_buffer,
                                     int input_size, float score_threshold,
                                     float iou_threshold, bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_size, input_size) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  input_size_ = input_size;
}

FaceDetectorSCRFD::FaceDetectorSCRFD(const std::string& param_file,
                                     const std::string& weight_file,
                                     int input_size, float score_threshold,
                                     float iou_threshold, bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_size, input_size) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  input_size_ = input_size;
}

#if __ANDROID__
FaceDetectorSCRFD::FaceDetectorSCRFD(AAssetManager* mgr,
                                     const std::string& param_file,
                                     const std::string& weight_file,
                                     int input_size, float score_threshold,
                                     float iou_threshold, bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  input_size_ = input_size;
}

#endif

float FaceDetectorSCRFD::IntersectionArea(const daisykit::types::FaceDet& a,
                                          const daisykit::types::FaceDet& b) {
  float xx1 = std::max(a.boxes.tl().x, b.boxes.tl().x);
  float yy1 = std::max(a.boxes.tl().y, b.boxes.tl().y);
  float xx2 = std::min(a.boxes.br().x, b.boxes.br().x);
  float yy2 = std::min(a.boxes.br().y, b.boxes.br().y);
  float w = std::max(float(0), xx2 - xx1 + 1);
  float h = std::max(float(0), yy2 - yy1 + 1);
  return w * h;
}

void FaceDetectorSCRFD::QsortDescentInplace(
    std::vector<daisykit::types::FaceDet>& faceobjects, int left, int right) {
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].confidence;
  while (i <= j) {
    while (faceobjects[i].confidence > p) i++;
    while (faceobjects[j].confidence < p) j--;

    if (i <= j) {
      std::swap(faceobjects[i], faceobjects[j]);
      i++;
      j--;
    }
  }
  if (left < j) QsortDescentInplace(faceobjects, left, j);
  if (i < right) QsortDescentInplace(faceobjects, i, right);
}

void FaceDetectorSCRFD::QsortDescentInplace(
    std::vector<daisykit::types::FaceDet>& faceobjects) {
  if (faceobjects.empty()) return;
  QsortDescentInplace(faceobjects, 0, faceobjects.size() - 1);
}

void FaceDetectorSCRFD::NmsSortedBboxes(
    std::vector<daisykit::types::FaceDet>& faceobjects,
    std::vector<int>& picked, float nms_threshold) {
  picked.clear();
  const int n = faceobjects.size();
  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].Area();
  }
  for (int i = 0; i < n; i++) {
    const daisykit::types::FaceDet& a = faceobjects[i];
    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const daisykit::types::FaceDet& b = faceobjects[picked[j]];
      float inter_area = IntersectionArea(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      if (inter_area / union_area > nms_threshold) keep = 0;
    }
    if (keep) picked.push_back(i);
  }
}

ncnn::Mat FaceDetectorSCRFD::GenerateAnchors(int base_size,
                                             const ncnn::Mat& ratios,
                                             const ncnn::Mat& scales) {
  int num_ratio = ratios.w;
  int num_scale = scales.w;

  ncnn::Mat anchors;
  anchors.create(4, num_ratio * num_scale);

  const float cx = 0;
  const float cy = 0;

  for (int i = 0; i < num_ratio; i++) {
    float ar = ratios[i];

    int r_w = round(base_size / sqrt(ar));
    int r_h = round(r_w * ar);

    for (int j = 0; j < num_scale; j++) {
      float scale = scales[j];

      float rs_w = r_w * scale;
      float rs_h = r_h * scale;

      float* anchor = anchors.row(i * num_scale + j);

      anchor[0] = cx - rs_w * 0.5f;
      anchor[1] = cy - rs_h * 0.5f;
      anchor[2] = cx + rs_w * 0.5f;
      anchor[3] = cy + rs_h * 0.5f;
    }
  }
  return anchors;
}

void FaceDetectorSCRFD::GenerateProposals(
    const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob,
    const ncnn::Mat& bbox_blob, const ncnn::Mat& kps_blob, float prob_threshold,
    std::vector<daisykit::types::FaceDet>& faceobjects) {
  int w = score_blob.w;
  int h = score_blob.h;

  const int num_anchors = anchors.h;

  for (int q = 0; q < num_anchors; q++) {
    const float* anchor = anchors.row(q);
    const ncnn::Mat score = score_blob.channel(q);
    const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);

    float anchor_y = anchor[1];
    float anchor_w = anchor[2] - anchor[0];
    float anchor_h = anchor[3] - anchor[1];

    for (int i = 0; i < h; i++) {
      float anchor_x = anchor[0];
      for (int j = 0; j < w; j++) {
        int index = i * w + j;
        float prob = score[index];

        if (prob >= prob_threshold) {
          float dx = bbox.channel(0)[index] * feat_stride;
          float dy = bbox.channel(1)[index] * feat_stride;
          float dw = bbox.channel(2)[index] * feat_stride;
          float dh = bbox.channel(3)[index] * feat_stride;
          float cx = anchor_x + anchor_w * 0.5f;
          float cy = anchor_y + anchor_h * 0.5f;
          float x0 = cx - dx;
          float y0 = cy - dy;
          float x1 = cx + dw;
          float y1 = cy + dh;

          daisykit::types::FaceDet obj;
          obj.boxes = cv::Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0));

          obj.confidence = prob;

          if (!kps_blob.empty()) {
            const ncnn::Mat kps = kps_blob.channel_range(q * 10, 10);
            obj.landmark.x[0] = cx + kps.channel(0)[index] * feat_stride;
            obj.landmark.y[0] = cy + kps.channel(1)[index] * feat_stride;
            obj.landmark.x[1] = cx + kps.channel(2)[index] * feat_stride;
            obj.landmark.y[1] = cy + kps.channel(3)[index] * feat_stride;
            obj.landmark.x[2] = cx + kps.channel(4)[index] * feat_stride;
            obj.landmark.y[2] = cy + kps.channel(5)[index] * feat_stride;
            obj.landmark.x[3] = cx + kps.channel(6)[index] * feat_stride;
            obj.landmark.y[3] = cy + kps.channel(7)[index] * feat_stride;
            obj.landmark.x[4] = cx + kps.channel(8)[index] * feat_stride;
            obj.landmark.y[4] = cy + kps.channel(9)[index] * feat_stride;
          }

          faceobjects.push_back(obj);
        }
        anchor_x += feat_stride;
      }
      anchor_y += feat_stride;
    }
  }
}

void FaceDetectorSCRFD::Preprocess(const cv::Mat& image, ncnn::Mat& net_input) {
  cv::Mat rgb;
  cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  int width = rgb.cols;
  int height = rgb.rows;

  int w = width;
  int h = height;
  scale_ = 1.f;
  if (w > h) {
    scale_ = (float)input_size_ / w;
    w = input_size_;
    h = h * scale_;
  } else {
    scale_ = (float)input_size_ / h;
    h = input_size_;
    w = w * scale_;
  }
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,
                                               width, height, w, h);
  wpad_ = (w + 31) / 32 * 32 - w;
  hpad_ = (h + 31) / 32 * 32 - h;
  ncnn::copy_make_border(in, net_input, hpad_ / 2, hpad_ - hpad_ / 2, wpad_ / 2,
                         wpad_ - wpad_ / 2, ncnn::BORDER_CONSTANT, 0.f);
  const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
  const float norm_vals[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};
  net_input.substract_mean_normalize(mean_vals, norm_vals);
}

std::vector<daisykit::types::FaceDet> FaceDetectorSCRFD::Predict(cv::Mat& img) {
  std::vector<daisykit::types::FaceDet> faceobjects;
  std::vector<daisykit::types::FaceDet> faceproposals;
  ncnn::Mat in;
  std::map<std::string, ncnn::Mat> out;
  int width = img.cols;
  int height = img.rows;

  Preprocess(img, in);
  int res = Infer(in, out, input_name_, output_names_);
  if (res != 0) {
    std::cerr << "Inference failed" << std::endl;
    return faceobjects;
  }

  for (int i = 0; i < 3; i++) {
    ncnn::Mat score_blob = (ncnn::Mat)out[output_names_[0 + i * 3]];
    ncnn::Mat bbox_blob = (ncnn::Mat)out[output_names_[1 + i * 3]];
    ncnn::Mat kps_blob;

    if (is_landmark_) kps_blob = (ncnn::Mat)out[output_names_[2 + i * 3]];
    const int base_size = 16 * pow(2, i * 2);
    const int feat_stride = 8 * pow(2, i);
    ncnn::Mat ratios(1);
    ratios[0] = 1.f;
    ncnn::Mat scales(2);
    scales[0] = 1.f;
    scales[1] = 2.f;
    ncnn::Mat anchors = GenerateAnchors(base_size, ratios, scales);
    std::vector<daisykit::types::FaceDet> objects;
    GenerateProposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob,
                      score_threshold_, objects);
    faceproposals.insert(faceproposals.end(), objects.begin(), objects.end());
  }

  QsortDescentInplace(faceproposals);
  std::vector<int> picked;
  NmsSortedBboxes(faceproposals, picked, iou_threshold_);
  int face_count = picked.size();
  faceobjects.resize(face_count);
  for (int i = 0; i < face_count; i++) {
    faceobjects[i] = faceproposals[picked[i]];
    float x0 = (faceobjects[i].boxes.tl().x - (wpad_ / 2)) / scale_;
    float y0 = (faceobjects[i].boxes.tl().y - (hpad_ / 2)) / scale_;
    float x1 = (faceobjects[i].boxes.br().x - (wpad_ / 2)) / scale_;
    float y1 = (faceobjects[i].boxes.br().y - (hpad_ / 2)) / scale_;

    x0 = std::max(std::min(x0, (float)width - 1), 0.f);
    y0 = std::max(std::min(y0, (float)height - 1), 0.f);
    x1 = std::max(std::min(x1, (float)width - 1), 0.f);
    y1 = std::max(std::min(y1, (float)height - 1), 0.f);

    faceobjects[i].boxes =
        cv::Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0));

    if (is_landmark_) {
      float x0 = (faceobjects[i].landmark.x[0] - (wpad_ / 2)) / scale_;
      float y0 = (faceobjects[i].landmark.y[0] - (hpad_ / 2)) / scale_;
      float x1 = (faceobjects[i].landmark.x[1] - (wpad_ / 2)) / scale_;
      float y1 = (faceobjects[i].landmark.y[1] - (hpad_ / 2)) / scale_;
      float x2 = (faceobjects[i].landmark.x[2] - (wpad_ / 2)) / scale_;
      float y2 = (faceobjects[i].landmark.y[2] - (hpad_ / 2)) / scale_;
      float x3 = (faceobjects[i].landmark.x[3] - (wpad_ / 2)) / scale_;
      float y3 = (faceobjects[i].landmark.y[3] - (hpad_ / 2)) / scale_;
      float x4 = (faceobjects[i].landmark.x[4] - (wpad_ / 2)) / scale_;
      float y4 = (faceobjects[i].landmark.y[4] - (hpad_ / 2)) / scale_;

      faceobjects[i].landmark.x[0] =
          std::max(std::min(x0, (float)width - 1), 0.f);
      faceobjects[i].landmark.y[0] =
          std::max(std::min(y0, (float)height - 1), 0.f);
      faceobjects[i].landmark.x[1] =
          std::max(std::min(x1, (float)width - 1), 0.f);
      faceobjects[i].landmark.y[1] =
          std::max(std::min(y1, (float)height - 1), 0.f);
      faceobjects[i].landmark.x[2] =
          std::max(std::min(x2, (float)width - 1), 0.f);
      faceobjects[i].landmark.y[2] =
          std::max(std::min(y2, (float)height - 1), 0.f);
      faceobjects[i].landmark.x[3] =
          std::max(std::min(x3, (float)width - 1), 0.f);
      faceobjects[i].landmark.y[3] =
          std::max(std::min(y3, (float)height - 1), 0.f);
      faceobjects[i].landmark.x[4] =
          std::max(std::min(x4, (float)width - 1), 0.f);
      faceobjects[i].landmark.y[4] =
          std::max(std::min(y4, (float)height - 1), 0.f);
    }
  }

  return faceobjects;
}
void FaceDetectorSCRFD::DrawFaceDets(
    cv::Mat& img, std::vector<daisykit::types::FaceDet>& dets) {
  for (int i = 0; i < dets.size(); i++) {
    cv::rectangle(img, dets[i].boxes, cv::Scalar(0, 255, 0), 2);
    for (int j = 0; j < 5; j++) {
      cv::circle(img, cv::Point(dets[i].landmark.x[j], dets[i].landmark.y[j]),
                 2, cv::Scalar(0, 0, 255), 2);
    }
  }
}
}  // namespace models
}  // namespace daisykit
