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

template <class FaceT>
FaceDetectorSCRFD<FaceT>::FaceDetectorSCRFD(const char* param_buffer,
                                            const unsigned char* weight_buffer,
                                            int input_size,
                                            float score_threshold,
                                            float iou_threshold, bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_size, input_size) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  input_size_ = input_size;
}

template <class FaceT>
FaceDetectorSCRFD<FaceT>::FaceDetectorSCRFD(const std::string& param_file,
                                            const std::string& weight_file,
                                            int input_size,
                                            float score_threshold,
                                            float iou_threshold, bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_size, input_size) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  input_size_ = input_size;
}

#if __ANDROID__
template <class FaceT>
FaceDetectorSCRFD<FaceT>::FaceDetectorSCRFD(AAssetManager* mgr,
                                            const std::string& param_file,
                                            const std::string& weight_file,
                                            int input_size,
                                            float score_threshold,
                                            float iou_threshold, bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  input_size_ = input_size;
}

#endif
template <class FaceT>
float FaceDetectorSCRFD<FaceT>::IntersectionArea(const FaceT& a,
                                                 const FaceT& b) {
  float xx1 = std::max(a.x, b.x);
  float yy1 = std::max(a.y, b.y);
  float xx2 = std::min(a.x + a.w, b.x + b.w);
  float yy2 = std::min(a.y + a.h, b.y + b.h);
  float w = std::max(float(0), xx2 - xx1 + 1);
  float h = std::max(float(0), yy2 - yy1 + 1);
  return w * h;
}

template <class FaceT>
void FaceDetectorSCRFD<FaceT>::QsortDescentInplace(std::vector<FaceT>& faces,
                                                   int left, int right) {
  int i = left;
  int j = right;
  float p = faces[(left + right) / 2].confidence;
  while (i <= j) {
    while (faces[i].confidence > p) i++;
    while (faces[j].confidence < p) j--;

    if (i <= j) {
      std::swap(faces[i], faces[j]);
      i++;
      j--;
    }
  }
  if (left < j) QsortDescentInplace(faces, left, j);
  if (i < right) QsortDescentInplace(faces, i, right);
}

template <class FaceT>
void FaceDetectorSCRFD<FaceT>::QsortDescentInplace(std::vector<FaceT>& faces) {
  if (faces.empty()) return;
  QsortDescentInplace(faces, 0, faces.size() - 1);
}

template <class FaceT>
void FaceDetectorSCRFD<FaceT>::NmsSortedBboxes(std::vector<FaceT>& faces,
                                               std::vector<int>& picked,
                                               float nms_threshold) {
  picked.clear();
  const int n = faces.size();
  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faces[i].Area();
  }
  for (int i = 0; i < n; i++) {
    const FaceT& a = faces[i];
    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const FaceT& b = faces[picked[j]];
      float inter_area = IntersectionArea(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      if (inter_area / union_area > nms_threshold) keep = 0;
    }
    if (keep) picked.push_back(i);
  }
}

template <class FaceT>
ncnn::Mat FaceDetectorSCRFD<FaceT>::GenerateAnchors(int base_size,
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

template <class FaceT>
void FaceDetectorSCRFD<FaceT>::GenerateProposals(
    const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob,
    const ncnn::Mat& bbox_blob, const ncnn::Mat& kps_blob, float prob_threshold,
    std::vector<FaceT>& faces) {
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

          FaceT obj;
          obj.x = int(x0);
          obj.y = int(y0);
          obj.w = int(x1 - x0);
          obj.h = int(y1 - y0);
          obj.confidence = prob;

          if (!kps_blob.empty()) {
            obj.landmark.resize(5);
            const ncnn::Mat kps = kps_blob.channel_range(q * 10, 10);
            obj.landmark[0].x = cx + kps.channel(0)[index] * feat_stride;
            obj.landmark[0].y = cy + kps.channel(1)[index] * feat_stride;
            obj.landmark[1].x = cx + kps.channel(2)[index] * feat_stride;
            obj.landmark[1].y = cy + kps.channel(3)[index] * feat_stride;
            obj.landmark[2].x = cx + kps.channel(4)[index] * feat_stride;
            obj.landmark[2].y = cy + kps.channel(5)[index] * feat_stride;
            obj.landmark[3].x = cx + kps.channel(6)[index] * feat_stride;
            obj.landmark[3].y = cy + kps.channel(7)[index] * feat_stride;
            obj.landmark[4].x = cx + kps.channel(8)[index] * feat_stride;
            obj.landmark[4].y = cy + kps.channel(9)[index] * feat_stride;
            obj.landmark[0].confidence = 1.0;
            obj.landmark[1].confidence = 1.0;
            obj.landmark[2].confidence = 1.0;
            obj.landmark[3].confidence = 1.0;
            obj.landmark[4].confidence = 1.0;
          }

          faces.push_back(obj);
        }
        anchor_x += feat_stride;
      }
      anchor_y += feat_stride;
    }
  }
}

template <class FaceT>
void FaceDetectorSCRFD<FaceT>::Preprocess(const cv::Mat& image,
                                          ncnn::Mat& net_input) {
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

template <class FaceT>
int FaceDetectorSCRFD<FaceT>::Predict(const cv::Mat& image,
                                      std::vector<FaceT>& faces) {
  std::vector<FaceT> proposals;
  ncnn::Mat in;
  std::map<std::string, ncnn::Mat> out;
  int width = image.cols;
  int height = image.rows;

  Preprocess(image, in);
  int res = Infer(in, out, input_name_, output_names_);
  if (res != 0) {
    std::cerr << "Inference failed" << std::endl;
    return -1;
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
    std::vector<FaceT> objects;
    GenerateProposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob,
                      score_threshold_, objects);
    proposals.insert(proposals.end(), objects.begin(), objects.end());
  }

  QsortDescentInplace(proposals);
  std::vector<int> picked;
  NmsSortedBboxes(proposals, picked, iou_threshold_);
  int face_count = picked.size();
  faces.resize(face_count);
  for (int i = 0; i < face_count; i++) {
    faces[i] = proposals[picked[i]];
    float x0 = (faces[i].x - (wpad_ / 2)) / scale_;
    float y0 = (faces[i].y - (hpad_ / 2)) / scale_;
    float x1 = (faces[i].x + faces[i].w - (wpad_ / 2)) / scale_;
    float y1 = (faces[i].y + faces[i].h - (hpad_ / 2)) / scale_;

    x0 = std::max(std::min(x0, (float)width - 1), 0.f);
    y0 = std::max(std::min(y0, (float)height - 1), 0.f);
    x1 = std::max(std::min(x1, (float)width - 1), 0.f);
    y1 = std::max(std::min(y1, (float)height - 1), 0.f);

    faces[i].x = int(x0);
    faces[i].y = int(y0);
    faces[i].w = int(x1 - x0);
    faces[i].h = int(y1 - y0);

    if (is_landmark_) {
      float x0 = (faces[i].landmark[0].x - (wpad_ / 2)) / scale_;
      float y0 = (faces[i].landmark[0].y - (hpad_ / 2)) / scale_;
      float x1 = (faces[i].landmark[1].x - (wpad_ / 2)) / scale_;
      float y1 = (faces[i].landmark[1].y - (hpad_ / 2)) / scale_;
      float x2 = (faces[i].landmark[2].x - (wpad_ / 2)) / scale_;
      float y2 = (faces[i].landmark[2].y - (hpad_ / 2)) / scale_;
      float x3 = (faces[i].landmark[3].x - (wpad_ / 2)) / scale_;
      float y3 = (faces[i].landmark[3].y - (hpad_ / 2)) / scale_;
      float x4 = (faces[i].landmark[4].x - (wpad_ / 2)) / scale_;
      float y4 = (faces[i].landmark[4].y - (hpad_ / 2)) / scale_;

      faces[i].landmark[0].x = std::max(std::min(x0, (float)width - 1), 0.f);
      faces[i].landmark[0].y = std::max(std::min(y0, (float)height - 1), 0.f);
      faces[i].landmark[1].x = std::max(std::min(x1, (float)width - 1), 0.f);
      faces[i].landmark[1].y = std::max(std::min(y1, (float)height - 1), 0.f);
      faces[i].landmark[2].x = std::max(std::min(x2, (float)width - 1), 0.f);
      faces[i].landmark[2].y = std::max(std::min(y2, (float)height - 1), 0.f);
      faces[i].landmark[3].x = std::max(std::min(x3, (float)width - 1), 0.f);
      faces[i].landmark[3].y = std::max(std::min(y3, (float)height - 1), 0.f);
      faces[i].landmark[4].x = std::max(std::min(x4, (float)width - 1), 0.f);
      faces[i].landmark[4].y = std::max(std::min(y4, (float)height - 1), 0.f);
    }
  }

  return 0;
}

template class FaceDetectorSCRFD<types::Face>;
template class FaceDetectorSCRFD<types::FaceExtended>;

}  // namespace models
}  // namespace daisykit
