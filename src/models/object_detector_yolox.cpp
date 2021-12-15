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
// limitations under the License.

#include "daisykit/models/object_detector_yolox.h"
#include "daisykit/models/yolox_utils.h"
#include "daisykit/processors/image_processors/img_utils.h"

#include <string>
#include <vector>

namespace daisykit {
namespace models {

// YOLOX use the same focus in yolov5
class YoloXFocus : public ncnn::Layer {
 public:
  YoloXFocus() { one_blob_only = true; }

  virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                      const ncnn::Option& opt) const {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = w / 2;
    int outh = h / 2;
    int outc = channels * 4;

    top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
    if (top_blob.empty()) return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outc; p++) {
      const float* ptr =
          bottom_blob.channel(p % channels).row((p / channels) % 2) +
          ((p / channels) / 2);
      float* outptr = top_blob.channel(p);

      for (int i = 0; i < outh; i++) {
        for (int j = 0; j < outw; j++) {
          *outptr = *ptr;

          outptr += 1;
          ptr += 2;
        }

        ptr += w;
      }
    }

    return 0;
  }
};

DEFINE_LAYER_CREATOR(YoloXFocus)

ObjectDetectorYOLOX::ObjectDetectorYOLOX(const char* param_buffer,
                                         const unsigned char* weight_buffer,
                                         float score_threshold,
                                         float iou_threshold, int input_width,
                                         int input_height, bool use_gpu)
    : NCNNModel(), ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  assert(input_width == input_height);
  LoadModel(param_buffer, weight_buffer, use_gpu, [](ncnn::Net& model) {
    model.register_custom_layer("YoloV5Focus", YoloXFocus_layer_creator);
    return 0;
  });
}

ObjectDetectorYOLOX::ObjectDetectorYOLOX(const std::string& param_file,
                                         const std::string& weight_file,
                                         float score_threshold,
                                         float iou_threshold, int input_width,
                                         int input_height, bool use_gpu)
    : NCNNModel(), ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  assert(input_width == input_height);
  LoadModel(param_file, weight_file, use_gpu, [](ncnn::Net& model) {
    model.register_custom_layer("YoloV5Focus", YoloXFocus_layer_creator);
    return 0;
  });
}

#if __ANDROID__
ObjectDetectorYOLOX::ObjectDetectorYOLOX(AAssetManager* mgr,
                                         const std::string& param_file,
                                         const std::string& weight_file,
                                         float score_threshold,
                                         float iou_threshold, int input_width,
                                         int input_height, bool use_gpu)
    : NCNNModel(), ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  assert(input_width == input_height);
  LoadModel(mgr, param_file, weight_file, use_gpu, [](ncnn::Net& model) {
    model.register_custom_layer("YoloV5Focus", YoloXFocus_layer_creator);
    return 0;
  });
}
#endif

void ObjectDetectorYOLOX::SetClassNames(
    const std::vector<std::string>& class_names) {
  class_names_ = class_names;
}

std::vector<std::string>& ObjectDetectorYOLOX::GetClassNames() {
  return class_names_;
}

void ObjectDetectorYOLOX::Preprocess(const cv::Mat& image,
                                     ncnn::Mat& net_input) {
  // Clone the original cv::Mat to ensure continuous address for memory
  cv::Mat rgb = image.clone();

  // Letterbox pad to multiple of 32
  int img_width = rgb.cols;
  int img_height = rgb.rows;

  int w = img_width;
  int h = img_height;

  if (w > h) {
    scale_ = (float)InputWidth() / w;
    w = InputWidth();
    h = h * scale_;
  } else {
    scale_ = (float)InputHeight() / h;
    h = InputHeight();
    w = w * scale_;
  }

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,
                                               img_width, img_height, w, h);

  // Pad to target_size rectangle
  // yolo/utils/datasets.py letterbox
  int wpad = InputWidth() - w;
  int hpad = InputHeight() - h;
  ncnn::copy_make_border(in, net_input, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT,
                         114.f);

  const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};

  const float norm_vals[3] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f),
                              1 / (255.f * 0.225f)};
  net_input.substract_mean_normalize(mean_vals, norm_vals);
}

int ObjectDetectorYOLOX::Predict(
    const cv::Mat& image, std::vector<daisykit::types::Object>& objects) {
  // Preprocess
  ncnn::Mat in;
  Preprocess(image, in);

  std::vector<types::Object> proposals;
  {
    // Inference
    ncnn::Mat out;
    int result = Infer(in, out, "images", "output");
    if (result != 0) {
      return result;
    }

    std::vector<int> strides = {8, 16, 32};  // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in.w, in.h, strides, grid_strides);
    generate_yolox_proposals(grid_strides, out, score_threshold_, proposals);
  }

  // Postprocess

  // sort all proposals by score from highest to lowest
  qsort_descent_inplace(proposals);

  // apply nms with nms_threshold
  std::vector<int> picked;
  nms_sorted_bboxes(proposals, picked, iou_threshold_);

  int img_width = image.cols;
  int img_height = image.rows;
  int count = picked.size();
  objects.resize(count);
  for (int i = 0; i < count; i++) {
    objects[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects[i].x) / scale_;
    float y0 = (objects[i].y) / scale_;
    float x1 = (objects[i].x + objects[i].w) / scale_;
    float y1 = (objects[i].y + objects[i].h) / scale_;

    // clip
    x0 = std::max(std::min(x0, (float)(img_width - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(img_height - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(img_width - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(img_height - 1)), 0.f);

    objects[i].x = x0;
    objects[i].y = y0;
    objects[i].w = x1 - x0;
    objects[i].h = y1 - y0;
  }

  return 0;
}

}  // namespace models
}  // namespace daisykit
