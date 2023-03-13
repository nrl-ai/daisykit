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

#include "daisykit/models/face_anti_spoofing.h"
#include "daisykit/processors/image_processors/img_utils.h"

#include <algorithm>
#include<chrono>
#include <iostream>
#include <string>
#include <vector>

namespace daisykit {
namespace models {
FakeRealClassifiers::FakeRealClassifiers(const char* param_buffer, 
                                        const unsigned char* weight_buffer,
                                        int input_width, int input_height,
                                        bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
    ImageModel(input_width, input_height) {}

FakeRealClassifiers::FakeRealClassifiers(const std::string& param_file, 
                                        const std::string& weight_file,
                                        int input_width, int input_height,
                                        bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
    ImageModel(input_width, input_height) {}

#if __ANDROID__
FakeRealClassifiers::FakeRealClassifiers(AAssetManager* mgr,
                                        const std::string& param_file, 
                                        const std::string& weight_file,
                                        int input_width, int input_height,
                                        bool use_gpu, bool smooth)
    : NCNNModel(param_file, weight_file, use_gpu),
    ImageModel(input_width, input_height) {
    smooth_ = smooth;
}
#endif


void Preprocess(const cv::Mat&image, ncnn::Mat& net_input, int input_size_=192) {
    int w = image.cols;
    int h = image.rows;
    float aspect_ratio = w / (float)h;

    int input_width = static_cast<int>(input_size_ * sqrt(aspect_ratio));
    int input_height = static_cast<int>(input_size_ / sqrt(aspect_ratio));

    net_input = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows,
                                                 input_width, input_height);
}

int FakeRealClassifiers::Detect(const cv::Mat&image, std::vector<types::FaceBox> &boxes) {
    int w = image.cols;
    int h = image.rows;
    // Preprocess 
    ncnn::Mat in;
    Preprocess(image, in, 192);

    const float mean_val_[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_val_, nullptr);
    
    // Model Inference
    ncnn::Mat out;
    int result = Infer(in, out, "data", "detection_out");
    if (result != 0) return result;

    //Postprocess
    for (int i = 0; i < out.h; ++i) {
        const float* values = out.row(i);
        float confidence = values[1];

        if(confidence < threshold_) continue;

        types::FaceBox box;
        box.confidence = confidence;
        box.x1 = values[2] * w;
        box.y1 = values[3] * h;
        box.x2 = values[4] * w;
        box.y2 = values[5] * h;

        // square
        float box_width = box.x2 - box.x1 + 1;
        float box_height = box.y2 - box.y1 + 1;

        float size = (box_width + box_height) * 0.5f;

        if(size < min_face_size_) continue;

        float cx = box.x1 + box_width * 0.5f;
        float cy = box.y1 + box_height * 0.5f;

        box.x1 = cx - size * 0.5f;
        box.y1 = cy - size * 0.5f;
        box.x2 = cx + size * 0.5f - 1;
        box.y2 = cy + size * 0.5f - 1;

        boxes.emplace_back(box);
    }

    std::sort(boxes.begin(), boxes.end());
    return 0;
}

}
}
