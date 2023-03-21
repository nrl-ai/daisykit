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

#include "daisykit/models/liveness_detection.h"
#include "daisykit/common/types/face_extended.h"
#include "daisykit/processors/image_processors/img_utils.h"
#include "daisykit/models/face_recognition/face_detector_scrfd.h"

#include <algorithm>
#include<chrono>
#include <iostream>
#include <string>
#include <vector>


daisykit::models::FaceDetectorSCRFD<daisykit::types::FaceExtended>* face_detector =
    new daisykit::models::FaceDetectorSCRFD<daisykit::types::FaceExtended>(
        "models/face_detection_scrfd/scrfd_2.5g_1.param",
        "models/face_detection_scrfd/scrfd_2.5g_1.bin", 640, 0.7, 0.5, true);

namespace daisykit {
namespace models {
    LivenessDetector::LivenessDetector(const char*param_buffer, const unsigned char*weight_buffer, 
                                        int input_width, int input_height, bool use_gpu) 
        : NCNNModel(param_buffer, weight_buffer, use_gpu),
        ImageModel(input_width, input_height) {}

    LivenessDetector::LivenessDetector(const std::string& param_file, const std::string& weight_file,
                            int input_width, int input_height, bool use_gpu)
        : NCNNModel(param_file, weight_file, use_gpu),
        ImageModel(input_width, input_height) {}

    #if __ANDROID__
    LivenessDetector::LivenessDetector(
                            AAssetManager* mgr,
                            const std::string& param_file, const std::string& weight_file)
        : NCNNModel(param_buffer, weight_buffer, use_gpu),
        ImageModel(input_width, input_height) {}
    #endif

    void LivenessDetector::Preprocess(const cv::Mat&image, ncnn::Mat &net_input) {
        std::vector<int> face_box; 
        std::vector<daisykit::types::FaceExtended> face;
        face_detector->Predict(image, face, face_box);
        cv::Mat roi;
        cv::Rect rect = cv::Rect(face_box[0], face_box[1], face_box[0]+face_box[2], face_box[1]+face_box[3]);
        cv::resize(image(rect), roi, cv::Size(80, 80));

        ncnn::Mat in = ncnn::Mat::from_pixels(roi.data, ncnn::Mat::PIXEL_BGR, roi.cols, roi.rows);
    }

    int LivenessDetector::Predict(const cv::Mat&image, std::vector<types::FaceExtended> &faces) {
        float liveness_score = 0.f;
        // Preprocess
        ncnn::Mat in;
        Preprocess(image, in);

        // Model Inference
        ncnn::Mat out;
        int result = Infer(in, out, "data", "softmax");
        if (result != 0) return 0;

        // Post process
        liveness_score += out.row(0)[1];
        faces[0].liveness_score = liveness_score;

        return 0;
    }
}
}
