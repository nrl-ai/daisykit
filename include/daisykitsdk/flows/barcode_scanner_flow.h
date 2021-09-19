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

#ifndef DAISYKIT_FLOWS_BARCODE_SCANNER_FLOW_H_
#define DAISYKIT_FLOWS_BARCODE_SCANNER_FLOW_H_

#include "ReadBarcode.h"

#include <opencv2/opencv.hpp>
#include <string>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

namespace daisykit {
namespace flows {
class BarcodeScannerFlow {
 public:
  BarcodeScannerFlow(const std::string& config_str);
#ifdef __ANDROID__
  BarcodeScannerFlow(AAssetManager* mgr, const std::string& config_str);
#endif
  std::string Process(cv::Mat& rgb, bool draw = true);

 private:
  void DrawRect(cv::Mat& rgb, const ZXing::Position& pos);
  ZXing::DecodeHints hints_;
};

}  // namespace flows
}  // namespace daisykit

#endif