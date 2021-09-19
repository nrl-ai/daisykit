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

#include "daisykitsdk/flows/barcode_scanner_flow.h"
#include "GTIN.h"
#include "ReadBarcode.h"
#include "TextUtfEncoding.h"
#include "daisykitsdk/common/visualizers/base_visualizer.h"
#include "daisykitsdk/thirdparties/json.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

namespace daisykit {
namespace flows {

BarcodeScannerFlow::BarcodeScannerFlow(const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);

  // Setting for barcode reader
  hints_.setEanAddOnSymbol(ZXing::EanAddOnSymbol::Read);
  hints_.setTryHarder(config["try_harder"]);
  hints_.setTryRotate(config["try_rotate"]);
}

#ifdef __ANDROID__
BarcodeScannerFlow::BarcodeScannerFlow(AAssetManager* mgr,
                                       const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  // Setting for barcode reader
  hints_.setEanAddOnSymbol(EanAddOnSymbol::Read);
  hints_.setTryHarder(config["try_harder"]);
  hints_.setTryRotate(config["try_rotate"]);
}
#endif

std::string BarcodeScannerFlow::Process(cv::Mat& rgb, bool draw) {
  ZXing::ImageView image{rgb.data, rgb.cols, rgb.rows, ZXing::ImageFormat::RGB};
  auto results = ZXing::ReadBarcodes(image, hints_);

  int ret;
  bool angle_escape = false;
  std::stringstream result_stream;

  // if we did not find anything, insert a dummy to produce some output for each
  // file
  if (results.empty()) results.emplace_back(ZXing::DecodeStatus::NotFound);

  for (auto&& result : results) {
    if (!result.isValid()) continue;

    if (draw) {
      DrawRect(rgb, result.position());
      visualizers::BaseVisualizer::PutText(
          rgb, ZXing::TextUtfEncoding::ToUtf8(result.text(), angle_escape),
          cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 10,
          cv::Scalar(0, 0, 0), cv::Scalar(0, 255, 0));
    }

    ret |= static_cast<int>(result.status());
    result_stream << ZXing::ToString(result.format());
    if (result.isValid())
      result_stream << " \""
                    << ZXing::TextUtfEncoding::ToUtf8(result.text(),
                                                      angle_escape)
                    << "\"";
    else if (result.format() != ZXing::BarcodeFormat::None)
      result_stream << " " << ZXing::ToString(result.status());
    result_stream << "\n";
  }

  return result_stream.str();
}

void BarcodeScannerFlow::DrawRect(cv::Mat& rgb, const ZXing::Position& pos) {
  for (int i = 0; i < 4; ++i) {
    ZXing::PointI a = pos[i];
    ZXing::PointI b = pos[(i + 1) % 4];
    cv::line(rgb, cv::Point(a.x, a.y), cv::Point(b.x, b.y),
             cv::Scalar(0, 255, 0), 2);
  }
}

}  // namespace flows
}  // namespace daisykit