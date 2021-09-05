#ifndef DAISYKIT_FLOWS_BARCODE_SCANNER_FLOW_H_
#define DAISYKIT_FLOWS_BARCODE_SCANNER_FLOW_H_

#include <atomic>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

#include "GTIN.h"
#include "ReadBarcode.h"
#include "TextUtfEncoding.h"

#include <daisykitsdk/thirdparties/json.hpp>
#include <daisykitsdk/utils/visualizer/viz_utils.h>

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
  ZXing::DecodeHints hints;
};

}  // namespace flows
}  // namespace daisykit

#endif