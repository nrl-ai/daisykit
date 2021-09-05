#ifndef DAISYKIT_FLOWS_BARCODE_SCANNER_FLOW_H_
#define DAISYKIT_FLOWS_BARCODE_SCANNER_FLOW_H_

#include <atomic>
#include <iostream>
#include <string>
#include <vector>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

#include <zxing/Binarizer.h>
#include <zxing/BinaryBitmap.h>
#include <zxing/DecodeHints.h>
#include <zxing/Exception.h>
#include <zxing/MatSource.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/ReaderException.h>
#include <zxing/Result.h>
#include <zxing/common/Counted.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
#include <zxing/common/IllegalArgumentException.h>
#include <zxing/qrcode/QRCodeReader.h>

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
  cv::Point ToCvPoint(zxing::Ref<zxing::ResultPoint> resultPoint);
  zxing::Ref<zxing::Reader> reader;
};

}  // namespace flows
}  // namespace daisykit

#endif