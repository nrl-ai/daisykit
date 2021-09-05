#ifndef DAISYKIT_FLOWS_PUSHUP_COUNTER_FLOW_H_
#define DAISYKIT_FLOWS_PUSHUP_COUNTER_FLOW_H_

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

#include <daisykitsdk/common/types.h>
#include <daisykitsdk/utils/img_proc/img_utils.h>
#include <daisykitsdk/thirdparties/json.hpp>

namespace daisykit {
namespace flows {
class BaseFlow {
 public:
  BarcodeScannerFlow(const std::string& config_str);
#ifdef __ANDROID__
  BarcodeScannerFlow(AAssetManager* mgr, const std::string& config_str);
#endif
  std::string Process(cv::Mat& rgb, bool draw = true);
};

}  // namespace flows
}  // namespace daisykit

#endif