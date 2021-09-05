#include <daisykitsdk/flows/barcode_scanner_flow.h>

using namespace cv;
using namespace std;
using namespace ZXing;
using namespace TextUtfEncoding;
using namespace daisykit::flows;
using namespace daisykit::utils::visualizer;

BarcodeScannerFlow::BarcodeScannerFlow(const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);

  // Setting for barcode reader
  hints.setEanAddOnSymbol(EanAddOnSymbol::Read);
  hints.setTryHarder(true);
  hints.setTryRotate(true);
}

#ifdef __ANDROID__
BarcodeScannerFlow::BarcodeScannerFlow(AAssetManager* mgr,
                                       const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  // Setting for barcode reader
  hints.setEanAddOnSymbol(EanAddOnSymbol::Read);
  hints.setTryHarder(true);
  hints.setTryRotate(true);
}
#endif

std::string BarcodeScannerFlow::Process(cv::Mat& rgb, bool draw) {
  ImageView image{rgb.data, rgb.cols, rgb.rows, ImageFormat::RGB};
  auto results = ReadBarcodes(image, hints);

  int ret;
  bool angle_escape = false;
  std::stringstream result_stream;

  // if we did not find anything, insert a dummy to produce some output for each
  // file
  if (results.empty()) results.emplace_back(DecodeStatus::NotFound);

  for (auto&& result : results) {

    if (!result.isValid()) continue;
  
    if (draw) {
      DrawRect(rgb, result.position());
      VizUtils::DrawLabel(
        rgb, ToUtf8(result.text(), angle_escape),
        cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 10,
        cv::Scalar(0, 0, 0), cv::Scalar(0, 255, 0));
    }
    
    ret |= static_cast<int>(result.status());
    result_stream << ToString(result.format());
    if (result.isValid())
      result_stream << " \"" << ToUtf8(result.text(), angle_escape) << "\"";
    else if (result.format() != BarcodeFormat::None)
      result_stream << " " << ToString(result.status());
    result_stream << "\n";
  }

  return result_stream.str();
}

void BarcodeScannerFlow::DrawRect(cv::Mat& rgb, const Position& pos) {
  for (int i = 0; i < 4; ++i) {
    PointI a = pos[i];
    PointI b = pos[(i + 1) % 4];
    cv::line(rgb, cv::Point(a.x, a.y), cv::Point(b.x, b.y),
             cv::Scalar(0, 255, 0), 2);
  }
}
