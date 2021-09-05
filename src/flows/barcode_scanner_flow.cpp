#include <daisykitsdk/flows/barcode_scanner_flow.h>

using namespace cv;
using namespace std;
using namespace zxing;
using namespace zxing::qrcode;
using namespace daisykit::flows;
using namespace daisykit::utils::visualizer;

BarcodeScannerFlow::BarcodeScannerFlow(const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
}

#ifdef __ANDROID__
BarcodeScannerFlow::BarcodeScannerFlow(AAssetManager* mgr,
                                       const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
}
#endif

std::string BarcodeScannerFlow::Process(cv::Mat& rgb, bool draw) {
  // Convert to grayscale
  cv::Mat grey;
  cvtColor(rgb, grey, cv::COLOR_BGR2GRAY);

  try {
    // Create luminance  source
    Ref<LuminanceSource> source = MatSource::create(grey);

    reader.reset(new MultiFormatReader);

    Ref<Binarizer> binarizer(new GlobalHistogramBinarizer(source));
    Ref<BinaryBitmap> bitmap(new BinaryBitmap(binarizer));
    Ref<Result> result(
        reader->decode(bitmap, DecodeHints(DecodeHints::TRYHARDER_HINT)));

    // Get result point count
    int resultPointCount = result->getResultPoints()->size();

    if (draw) {
      for (int j = 0; j < resultPointCount; j++) {
        // Draw circle
        circle(rgb, ToCvPoint(result->getResultPoints()[j]), 10,
               Scalar(110, 220, 0), 2);
      }

      // Draw boundary on image
      if (resultPointCount > 1) {
        for (int j = 0; j < resultPointCount; j++) {
          // Get start result point
          Ref<ResultPoint> previousResultPoint =
              (j > 0) ? result->getResultPoints()[j - 1]
                      : result->getResultPoints()[resultPointCount - 1];

          // Draw line
          line(rgb, ToCvPoint(previousResultPoint),
               ToCvPoint(result->getResultPoints()[j]), Scalar(110, 220, 0), 2,
               8);

          // Update previous point
          previousResultPoint = result->getResultPoints()[j];
        }
      }

      if (resultPointCount > 0) {
        // Draw text
        VizUtils::DrawLabel(
          rgb, result->getText()->getText(),
          ToCvPoint(result->getResultPoints()[0]), cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, 10,
          cv::Scalar(0, 0, 0), cv::Scalar(0, 255, 0));
      }
    }

    if (resultPointCount > 0) {
      return result->getText()->getText();
    }

  } catch (const ReaderException& e) {
    cerr << e.what() << " (ignoring)" << endl;
  } catch (const zxing::IllegalArgumentException& e) {
    cerr << e.what() << " (ignoring)" << endl;
  } catch (const zxing::Exception& e) {
    cerr << e.what() << " (ignoring)" << endl;
  } catch (const std::exception& e) {
    cerr << e.what() << " (ignoring)" << endl;
  }

  return "";
}

cv::Point BarcodeScannerFlow::ToCvPoint(Ref<ResultPoint> resultPoint) {
  return Point(resultPoint->getX(), resultPoint->getY());
}
