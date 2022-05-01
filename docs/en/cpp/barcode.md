# C++: Barcode Detection

Barcodes can be used in a wide range of robotics and software
applications. That’s why we integrated a barcode reader into Daisykit.
The core algorithms of the barcode reader are from [*the Zxing-CPP
project*](https://github.com/nu-book/zxing-cpp). This barcode processor
can read QR codes and bar codes in different formats.

![](/images/python/image10.gif)

Source code: `src/examples/demo_barcode_scanner.cpp`.

```cpp
#include "daisykit/flows/barcode_scanner_flow.h"
#include "third_party/json.hpp"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <streambuf>
#include <string>

using namespace cv;
using namespace std;
using json = nlohmann::json;
using namespace daisykit::flows;

int main(int, char**) {
  std::ifstream t("configs/barcode_scanner_config.json");
  std::string config_str((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());

  BarcodeScannerFlow flow(config_str);

  Mat frame;
  VideoCapture cap(0);

  while (1) {
    cap >> frame;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    std::string result = flow.Process(rgb, true);
    if (!result.empty()) {
      std::cout << "New Scan Finished" << std::endl;
      std::cout << result << std::endl;
    }

    cv::Mat draw;
    cv::cvtColor(rgb, draw, cv::COLOR_RGB2BGR);
    imshow("Image", draw);
    waitKey(1);
  }

  return 0;
}
```

Update the configurations by modifying config files in `assets/configs`.
