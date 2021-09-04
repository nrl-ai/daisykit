#ifndef DAISYKIT_EXAMPLES_PUSHUPS_PUSHUP_ANALYZER_H_
#define DAISYKIT_EXAMPLES_PUSHUPS_PUSHUP_ANALYZER_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "daisykitsdk/utils/logging/mjpeg_server.h"
#include "daisykitsdk/utils/signal_proc/signal_smoothing.h"
#include "daisykitsdk/utils/signal_proc/z_score_filter.h"
#include "daisykitsdk/utils/visualizer/viz_utils.h"

namespace daisykit {
namespace examples {

class PushupAnalyzer {
 public:
  PushupAnalyzer();
  int CountPushups(const cv::Mat &rgb, bool is_pushing_up);

 private:
  int CountWithNewPoint(double data, bool is_pushing_up);
  double CalcOpticalFlow(const cv::Mat &img);
  std::shared_ptr<daisykit::utils::logging::MJPEGServer> debug_img_server_;
  cv::Mat prvs_;
  std::vector<daisykit::utils::signal_proc::ld> input_ = {0, 0, 0, 0,
                                                          0, 0, 0, 0};
  std::vector<bool> pushing_;
  bool is_first_frame_ = true;
  int count_interval_ = 50;  // ms
  int current_count_ = 0;
  long long int last_count_time_ = 0;
};

}  // namespace examples
}  // namespace daisykit

#endif