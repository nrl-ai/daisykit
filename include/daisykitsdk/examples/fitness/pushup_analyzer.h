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
#include "daisykitsdk/utils/visualizer/viz_utils.h"
#include "daisykitsdk/utils/signal_proc/signal_smoothing.h"
#include "daisykitsdk/utils/signal_proc/z_score_filter.h"

namespace daisykit {
namespace examples {
  
class PushupAnalyzer {
 public:
  PushupAnalyzer();
  int count_pushups(const cv::Mat &rgb, bool is_pushing_up);
  int count_with_new_point(double data, bool is_pushing_up);
  double calc_optical_flow(const cv::Mat &img);
  cv::Mat _prvs;  // TODO: Init image from camera
  std::vector<daisykit::utils::signal_proc::ld> _input = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<bool> _pushing;
  std::shared_ptr<daisykit::utils::logging::MJPEGServer> _debug_server;
  bool _is_first_frame = true;
  int _count_interval = 50; // ms
  int _current_count = 0;
  long long int _last_count_time = 0;
};

}  // namespace examples
}  // namespace daisykit

#endif