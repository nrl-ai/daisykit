#ifndef PUSHUP_ANALYZER_
#define PUSHUP_ANALYZER_

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

#include "daisykitsdk/utils/logging/mjpeg_writer.h"
#include "daisykitsdk/utils/signalproc/signal_processor.h"
#include "daisykitsdk/utils/visualizer/viz_graph.h"

class PushupAnalyzer {
 public:
  PushupAnalyzer();
  int count_pushups(const cv::Mat &rgb, bool is_pushing_up);
  int count_with_new_point(double data, bool is_pushing_up);
  double calc_optical_flow(const cv::Mat &img);
  cv::Mat prvs;  // TODO: Init image from camera
  std::vector<ld> input = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<bool> pushing;
  std::shared_ptr<MJPEGWriter> img_server;
  bool is_first_frame = true;
  int _count_interval = 50; // ms
  int current_count = 0;
  long long int _last_count_time = 0;
};

#endif