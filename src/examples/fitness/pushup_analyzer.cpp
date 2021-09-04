#include <daisykitsdk/examples/fitness/pushup_analyzer.h>

using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace daisykit::examples;
using namespace daisykit::utils::logging;
using namespace daisykit::utils::signal_proc;
using namespace daisykit::utils::visualizer;

PushupAnalyzer::PushupAnalyzer() {
  _debug_server = std::make_shared<MJPEGServer>(8080);
}

int PushupAnalyzer::count_with_new_point(double data, bool is_pushing_up) {
  _input.push_back(data);
  _pushing.push_back(is_pushing_up);
  vector<ld> signal = SignalSmoothing::mean_filter_1d(_input);

  int range[2] = {0, 300};

  vector<int> filtered_signals = ZScoreFilter::filter(signal);

  cv::Mat line_graph = VizUtils::line_graph(signal, range, filtered_signals, 500);

  std::vector<int> countPostions;
  for (int i = 0; i < (int)filtered_signals.size() - 1; ++i) {
    if (filtered_signals[i] == 1 && filtered_signals[i + 1] == 0 &&
        i < _pushing.size() && _pushing[i]) {
      countPostions.push_back((int)i);
    }
  }

  if (_is_first_frame) {
    _debug_server->start();
    _is_first_frame = false;
  } else {
    _debug_server->write(line_graph);
  }

  return (int)countPostions.size();
}

int PushupAnalyzer::count_pushups(const cv::Mat &rgb,
                                  bool is_pushing_up) {

  long long int current_time =
      duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (current_time - _last_count_time < _count_interval) {
    // Skip counting
    return _current_count;
  } else {
    _last_count_time = current_time;
  }

  double x = calc_optical_flow(rgb);
  
  int count = count_with_new_point(x, is_pushing_up);
  _current_count = count;

  return _current_count;
}

double PushupAnalyzer::calc_optical_flow(const cv::Mat &frame2) {
  Mat next;
  cv::resize(frame2, next, cv::Size(224, 224));
  cvtColor(next, next, COLOR_BGR2GRAY);
  if (_prvs.empty()) {
    _prvs = next;
  }
  Mat flow(_prvs.size(), CV_32FC2);
  calcOpticalFlowFarneback(_prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

  Mat flow_parts[2];
  split(flow, flow_parts);
  Mat magnitude, angle;
  cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

  angle = angle * magnitude;

  double avg_angle = cv::sum(angle)[0];
  avg_angle /= angle.rows * angle.cols;
  _prvs = next;

  return avg_angle;
}
