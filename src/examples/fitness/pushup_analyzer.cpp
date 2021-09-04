#include <daisykitsdk/examples/fitness/pushup_analyzer.h>

using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace daisykit::examples;
using namespace daisykit::utils::logging;
using namespace daisykit::utils::signal_proc;
using namespace daisykit::utils::visualizer;

PushupAnalyzer::PushupAnalyzer() {
  debug_img_server_ = std::make_shared<MJPEGServer>(8080);
}

int PushupAnalyzer::CountWithNewPoint(double data, bool is_pushing_up) {
  input_.push_back(data);
  pushing_.push_back(is_pushing_up);
  vector<ld> signal = SignalSmoothing::MeanFilter1D(input_);

  int range[2] = {0, 300};

  vector<int> filtered_signals = ZScoreFilter::Filter(signal);

  cv::Mat line_graph = VizUtils::LineGraph(signal, range, filtered_signals, 500);

  std::vector<int> countPostions;
  for (int i = 0; i < (int)filtered_signals.size() - 1; ++i) {
    if (filtered_signals[i] == 1 && filtered_signals[i + 1] == 0 &&
        i < pushing_.size() && pushing_[i]) {
      countPostions.push_back((int)i);
    }
  }

  if (is_first_frame_) {
    debug_img_server_->Start();
    is_first_frame_ = false;
  } else {
    debug_img_server_->WriteFrame(line_graph);
  }

  return (int)countPostions.size();
}

int PushupAnalyzer::CountPushups(const cv::Mat &rgb,
                                  bool is_pushing_up) {

  long long int current_time =
      duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (current_time - last_count_time_ < count_interval_) {
    // Skip counting
    return current_count_;
  } else {
    last_count_time_ = current_time;
  }

  double x = CalcOpticalFlow(rgb);
  
  int count = CountWithNewPoint(x, is_pushing_up);
  current_count_ = count;

  return current_count_;
}

double PushupAnalyzer::CalcOpticalFlow(const cv::Mat &frame2) {
  Mat next;
  cv::resize(frame2, next, cv::Size(224, 224));
  cvtColor(next, next, COLOR_BGR2GRAY);
  if (prvs_.empty()) {
    prvs_ = next;
  }
  Mat flow(prvs_.size(), CV_32FC2);
  calcOpticalFlowFarneback(prvs_, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

  Mat flow_parts[2];
  split(flow, flow_parts);
  Mat magnitude, angle;
  cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

  angle = angle * magnitude;

  double avg_angle = cv::sum(angle)[0];
  avg_angle /= angle.rows * angle.cols;
  prvs_ = next;

  return avg_angle;
}
