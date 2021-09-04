#include <daisykitsdk/utils/signal_proc/signal_smoothing.h>

using namespace daisykit::utils::signal_proc;

std::vector<ld> SignalSmoothing::MeanFilter1D(std::vector<ld> input) {
  std::vector<ld> processing_signal;
  double runningTotal = 0.0;
  int windowSize = 8;
  for (int i = 0; i < input.size(); i++) {
    runningTotal += input[i];                                    // add
    if (i >= windowSize) runningTotal -= input[i - windowSize];  // subtract
    if (i >= (windowSize - 1))  // output moving average
      processing_signal.push_back(runningTotal / (double)windowSize);
  }
  return processing_signal;
}