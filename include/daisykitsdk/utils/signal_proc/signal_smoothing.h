#ifndef DAISYKIT_UTILS_SIGNAL_PROC_SIGNAL_SMOOTHING_H_
#define DAISYKIT_UTILS_SIGNAL_PROC_SIGNAL_SMOOTHING_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace daisykit {
namespace utils {
namespace signal_proc {

typedef long double ld;
typedef unsigned int uint;
typedef std::vector<ld>::iterator vec_iter_ld;

class SignalSmoothing {
 public:
  static std::vector<ld> MeanFilter1D(std::vector<ld> input);
};

}  // namespace signal_proc
}  // namespace utils
}  // namespace daisykit

#endif