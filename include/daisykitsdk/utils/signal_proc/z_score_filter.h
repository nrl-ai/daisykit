#ifndef DAISYKIT_UTILS_SIGNALPROC_Z_SCORE_FILTER_H_
#define DAISYKIT_UTILS_SIGNALPROC_Z_SCORE_FILTER_H_

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

class ZScoreFilter {
 public:
  static std::vector<int> Filter(std::vector<ld> input);
};

}  // namespace signal_proc
}  // namespace utils
}  // namespace daisykit

#endif