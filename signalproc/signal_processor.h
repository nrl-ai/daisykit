#ifndef SIGNAL_PROCESSOR_
#define SIGNAL_PROCESSOR_

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

typedef long double ld;
typedef unsigned int uint;
typedef std::vector<ld>::iterator vec_iter_ld;

class SignalProcessor {
 public:
  static std::vector<int> z_score_thresholding(std::vector<ld> input);
  static std::vector<ld> smooth_signal(std::vector<ld> input);
};

#endif