// Copyright 2021 The DaisyKit Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "daisykitsdk/processors/signal_processors/z_score_filter.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace daisykit {
namespace processors {

std::vector<int> ZScoreFilter::Filter(std::vector<double> input) {
  // lag for the smoothing functions
  int lag = 10;
  // standard deviations for signal
  float threshold = 2.5;
  // between 0 and 1, where 1 is normal influence, 0.5 is half
  float influence = 0.8;

  if (input.size() <= lag + 2) {
    std::vector<int> emptyVec;
    return emptyVec;
  }

  // Initialise variables
  std::vector<int> signals(input.size(), 0.0);
  std::vector<float> filteredY(input.size(), 0.0);
  std::vector<float> avgFilter(input.size(), 0.0);
  std::vector<float> stdFilter(input.size(), 0.0);
  std::vector<float> subVecStart(input.begin(), input.begin() + lag);

  double sum =
      std::accumulate(std::begin(subVecStart), std::end(subVecStart), 0.0);
  avgFilter[lag] = sum / subVecStart.size();

  double accum = 0.0;
  std::for_each(std::begin(subVecStart), std::end(subVecStart),
                [&](const double d) {
                  accum += (d - avgFilter[lag]) * (d - avgFilter[lag]);
                });
  stdFilter[lag] = sqrt(accum / (subVecStart.size() - 1));

  for (size_t i = lag + 1; i < input.size(); i++) {
    if (std::abs(input[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]) {
      if (input[i] > avgFilter[i - 1]) {
        signals[i] = 1;  //# Positive signal
      } else {
        signals[i] = -1;  //# Negative signal
      }
      // Make influence lower
      filteredY[i] = influence * input[i] + (1 - influence) * filteredY[i - 1];
    } else {
      signals[i] = 0;  //# No signal
      filteredY[i] = input[i];
    }
    // Adjust the filters
    std::vector<float> subVec(filteredY.begin() + i - lag,
                              filteredY.begin() + i);

    double sum = std::accumulate(std::begin(subVec), std::end(subVec), 0.0);
    avgFilter[i] = sum / subVec.size();

    double accum = 0.0;
    std::for_each(std::begin(subVec), std::end(subVec), [&](const double d) {
      accum += (d - avgFilter[i]) * (d - avgFilter[i]);
    });
    stdFilter[i] = sqrt(accum / (subVec.size() - 1));
  }

  return signals;
}

}  // namespace processors
}  // namespace daisykit