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

#include "daisykitsdk/processors/signal_processors/signal_smoothing.h"

namespace daisykit {
namespace processors {

std::vector<double> SignalSmoothing::MeanFilter1D(std::vector<double> input) {
  std::vector<double> processing_signal;
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

}  // namespace processors
}  // namespace daisykit