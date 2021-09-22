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

#include "daisykitsdk/common/utils/timer.h"

#include <chrono>
#include <ctime>

namespace daisykit {
namespace utils {

// Get current time point.
TimePoint Timer::GetCurrentTime() {
  return std::chrono::high_resolution_clock::now();
}

// Get time eslapsed from `start` to `end` time points.
double Timer::CalcTimeElapsedMs(TimePoint start, TimePoint end) {
  double elapsed_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return elapsed_time_ms;
}

// Get time eslapsed from `start` to now.
double Timer::CalcTimeElapsedMs(TimePoint start) {
  auto now = GetCurrentTime();
  return CalcTimeElapsedMs(start, now);
}

}  // namespace utils
}  // namespace daisykit
