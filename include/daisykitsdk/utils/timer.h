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

#ifndef DAISYKIT_UTILS_TIMER_H_
#define DAISYKIT_UTILS_TIMER_H_

#include <chrono>
#include <ctime>

namespace daisykit {
namespace utils {

typedef std::time_t TimePoint;

class Timer {
 public:
  static TimePoint GetCurrentTime() {
    std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  }
};

}  // namespace utils
}  // namespace daisykit

#endif