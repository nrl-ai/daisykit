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

#ifndef DAISYKIT_COMMON_PROFILER_H_
#define DAISYKIT_COMMON_PROFILER_H_

#include <chrono>
#include <iostream>

namespace daisykit {

/// Profiler to meas
class Profiler {
 public:
  Profiler() { first_tp_ = std::chrono::high_resolution_clock::now(); }

  double Duration() {
    auto duration = std::chrono::high_resolution_clock::now() - first_tp_;
    double elapsed_time_ms =
        std::chrono::duration<double, std::milli>(duration).count();
    return elapsed_time_ms / 1000.0;
  }

  double Tick() {
    const double uptime_sec = Duration();
    ++frame_count_;
    if (uptime_sec > kFPSCountDuration) {
      if (uptime_sec == 0) return 0;
      current_fps_ = frame_count_ / uptime_sec;
      first_tp_ = std::chrono::high_resolution_clock::now();
      frame_count_ = 0;
    }
    return current_fps_;
  }

  double CurrentFPS() { return current_fps_; }

 private:
  std::chrono::high_resolution_clock::time_point first_tp_;
  unsigned long frame_count_ = 0;
  double current_fps_ = 0;
  double kFPSCountDuration = 10.0;  // secs
};

}  // namespace daisykit

#endif