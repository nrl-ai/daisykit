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

#include "daisykitsdk/common/utils/timer.h"

#include <chrono>
#include <iostream>

namespace daisykit {

/// Profiler to measure system performance.
/// Currently support FPS only
class Profiler {
 public:
  /// Constructor.
  /// FPS will be measure in fps_count_duration (seconds) continuously
  Profiler(double fps_count_duration = 10.0);

  /// Run this method once when a processing task is done.
  /// Return the current FPS.
  double Tick();

  /// Get current FPS.
  double CurrentFPS();

 private:
  utils::TimePoint first_tp_;      /// Time point on the last FPS measurement
  unsigned long frame_count_ = 0;  /// FPS count from the last FPS measurement
  double current_fps_ = 0;         /// Current FPS
  double fps_count_duration_ = 10.0;  /// Duration to measure FPS. Should be
                                      /// high enough to get a stable FPS
};

}  // namespace daisykit

#endif