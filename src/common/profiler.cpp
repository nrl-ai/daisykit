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

#include "daisykitsdk/common/profiler.h"
#include "daisykitsdk/common/utils/timer.h"

namespace daisykit {

Profiler::Profiler(double fps_count_duration) {
  first_tp_ = utils::Timer::GetCurrentTime();
}

double Profiler::Tick() {
  // Calculate elapsed time from the last time point that FPS was measured.
  const double uptime_sec = utils::Timer::CalcTimeElapsedMs(first_tp_) / 1000.0;
  ++frame_count_;

  // Recalculate FPS after a time duration
  if (uptime_sec > fps_count_duration_) {
    if (uptime_sec == 0) return 0;
    current_fps_ = frame_count_ / uptime_sec;
    first_tp_ = utils::Timer::GetCurrentTime();
    frame_count_ = 0;
  }
  return current_fps_;
}

double Profiler::CurrentFPS() { return current_fps_; }

}  // namespace daisykit
