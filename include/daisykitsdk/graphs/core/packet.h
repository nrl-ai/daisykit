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

#ifndef DAISYKIT_GRAPHS_CORE_PACKET_H_
#define DAISYKIT_GRAPHS_CORE_PACKET_H_

#include "daisykitsdk/utils/timer.h"

#include <chrono>
#include <ctime>
#include <memory>
#include <mutex>
#include <atomic>

namespace daisykit {
namespace graphs {

class Packet {
 public:
  Packet() {
    data_available_ = true;
  }
  Packet(std::shared_ptr<void> data, utils::TimePoint timestamp) {
    // We dont use SetData() here because using mutex lock
    // in the constructor causes crashing
    data_ = data;
    timestamp_ = timestamp;
    data_available_ = true;
  }

  void SetData(std::shared_ptr<void> data, utils::TimePoint timestamp) {
    const std::lock_guard<std::mutex> lock(data_mutex_);
    data_ = data;
    timestamp_ = timestamp;
    data_available_ = true;
  }

  bool GetData(std::shared_ptr<void>& data, utils::TimePoint& timestamp) {
    // If no data
    if (!data_available_) {
      return -1;
    }
    const std::lock_guard<std::mutex> lock(data_mutex_);
    data = data_;
    timestamp = timestamp_;
  }

 private:
  std::mutex data_mutex_;
  std::shared_ptr<void> data_;
  utils::TimePoint timestamp_;
  std::atomic<bool> data_available_;
};

typedef std::shared_ptr<Packet> PacketPtr;

}  // namespace graphs
}  // namespace daisykit

#endif