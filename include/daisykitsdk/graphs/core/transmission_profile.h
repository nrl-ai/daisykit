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

#ifndef DAISYKIT_GRAPHS_CORE_TRANSMISSION_PROFILE_H_
#define DAISYKIT_GRAPHS_CORE_TRANSMISSION_PROFILE_H_

namespace daisykit {
namespace graphs {

class TransmissionProfile {
 public:
  TransmissionProfile(int max_queue_size = 5, bool allow_drop = true) {
    max_queue_size_ = max_queue_size;
    allow_drop_ = allow_drop;
  };
  int GetMaxQueueSize() { return max_queue_size_; }
  bool AllowDrop() { return allow_drop_; }

 private:
  int max_queue_size_;  // Max queue size for the connection
  bool allow_drop_;  // Allow dropping input packets if processing capacity is
                     // full
};

}  // namespace graphs
}  // namespace daisykit

#endif