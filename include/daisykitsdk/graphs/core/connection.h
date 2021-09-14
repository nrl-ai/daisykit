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

#ifndef DAISYKIT_GRAPHS_CORE_CONNECTION_H_
#define DAISYKIT_GRAPHS_CORE_CONNECTION_H_

#include "daisykitsdk/graphs/core/node_type.h"
#include "daisykitsdk/graphs/core/packet.h"
#include "daisykitsdk/graphs/core/queue.h"
#include "daisykitsdk/graphs/core/transmission_profile.h"

#include <atomic>
#include <memory>
#include <queue>

namespace daisykit {
namespace graphs {

template <typename T>
class Connection {
 public:
  Connection(T* prev_node, const std::string& prev_output_name, T* next_node,
             const std::string& next_input_name,
             TransmissionProfile transmit_profile = TransmissionProfile(),
             bool require_data_on_tick = true) {
    prev_node_ = prev_node;
    prev_output_name_ = prev_output_name;
    next_node_ = next_node;
    next_input_name_ = next_input_name;
    transmit_profile_ = transmit_profile;
    require_data_on_tick_ = require_data_on_tick;
  }

  void Transmit(PacketPtr packet) {
    if (next_node_->GetNodeType() == NodeType::kSyncNode) {
      // If the output node is a synchronous node,
      // push packet and run Tick() of the next node
      // for processing
      PushPacket(packet);
      next_node_->Tick();
    } else if (next_node_->GetNodeType() == NodeType::kAsyncNode) {
      // If the output node is an asynchronous node,
      // we only have to push
      PushPacket(packet);
    } else {
      std::cerr << "Not supported node type!" << std::endl;
      exit(1);
    }
  }

  int QueueSize() { return queue_.Size(); }

  void WaitForData() { queue_.WaitForData(); }

  void PushPacket(PacketPtr& packet) {
    // If queue is full, do a special processing to reduce
    // the number of element before further processing
    if (queue_.Size() >= transmit_profile_.GetMaxQueueSize()) {
      // If allow drop packets then
      // pop the front element and push
      if (transmit_profile_.AllowDrop()) {
        queue_.WaitPop();
        queue_.Push(packet);
      } else {
        // If dropping is not allowed then
        // wait the consumer node to process more packets before
        // push a new one in
        while (queue_.Size() >= transmit_profile_.GetMaxQueueSize()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        queue_.Push(packet);
      }
    } else {
      queue_.Push(packet);
    }
  }

  PacketPtr WaitPopPacket() { return queue_.WaitPop(); }
  int TryPopPacket(PacketPtr& packet) { return queue_.Pop(packet); }

  std::string GetPrevOutputName() { return prev_output_name_; }
  std::string GetNextInputName() { return next_input_name_; }

  bool RequireDataOnTick() { return require_data_on_tick_; }

 private:
  T* prev_node_;
  std::string prev_output_name_;
  T* next_node_;
  std::string next_input_name_;
  Queue<PacketPtr> queue_;
  TransmissionProfile transmit_profile_;
  std::atomic<bool> require_data_on_tick_;
};

}  // namespace graphs
}  // namespace daisykit

#endif