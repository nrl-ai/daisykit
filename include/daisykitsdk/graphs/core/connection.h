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

/// Add this line for cross including
/// between node.h and connection.h
class Node;

/// Connection provides immediate transmission pipe between nodes.
/// It connects 2 nodes, handles packet transmission tasks, for example packet
/// queue, pushing, popping packets.
class Connection {
 public:
  /// Initialize a connection between 2 nodes. Receives the node pointers and
  /// input/output names. `prev_output_name` is the output of `prev_node`, which
  /// is connected with input `next_input_name` of `next_node` by this
  /// connection.
  /// `transmit_profile` keeps the policies for packet transmission between 2
  /// node, such as queue size or dropping option.
  /// When `require_data_on_tick` is set to `true`, `next_node` need to wait for
  /// the packet from `prev_node` for data at the beginning of each processing
  /// round.
  Connection(Node* prev_node, const std::string& prev_output_name,
             Node* next_node, const std::string& next_input_name,
             TransmissionProfile transmit_profile = TransmissionProfile(),
             bool require_data_on_tick = true);

  /// Transmit a packet from previous node to the next node.
  void Transmit(PacketPtr packet);

  /// Get transmission queue size.
  int QueueSize();

  /// Blocking function to wait until all data is available for the input node
  /// with `require_data_on_tick` equal to `true`.
  void WaitForData();

  /// Wait until data is available and pop a packet from the connection
  PacketPtr WaitPopPacket();

  /// Try to pop a packet.
  /// Return a packet if successfully, otherwise return a nullptr;
  PacketPtr TryPopPacket();

  /// Getters for node names
  std::string GetPrevOutputName();
  std::string GetNextInputName();

  /// Return true if the next node require data on tick,
  /// otherwise return false
  bool RequireDataOnTick();

 private:
  /// Push packet to the transmission queue.
  void PushPacket(PacketPtr& packet);

  /// Previous node
  Node* prev_node_;
  /// The output name of the previous node
  std::string prev_output_name_;
  /// Next node
  Node* next_node_;
  /// The input name of the next node
  std::string next_input_name_;
  /// Transmission queue
  Queue<PacketPtr> queue_;
  TransmissionProfile transmit_profile_;
  std::atomic<bool> require_data_on_tick_;
};

}  // namespace graphs
}  // namespace daisykit

#endif