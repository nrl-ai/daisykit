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

#ifndef DAISYKIT_GRAPHS_CORE_NODE_H_
#define DAISYKIT_GRAPHS_CORE_NODE_H_

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/graphs/core/connection.h"
#include "daisykitsdk/graphs/core/node_type.h"
#include "daisykitsdk/graphs/core/packet.h"

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>

namespace daisykit {
namespace graphs {

/// A node is a processing unit which handle a task such as inferecing a model,
/// running an image processing operation, or visualizing data.
class Node {
 public:
  /// Node constructor. Passing a name and node type here.
  /// Synchronous nodes (kSyncNode) processing function Tick() is activated by
  /// the previous node, which means all processing pipeline runs node by node.
  /// Asynchronous node (kAsyncNode) has a processing thread inside to run
  /// processing Tick() in a loop. Thus, these node can run paralelly.
  Node(
      const std::string& node_name,  /// Node name
      NodeType node_type =
          NodeType::kAsyncNode  /// Node type / operation mode.
                                /// NodeType::kSyncNode for synchronous node,
                                /// NodeType::kAsyncNode for multithreading node
  );

  /// Activate a node. This function create and activate processing thread for
  /// asynchronous node.
  void Activate();

  /// Feed data to a node. This function can be used to feed data to the input
  /// node of a graph, where there is no connection in.
  void Input(const std::string& input_name, PacketPtr packet);

  /// Add an input connection to the node.
  /// Input connections are used to get input to this node.
  void AddInputConnection(std::shared_ptr<Connection> connection);

  /// Add an output connection from the node.
  /// Output connections are used to push processing results to next nodes.
  void AddOutputConnection(std::shared_ptr<Connection> connection);

  /// Check and verify all required data is available before run processing
  /// pipeline of this node
  bool IsAllDataAvailable();

  /// Wait for all required data.
  void WaitForData();

  /// Virtual method for processing data, needs to be implemented by derived
  /// classes. This method checks and gets all required data, processes data and
  /// outputs to out connections.
  virtual void Tick() = 0;

  /// Getters for node info
  std::string GetNodeName();
  NodeType GetNodeType();

 protected:
  /// Prepare all needed input as a map for processing function.
  void PrepareInputs(std::map<std::string, PacketPtr>& input_map);

  /// Publish outputs to output connections.
  void Publish(const std::map<std::string, PacketPtr>& outputs);

 private:
  /// Worker thread for each node.
  /// This thread runs Tick() function in a loop
  void WorkerThread();

  /// Start processing thread.
  std::thread SpawnWorker();

  std::atomic<bool> is_activated_;
  std::thread worker_thread_;
  std::string node_name_;
  NodeType node_type_;
  std::vector<std::shared_ptr<Connection>> in_connections_;
  std::vector<std::shared_ptr<Connection>> out_connections_;
  std::thread processing_worker_;
};

}  // namespace graphs
}  // namespace daisykit

#endif