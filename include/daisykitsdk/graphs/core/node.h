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

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>

namespace daisykit {
namespace graphs {

class Node {
 public:
  Node(const std::string& node_name, NodeType node_type = NodeType::kSyncNode) {
    node_name_ = node_name;
    node_type_ = node_type;
  }
  ~Node() {}

  void Activate() {
    if (is_initialized_) return;
    if (node_type_ == NodeType::kAsyncNode) {
      worker_thread_ = SpawnWorker();
      worker_thread_.detach();
    }
    is_initialized_ = true;
  }

  void Input(const std::string& input_name, PacketPtr packet) {
    if (!is_initialized_) {
      std::cerr << "The node has not been initialized! Exiting..." << std::endl;
      exit(1);
    }
    for (auto const& conn : in_connections_) {
      if (conn->GetNextInputName() == input_name) {
        conn->Transmit(packet);
      }
    }
  }

  virtual void Process(PacketPtr in_packet, PacketPtr& out_packet) = 0;

  void Output(PacketPtr& packet) {
    // Reimplement
  }

  NodeType GetNodeType() { return node_type_; }

  void AddInputConnection(std::shared_ptr<Connection<Node>> connection) {
    in_connections_.push_back(connection);
  }

  void AddOutputConnection(std::shared_ptr<Connection<Node>> connection) {
    out_connections_.push_back(connection);
  }

  bool IsAllDataAvailable() {
    if (in_connections_.empty()) {
      std::cerr << node_name_ << ": No input connection." << std::endl;
      return false;
    }
    for (auto const& conn : in_connections_) {
      if (conn->RequireDataOnTick() && conn->QueueSize() == 0) {
        return false;
      }
    }
    return true;
  }

  void WaitForData() {
    if (in_connections_.empty()) {
      std::cerr << node_name_ << ": No input connection." << std::endl;
      return;
    }
    for (auto const& conn : in_connections_) {
      if (conn->RequireDataOnTick() && conn->QueueSize() == 0) {
        conn->WaitForData();
      }
    }
  }

  void WorkerThread() {
    while (1) {
      if (!is_initialized_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      };
      Tick();
    }
  };

  virtual void Tick() = 0;

  std::thread SpawnWorker() { return std::thread(&Node::WorkerThread, this); }

  int PrepareInputs(std::map<std::string, PacketPtr>& input_map) {
    for (auto const& conn : in_connections_) {
      PacketPtr packet;
      if (conn->RequireDataOnTick()) {
        packet = conn->WaitPopPacket();
      } else if (conn->TryPopPacket(packet) != 0) {
        continue;
      }
      input_map.insert(std::make_pair(conn->GetNextInputName(), packet));
    }
    return 0;
  }

  void Publish(const std::map<std::string, PacketPtr>& outputs) {
    for (auto const& conn : out_connections_) {
      std::string input_name = conn->GetPrevOutputName();
      for (auto const& output : outputs) {
        if (output.first == input_name) {
          conn->Transmit(output.second);
        }
      }
    }
  }

  std::string GetNodeName() { return node_name_; }

 private:
  bool is_initialized_ = false;
  std::thread worker_thread_;
  std::string node_name_;
  NodeType node_type_;
  std::vector<std::shared_ptr<Connection<Node>>> in_connections_;
  std::vector<std::shared_ptr<Connection<Node>>> out_connections_;
  std::thread processing_worker_;
};

}  // namespace graphs
}  // namespace daisykit

#endif