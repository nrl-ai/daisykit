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

#include "daisykitsdk/graphs/core/node.h"
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

Node::Node(const std::string& node_name, NodeType node_type) {
  node_name_ = node_name;
  node_type_ = node_type;
  is_activated_ = false;
}

void Node::Activate() {
  if (is_activated_) return;
  if (node_type_ == NodeType::kAsyncNode) {
    worker_thread_ = SpawnWorker();
    worker_thread_.detach();
  }
  is_activated_ = true;
}

void Node::Input(const std::string& input_name, PacketPtr packet) {
  if (!is_activated_) {
    std::cerr << "The node has not been initialized! Exiting..." << std::endl;
    exit(1);
  }
  for (auto const& conn : in_connections_) {
    if (conn->GetNextInputName() == input_name) {
      conn->Transmit(packet);
    }
  }
}

void Node::AddInputConnection(std::shared_ptr<Connection> connection) {
  in_connections_.push_back(connection);
}

void Node::AddOutputConnection(std::shared_ptr<Connection> connection) {
  out_connections_.push_back(connection);
}

bool Node::IsAllDataAvailable() {
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

void Node::WaitForData() {
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

std::string Node::GetNodeName() { return node_name_; }
NodeType Node::GetNodeType() { return node_type_; }

void Node::PrepareInputs(std::map<std::string, PacketPtr>& input_map) {
  for (auto const& conn : in_connections_) {
    PacketPtr packet;
    if (conn->RequireDataOnTick()) {
      packet = conn->WaitPopPacket();
    } else {
      packet = conn->TryPopPacket();
    }
    // If packet is pop successfully
    if (packet != nullptr) {
      input_map.insert(std::make_pair(conn->GetNextInputName(), packet));
    }
  }
}

void Node::Publish(const std::map<std::string, PacketPtr>& outputs) {
  for (auto const& conn : out_connections_) {
    std::string input_name = conn->GetPrevOutputName();
    for (auto const& output : outputs) {
      if (output.first == input_name) {
        conn->Transmit(output.second);
      }
    }
  }
}

void Node::WorkerThread() {
  while (1) {
    // If the node is not activated, sleep to wait
    if (!is_activated_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    };
    // Run Tick() as the processing function
    Tick();
  }
};

std::thread Node::SpawnWorker() {
  return std::thread(&Node::WorkerThread, this);
}
}  // namespace graphs
}  // namespace daisykit
