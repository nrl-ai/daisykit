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

#ifndef DAISYKIT_GRAPHS_NODES_PACKET_DISTRIBUTOR_NODE_H_
#define DAISYKIT_GRAPHS_NODES_PACKET_DISTRIBUTOR_NODE_H_

#include "daisykitsdk/graphs/core/node.h"

#include <chrono>
#include <memory>

namespace daisykit {
namespace graphs {

namespace nodes {

/// This node copy packet from "input" and distribute to "output"
/// Used to copy and distribute packets between nodes
class PacketDistributorNode : public Node {
 public:
  using Node::Node;  // For constructor inheritance

  void Tick() {
    // Wait for data
    WaitForData();

    // Prepare input packets
    std::map<std::string, PacketPtr> inputs;
    PrepareInputs(inputs);

    // Copy input to output
    PacketPtr input = inputs["input"];
    std::map<std::string, PacketPtr> outputs;
    outputs.insert(std::make_pair("output", input));
    Publish(outputs);
  }
};

}  // namespace nodes
}  // namespace graphs
}  // namespace daisykit

#endif