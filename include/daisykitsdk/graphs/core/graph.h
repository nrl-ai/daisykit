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

#ifndef DAISYKIT_GRAPHS_CORE_GRAPH_H_
#define DAISYKIT_GRAPHS_CORE_GRAPH_H_

#include "daisykitsdk/graphs/core/node.h"

#include <map>
#include <string>

namespace daisykit {
namespace graphs {

/// Graph class is used to manage nodes, connections and execution tasks in
/// DaisyKit framework
class Graph {
 public:
  /// Connect 2 nodes by a connection.
  /// This utility function create a connection between 2 node in a graph and
  /// add that connection to the input/output connection list of each node.
  //
  /// When `require_data_on_tick` is set to `true`, `next_node` need to wait for
  /// the packet from `prev_node` for data at the beginning of each processing
  /// round.
  static void Connect(
      Node* prev_node, const std::string& output_name, Node* next_node,
      const std::string& input_name,
      TransmissionProfile transmit_profile = TransmissionProfile(),
      bool require_data_on_tick = true);
};

}  // namespace graphs
}  // namespace daisykit

#endif