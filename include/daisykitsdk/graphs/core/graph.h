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

class Graph {
 public:
  static void Connect(
      Node* prev_node, const std::string& output_name, Node* next_node,
      const std::string& input_name,
      TransmissionProfile transmit_profile = TransmissionProfile(),
      bool require_data_on_tick = true) {
    std::shared_ptr<Connection<Node>> connection =
        std::make_shared<Connection<Node>>(prev_node, output_name, next_node,
                                           input_name, transmit_profile,
                                           require_data_on_tick);
    if (prev_node) prev_node->AddOutputConnection(connection);
    if (next_node) next_node->AddInputConnection(connection);
  }
};

}  // namespace graphs
}  // namespace daisykit

#endif