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

#ifndef DAISYKIT_GRAPHS_CORE_NODE_TYPE_H_
#define DAISYKIT_GRAPHS_CORE_NODE_TYPE_H_

namespace daisykit {
namespace graphs {

/// There are too node types in DaisyKit framework:
/// Synchronous nodes (kSyncNode) processing function Tick() is activated by the
/// previous node, which means all processing pipeline runs node by node.
/// Asynchronous node (kAsyncNode) has a processing thread inside to run
/// processing Tick() in a loop. Thus, these node can run paralelly.
enum NodeType { kSyncNode = 0, kAsyncNode = 1 };

}  // namespace graphs
}  // namespace daisykit

#endif