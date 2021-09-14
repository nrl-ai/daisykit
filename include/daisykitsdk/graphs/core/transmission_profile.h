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