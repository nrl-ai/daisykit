#ifndef DAISYKIT_UTILS_TIMER_H_
#define DAISYKIT_UTILS_TIMER_H_

#include <chrono>
#include <ctime>

namespace daisykit {
namespace utils {

typedef std::time_t TimePoint;

class Timer {
 public:
  static TimePoint GetCurrentTime() {
    std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  }
};

}  // namespace utils
}  // namespace daisykit

#endif