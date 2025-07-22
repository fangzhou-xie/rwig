
// header for a timer class for timing the functions
// useful for improving the performance of codes

#ifndef WIG_TIMER_H
#define WIG_TIMER_H

#include <chrono>              // std::chrono::steady_clock


class TicToc {

private:

  std::chrono::duration<double> _sec_last;                    // last time
  std::chrono::duration<double> _secs;                        // elapsed time
  std::chrono::time_point<std::chrono::steady_clock> _t1;     // tic time
  std::chrono::time_point<std::chrono::steady_clock> _t2;     // toc time

  // also record the number of tic-toc pairs
  int _cnt = 0;

public:

  void tic() {
    this->_t1 = std::chrono::steady_clock::now();
  }

  void toc() {
    this->_t2 = std::chrono::steady_clock::now();
    this->_sec_last = this->_t2 - this->_t1;
    this->_secs += this->_sec_last;
    // increment counter
    this->_cnt++;
  }

  auto elapsed() {
    return this->_secs.count();
  }
  auto speed_avg() {
    // average speed over the course of the class
    return this->_secs.count() / this->_cnt;
  }
  auto speed_last() {
    // time difference for the last tic-toc pair
    return this->_sec_last.count();
  }

};


#endif // WIG_TIMER_H
