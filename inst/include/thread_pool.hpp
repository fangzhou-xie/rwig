// A small persistent thread pool.
//
// ThreadPool(n) keeps n-1 worker threads alive; the calling thread acts as
// worker 0. parallel_for(n_items, fn) splits [0, n_items) into n contiguous
// chunks (the last one takes the remainder, like the original std::thread
// code) and runs fn(start, end) on each. With n <= 1 everything runs inline
// on the caller, so n_threads = 0 keeps exact serial semantics.
//
// Workers must never touch the R API. Kernels passed here are pure C++.

#ifndef RWIG_THREAD_POOL_H
#define RWIG_THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
public:
  using Task = std::function<void(int, int)>;

  explicit ThreadPool(int n_threads) : _n(n_threads < 1 ? 1 : n_threads) {
    for (int t = 1; t < _n; ++t) {
      _workers.emplace_back([this, t]() { this->_worker(t); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(_m);
      _stop = true;
      ++_generation;
    }
    _cv_task.notify_all();
    for (auto &w : _workers) w.join();
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  int size() const { return _n; }

  void parallel_for(int n_items, const Task &fn) {
    if (_n == 1 || n_items < _n) {
      // serial fallback (also avoids empty chunks for tiny problems)
      fn(0, n_items);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(_m);
      _task = &fn;
      _n_items = n_items;
      _remaining = _n - 1;
      ++_generation;
    }
    _cv_task.notify_all();
    // caller runs chunk 0
    _run_chunk(0, n_items, fn);
    std::unique_lock<std::mutex> lk(_m);
    _cv_done.wait(lk, [this]() { return _remaining == 0; });
    _task = nullptr;
  }

private:
  int _n;
  std::vector<std::thread> _workers;
  std::mutex _m;
  std::condition_variable _cv_task, _cv_done;
  const Task *_task = nullptr;
  int _n_items = 0;
  int _remaining = 0;
  unsigned long _generation = 0;
  bool _stop = false;

  void _run_chunk(int t, int n_items, const Task &fn) const {
    const int chunk = n_items / _n;
    const int start = t * chunk;
    const int end = (t == _n - 1) ? n_items : (t + 1) * chunk;
    fn(start, end);
  }

  void _worker(int t) {
    unsigned long seen = 0;
    for (;;) {
      const Task *task = nullptr;
      int n_items = 0;
      {
        std::unique_lock<std::mutex> lk(_m);
        _cv_task.wait(lk, [&]() { return _generation != seen; });
        seen = _generation;
        if (_stop) return;
        task = _task;
        n_items = _n_items;
      }
      if (task) _run_chunk(t, n_items, *task);
      {
        std::lock_guard<std::mutex> lk(_m);
        if (--_remaining == 0) _cv_done.notify_one();
      }
    }
  }
};

#endif // RWIG_THREAD_POOL_H
