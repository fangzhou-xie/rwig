// printf-style formatting into a std::string
// https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

#ifndef RWIG_VFORMAT_H
#define RWIG_VFORMAT_H

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

template <typename... Args>
inline const std::string vformat(const std::string &format, Args... args) {
  // extra space for '\0'
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]());
  std::snprintf(buf.get(), size, format.c_str(), args...);
  // we don't want the '\0' inside
  return std::string(buf.get(), buf.get() + size - 1);
}

#endif // RWIG_VFORMAT_H
