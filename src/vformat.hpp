

// requires C++11
// Reference:
// https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

#ifndef WIG_VFORMAT_H
#define WIG_VFORMAT_H

// #include <string>
// #include <cstdarg>
// #include <vector>

// requires at least C++11
// inline const std::string vformat(const char * const zcFormat, ...) {
//
//   // initialize use of the variable argument array
//   va_list vaArgs;
//   va_start(vaArgs, zcFormat);
//
//   // reliably acquire the size
//   // from a copy of the variable argument array
//   // and a functionally reliable call to mock the formatting
//   va_list vaArgsCopy;
//   va_copy(vaArgsCopy, vaArgs);
//   const int iLen = std::vsnprintf(NULL, 0, zcFormat, vaArgsCopy);
//   va_end(vaArgsCopy);
//
//   // return a formatted string without risking memory mismanagement
//   // and without assuming any compiler or platform specific behavior
//   std::vector<char> zc(iLen + 1);
//   std::vsnprintf(zc.data(), zc.size(), zcFormat, vaArgs);
//   va_end(vaArgs);
//   return std::string(zc.data(), iLen);
//
// }

#include <memory>
#include <string>
#include <stdexcept>

template<typename ... Args>
inline const std::string vformat(const std::string& format, Args ... args) {
  // Extra space for '\0'
  int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1;
  if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
  auto size = static_cast<size_t>( size_s );
  std::unique_ptr<char[]> buf( new char[ size ]() );
  std::snprintf( buf.get(), size, format.c_str(), args ... );
  // We don't want the '\0' inside
  return std::string( buf.get(), buf.get() + size - 1 );
}

#endif // WIG_VFORMAT_H
