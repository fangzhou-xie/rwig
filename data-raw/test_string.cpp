
// test how to convert the list of list of strings in cpp11 string
// and then to c++ string

#include <iostream>
// #include <string>
// #include <vector>
// #include <iterator>
//#include <algorithm>


#include <cpp11.hpp>

// template <typename T> std::string type_name();

[[cpp11::register]]
void teststring(cpp11::strings s_r) {
  std::string s = cpp11::r_string(s_r[0]);
}

[[cpp11::register]]
void findstring(cpp11::strings s_r, cpp11::strings strs_r) {
  cpp11::r_string s = s_r[0];
  auto it = std::find(strs_r.begin(), strs_r.end(), s);
  std::cout << std::distance(strs_r.begin(), it) << std::endl;


  // std::string s = cpp11::r_string(s_r[0]);
  // std::cout << s << std::endl;
  // std::cout << typeid(s_r).name() << std::endl;
  // std::vector<std::string> strs(strs_r.size());
  // for (int i = 0; i < strs.size(); ++i) {
  //   std::string s_i = cpp11::r_string(strs_r[i]);
  //   strs[i] = s_i;
  //   std::cout << strs[i] << ' ' << strs_r[i] << std::endl;
  // }
  // auto it = std::find(strs.begin(), strs.end(), s_r);
  // std::cout << std::distance(strs.begin(), it) << std::endl;
}

