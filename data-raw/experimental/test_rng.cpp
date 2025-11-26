
#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

using namespace arma;

class local_rng {
public:
  local_rng() {
    GetRNGstate();
  }

  ~local_rng(){
    PutRNGstate();
  }
};

