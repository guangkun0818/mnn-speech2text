// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.08.29
// Real discrete Fourier transform impl.

#include "mnn-s2trt/frontend/rfft.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "glog/logging.h"

namespace s2trt {
namespace frontend {

// see fftsg.cc
void rdft(int n, int isgn, double *a, int *ip, double *w);

class Rfft::RfftImpl {
 public:
  explicit RfftImpl(int32_t n) : n_(n), ip_(2 + std::sqrt(n / 2)), w_(n / 2) {
    CHECK_EQ(n & (n - 1), 0);
  }

  void Compute(float *in_out) {
    std::vector<double> d(in_out, in_out + n_);

    Compute(d.data());

    std::copy(d.begin(), d.end(), in_out);
  }

  void Compute(double *in_out) {
    // 1 means forward fft
    rdft(n_, 1, in_out, ip_.data(), w_.data());
  }

 private:
  int32_t n_;
  std::vector<int32_t> ip_;
  std::vector<double> w_;
};

Rfft::Rfft(int32_t n) : impl_(std::make_unique<RfftImpl>(n)) {}

Rfft::~Rfft() = default;

void Rfft::Compute(float *in_out) { impl_->Compute(in_out); }
void Rfft::Compute(double *in_out) { impl_->Compute(in_out); }

}  // namespace frontend
}  // namespace s2trt
