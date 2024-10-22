// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.08.29
// This file is copied/modified from kaldi/src/feat/feature-functions.cc

#include "mnn-s2t/frontend/feature-function.h"

#include <cstdint>
#include <vector>

namespace s2t {
namespace frontend {

void ComputePowerSpectrum(std::vector<float> *complex_fft) {
  int32_t dim = complex_fft->size();

  // now we have in complex_fft, first half of complex spectrum
  // it's stored as [real0, realN/2, real1, im1, real2, im2, ...]

  float *p = complex_fft->data();
  int32_t half_dim = dim / 2;
  float first_energy = p[0] * p[0];
  float last_energy = p[1] * p[1];  // handle this special case

  for (int32_t i = 1; i < half_dim; ++i) {
    float real = p[i * 2];
    float im = p[i * 2 + 1];
    p[i] = real * real + im * im;
  }
  p[0] = first_energy;
  p[half_dim] = last_energy;  // Will actually never be used, and anyway
  // if the signal has been bandlimited sensibly this should be zero.
}

}  // namespace frontend
}  // namespace s2t