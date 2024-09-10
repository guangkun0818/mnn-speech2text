// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.08.29
// This file is copied/modified from kaldi/src/feat/feature-functions.h

#ifndef _MNN_S2TRT_FEATURE_FUNCTIONS_H_
#define _MNN_S2TRT_FEATURE_FUNCTIONS_H_

#include <vector>

namespace s2trt {
namespace frontend {

// ComputePowerSpectrum converts a complex FFT (as produced by the FFT
// functions in csrc/rfft.h), and converts it into
// a power spectrum.  If the complex FFT is a vector of size n (representing
// half of the complex FFT of a real signal of size n, as described there),
// this function computes in the first (n/2) + 1 elements of it, the
// energies of the fft bins from zero to the Nyquist frequency.  Contents of the
// remaining (n/2) - 1 elements are undefined at output.

void ComputePowerSpectrum(std::vector<float> *complex_fft);

}  // namespace frontend
}  // namespace s2trt

#endif  // _MNN_S2TRT_FEATURE_FUNCTIONS_H_
