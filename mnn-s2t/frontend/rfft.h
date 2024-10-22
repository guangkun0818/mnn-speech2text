// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.08.29
// Real discrete Fourier transform impl.

#ifndef _MNN_S2T_RFFT_H_
#define _MNN_S2T_RFFT_H_

#include <cstdint>
#include <memory>

namespace s2t {
namespace frontend {

// n-point Real discrete Fourier transform
// where n is a power of 2. n >= 2
//
//  R[k] = sum_j=0^n-1 in[j]*cos(2*pi*j*k/n), 0<=k<=n/2
//  I[k] = sum_j=0^n-1 in[j]*sin(2*pi*j*k/n), 0<k<n/2
class Rfft {
 public:
  // @param n Number of fft bins. it should be a power of 2.
  explicit Rfft(int32_t n);
  ~Rfft();

  /** @param in_out A 1-D array of size n.
   *             On return:
   *               in_out[0] = R[0]
   *               in_out[1] = R[n/2]
   *               for 1 < k < n/2,
   *                 in_out[2*k] = R[k]
   *                 in_out[2*k+1] = I[k]
   *
   */
  void Compute(float *in_out);
  void Compute(double *in_out);

 private:
  class RfftImpl;

  std::unique_ptr<RfftImpl> impl_;
};

}  // namespace frontend
}  // namespace s2t

#endif