// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Greedy decoding of Transducer.

#ifndef _MNN_S2TRT_DECODING_DECODING_H_
#define _MNN_S2TRT_DECODING_DECODING_H_

#include <string>

#include "mnn-s2trt/common/common.h"

namespace s2trt {
namespace decoding {

class DecodingMethod {
 public:
  virtual ~DecodingMethod() {}
  virtual std::string Decode(mnn::Tensor* enc_out) = 0;
};

}  // namespace decoding
}  // namespace s2trt

#endif