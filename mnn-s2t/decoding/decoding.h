// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Greedy decoding of Transducer.

#ifndef _MNN_S2T_DECODING_DECODING_H_
#define _MNN_S2T_DECODING_DECODING_H_

#include <string>

#include "mnn-s2t/common/common.h"

namespace s2t {
namespace decoding {

class DecodingMethod {
 public:
  virtual ~DecodingMethod() {}
  virtual void Init() = 0;
  virtual std::string Decode(mnn::Tensor* enc_out) = 0;
};

}  // namespace decoding
}  // namespace s2t

#endif