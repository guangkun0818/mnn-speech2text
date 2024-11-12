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

enum DecodingType {
  kRnntGreedyDecoding = 0x01,
  kRnntBeamDecoding = 0x02,
};

struct DecodingCfg {
  DecodingType decoding_type = DecodingType::kRnntGreedyDecoding;
  size_t max_token_step = 1;  // For Greedy Decoding
  size_t beam_size;           // For Beam Decoding.
  size_t cutoff_top_k = 4;    // For Beam Decoding.
};

class DecodingMethod {
 public:
  virtual ~DecodingMethod() {}
  virtual void Init() = 0;
  virtual void Reset() = 0;
  virtual void Decode(mnn::Tensor* enc_out) = 0;
  virtual std::string GetResults() = 0;
};

}  // namespace decoding
}  // namespace s2t

#endif