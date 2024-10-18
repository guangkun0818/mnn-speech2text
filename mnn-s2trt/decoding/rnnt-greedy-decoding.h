// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Greedy decoding of Transducer.

#ifndef _MNN_S2TRT_DECODING_RNNT_GREEDY_DECODING_H_
#define _MNN_S2TRT_DECODING_RNNT_GREEDY_DECODING_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "mnn-s2trt/decoding/decoding.h"
#include "mnn-s2trt/decoding/tokenizer.h"
#include "mnn-s2trt/models/joiner.h"
#include "mnn-s2trt/models/predictor.h"

namespace s2trt {
namespace decoding {

class RnntGreedyDecoding : public DecodingMethod {
 public:
  explicit RnntGreedyDecoding(
      const std::shared_ptr<models::MnnPredictor>& predictor,
      const std::shared_ptr<models::MnnJoiner>& joiner,
      const std::shared_ptr<SubwordTokenzier>& tokenizer,
      size_t max_token_step);

  std::string Decode(mnn::Tensor* enc_out) override;

 private:
  std::shared_ptr<models::MnnPredictor> predictor_;
  std::shared_ptr<models::MnnJoiner> joiner_;
  std::shared_ptr<SubwordTokenzier> tokenizer_;
  size_t max_token_step_;
};

}  // namespace decoding
}  // namespace s2trt

#endif
