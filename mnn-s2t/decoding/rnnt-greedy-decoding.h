// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Greedy decoding of Transducer.

#ifndef _MNN_S2T_DECODING_RNNT_GREEDY_DECODING_H_
#define _MNN_S2T_DECODING_RNNT_GREEDY_DECODING_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "mnn-s2t/decoding/decoding.h"
#include "mnn-s2t/decoding/tokenizer.h"
#include "mnn-s2t/models/joiner.h"
#include "mnn-s2t/models/predictor.h"

namespace s2t {
namespace decoding {

class RnntGreedyDecoding : public DecodingMethod {
 public:
  explicit RnntGreedyDecoding(
      const std::shared_ptr<models::MnnPredictor>& predictor,
      const std::shared_ptr<models::MnnJoiner>& joiner,
      const std::shared_ptr<SubwordTokenizer>& tokenizer,
      size_t max_token_step);

  void Init() override;

  inline bool IsBlank(int token) const;

  int Argmax(const std::vector<std::vector<float>>& logits) const;

  std::string Decode(mnn::Tensor* enc_out) override;

 private:
  std::shared_ptr<models::MnnPredictor> predictor_;
  std::shared_ptr<models::MnnJoiner> joiner_;
  std::shared_ptr<SubwordTokenizer> tokenizer_;
  size_t max_token_step_;
};

}  // namespace decoding
}  // namespace s2t

#endif
