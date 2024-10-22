// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Greedy decoding of Transducer.

#include "mnn-s2t/decoding/rnnt-greedy-decoding.h"

#include <algorithm>

namespace s2t {
namespace decoding {

RnntGreedyDecoding::RnntGreedyDecoding(
    const std::shared_ptr<models::MnnPredictor>& predictor,
    const std::shared_ptr<models::MnnJoiner>& joiner,
    const std::shared_ptr<SubwordTokenizer>& tokenizer, size_t max_token_step)
    : predictor_(predictor),
      joiner_(joiner),
      tokenizer_(tokenizer),
      max_token_step_(max_token_step) {}

void RnntGreedyDecoding::Init() {
  // Beam size = 1
  predictor_->Init(1);
  joiner_->Init(1);
  predictor_->StreamingStep({0});  // Init predictor with <blank_id>.
}

bool RnntGreedyDecoding::IsBlank(int token) const { return token == 0; }

int RnntGreedyDecoding::Argmax(
    const std::vector<std::vector<float>>& logits) const {
  // Logit shape {1, vocab_size}
  CHECK_EQ(logits.size(), 1);  // Beam size = 1;
  auto max = std::max_element(logits[0].begin(), logits[0].end());  // [2, 4)
  int argmax = std::distance(logits[0].begin(), max);  // absolute index of max
  return argmax;
};

std::string RnntGreedyDecoding::Decode(mnn::Tensor* enc_out) {
  this->Init();
  CHECK_EQ(enc_out->shape()[0], 1);           // Batch Size = 1.
  auto tot_time_steps = enc_out->shape()[1];  // (1, tot_time_steps, enc_dim)
  auto enc_dim = enc_out->shape()[2];

  std::vector<int> enc_frame_shape = {1, /*tot_time_step=*/1,
                                      /*enc_dim=*/enc_dim};
  auto enc_frame =
      mnn::Tensor::create<float>(enc_frame_shape, NULL, MNN::Tensor::CAFFE);

  int curr_time_step = 0;
  int num_token_step = 0;
  std::vector<int> decoded_result;
  while (curr_time_step < tot_time_steps) {
    // Slice encoder_out as frame
    memcpy(enc_frame->host<float>(),
           enc_out->host<float>() + curr_time_step * enc_dim,
           sizeof(float) * enc_dim);

    // Joiner streaming step.
    joiner_->StreamingStep(/*enc_out=*/enc_frame,
                           /*pred_out=*/predictor_->GetPredOut());
    auto logits = joiner_->GetJoinerOut();
    auto pred_token = Argmax(logits);

    if (IsBlank(pred_token) || num_token_step > max_token_step_) {
      // If <blank_id> predicted or num token step reach the predefined limit of
      // max_token_step, move to next time_step (lattice move rightward).
      // Token_step maintained.
      curr_time_step++;
      num_token_step = 0;
      continue;
    } else {
      // If not <blank_id>, move to next token_step (lattice move upward)
      // time_step maintained, num_token_step update.
      num_token_step++;
      predictor_->StreamingStep({pred_token});
      decoded_result.push_back(pred_token);
      continue;
    }
  }
  return tokenizer_->Decode(decoded_result);
}

}  // namespace decoding
}  // namespace s2t