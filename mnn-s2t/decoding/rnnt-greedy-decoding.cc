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
    const std::shared_ptr<models::RnntModelSession>& model_sess,
    const std::shared_ptr<SubwordTokenizer>& tokenizer, const DecodingCfg& cfg)
    : predictor_(predictor),
      joiner_(joiner),
      model_sess_(model_sess),
      tokenizer_(tokenizer),
      max_token_step_(cfg.max_token_step) {
  decoding_states_ = std::make_shared<RnntGreedyDecodingStates>();
}

void RnntGreedyDecoding::Init() {
  // Beam size = 1
  model_sess_->predictor_session = predictor_->Init(1);
  model_sess_->joiner_session = joiner_->Init(1);
  predictor_->StreamingStep(
      {0}, model_sess_->predictor_session);  // Init predictor with <blank_id>.
}

void RnntGreedyDecoding::Reset() {
  model_sess_->predictor_session = predictor_->Reset(
      model_sess_->predictor_session);  // Release Predictor session.
  model_sess_->joiner_session =
      joiner_->Reset(model_sess_->joiner_session);  // Release Joiner session.
  this->ResetDecodingStates();
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

void RnntGreedyDecoding::Decode(mnn::Tensor* enc_out) {
  CHECK_EQ(enc_out->shape()[0], 1);           // Batch Size = 1.
  auto tot_time_steps = enc_out->shape()[1];  // (1, tot_time_steps, enc_dim)
  auto enc_dim = enc_out->shape()[2];

  std::vector<int> enc_frame_shape = {1, /*tot_time_step=*/1,
                                      /*enc_dim=*/enc_dim};
  auto enc_frame =
      mnn::Tensor::create<float>(enc_frame_shape, NULL, MNN::Tensor::CAFFE);

  int curr_time_step = 0;
  int num_token_step = 0;
  std::vector<int> decoded_tokens;
  while (curr_time_step < tot_time_steps) {
    // Slice encoder_out as frame
    memcpy(enc_frame->host<float>(),
           enc_out->host<float>() + curr_time_step * enc_dim,
           sizeof(float) * enc_dim);

    // Joiner streaming step.
    joiner_->StreamingStep(
        /*enc_out=*/enc_frame,
        /*pred_out=*/predictor_->GetPredOut(model_sess_->predictor_session),
        /*joiner_session=*/model_sess_->joiner_session);
    auto logits = joiner_->GetJoinerOut(model_sess_->joiner_session);
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
      predictor_->StreamingStep(
          /*pred_token=*/{pred_token},
          /*predictor_session=*/model_sess_->predictor_session);
      decoded_tokens.push_back(pred_token);
      continue;
    }
  }
  mnn::Tensor::destroy(enc_frame);  // Release enc frame tensor.
  this->UpdateStates(decoded_tokens);
}

std::string RnntGreedyDecoding::GetResults() {
  return this->tokenizer_->Decode(decoding_states_->partial_result);
}

void RnntGreedyDecoding::UpdateStates(const std::vector<int>& tokens) {
  this->decoding_states_->partial_result.insert(
      this->decoding_states_->partial_result.end(), tokens.begin(),
      tokens.end());
}

void RnntGreedyDecoding::ResetDecodingStates() {
  std::vector<int>().swap(this->decoding_states_->partial_result);
}

}  // namespace decoding
}  // namespace s2t