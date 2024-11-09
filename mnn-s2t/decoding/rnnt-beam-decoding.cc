// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.11.09
// Beam decoding of Transducer.

#include "mnn-s2t/decoding/rnnt-beam-decoding.h"

#include <numeric>

namespace s2t {
namespace decoding {

DecodingBeam::DecodingBeam(const std::vector<int>& updated_decoded_tokens,
                           bool updated_end_with_blank, float updated_score,
                           const mnn::Tensor* last_pred_out,
                           const mnn::Tensor* last_pred_state) {
  decoded_tokens.assign(updated_decoded_tokens.begin(),
                        updated_decoded_tokens.end());
  end_with_blank = updated_end_with_blank;
  score = updated_score;
  pred_out = mnn::Tensor::create<float>(last_pred_out->shape(), NULL,
                                        MNN::Tensor::CAFFE);
  memcpy(pred_out->host<float>(), last_pred_out->host<float>(),
         sizeof(float) * last_pred_out->elementSize());
  pred_state = mnn::Tensor::create<int>(last_pred_state->shape(), NULL,
                                        MNN::Tensor::CAFFE);
  memcpy(pred_state->host<int>(), last_pred_state->host<int>(),
         sizeof(int) * last_pred_state->elementSize());
};

DecodingBeam::~DecodingBeam() {
  // Release cached pred_out/pred_state.
  mnn::Tensor::destroy(pred_out);
  mnn::Tensor::destroy(pred_state);
}

RnntBeamDecoding::RnntBeamDecoding(
    const std::shared_ptr<models::MnnPredictor>& predictor,
    const std::shared_ptr<models::MnnJoiner>& joiner,
    const std::shared_ptr<models::RnntModelSession>& model_sess,
    const std::shared_ptr<SubwordTokenizer>& tokenizer, const DecodingCfg& cfg)
    : predictor_(predictor),
      joiner_(joiner),
      model_sess_(model_sess),
      tokenizer_(tokenizer),
      beam_size_(cfg.beam_size),
      cutoff_top_k_(cfg.cutoff_top_k),
      on_start_(true) {}

void RnntBeamDecoding::Init() {
  // Decoding setup, init predictor state, joiner is cache-free. Deocding
  // process should be start with <blank_id> = 0.
  model_sess_->predictor_session = predictor_->Init(1);
  model_sess_->joiner_session = joiner_->Init(this->beam_size_);
  predictor_->StreamingStep(
      {0}, model_sess_->predictor_session);  // Init predictor with <blank_id>.

  // Init beams.
  for (int i = 0; i < this->beam_size_; ++i) {
    this->beams_.emplace_back(new DecodingBeam(
        /*updated_decoded_tokens=*/{},
        /*updated_end_with_blank=*/true,
        /*updated_score=*/0.0,
        /*last_pred_out=*/
        predictor_->GetPredOut(model_sess_->predictor_session),
        /*last_pred_state=*/
        predictor_->GetPredState(model_sess_->predictor_session)));
  }
}

void RnntBeamDecoding::Reset() {
  model_sess_->predictor_session = predictor_->Reset(
      model_sess_->predictor_session);  // Release Predictor session.
  model_sess_->joiner_session =
      joiner_->Reset(model_sess_->joiner_session);  // Release Joiner session.
  this->ResetDecodingStates();
}

bool RnntBeamDecoding::IsBlank(int token) const { return token == 0; }

void RnntBeamDecoding::ArgSort(const std::vector<float>& log_probs,
                               std::vector<size_t>& args) const {
  CHECK_EQ(log_probs.size(), args.size());
  std::iota(args.begin(), args.end(), 0);  // Initializing indexs of log_probs
  std::sort(args.begin(), args.end(),
            [&](int i, int j) { return log_probs[i] > log_probs[j]; });
};

void RnntBeamDecoding::Decode(mnn::Tensor* enc_out) {
  CHECK_EQ(enc_out->shape()[0], 1);           // Batch Size = 1.
  auto tot_time_steps = enc_out->shape()[1];  // (1, tot_time_steps, enc_dim)
  auto enc_dim = enc_out->shape()[2];

  std::vector<int> enc_frame_shape = {1, /*tot_time_step=*/1,
                                      /*enc_dim=*/enc_dim};
  auto enc_frame =
      mnn::Tensor::create<float>(enc_frame_shape, NULL, MNN::Tensor::CAFFE);

  auto pred_out_dim = predictor_->GetPredOut(model_sess_->predictor_session)
                          ->shape()[2];  // pred_out shape: {1, 1, pred_out_dim}
  std::vector<int> pred_out_beam_shape = {beam_size_, 1, pred_out_dim};
  auto pred_out_beam =
      mnn::Tensor::create<float>(pred_out_beam_shape, NULL, MNN::Tensor::CAFFE);
  int curr_time_step = 0;
  while (curr_time_step < tot_time_steps) {
    // Slice encoder_out as frame
    memcpy(enc_frame->host<float>(),
           enc_out->host<float>() + curr_time_step * enc_dim,
           sizeof(float) * enc_dim);
    this->BuildBeamPredOut(pred_out_beam);

    // Joiner streaming step.
    joiner_->StreamingStep(
        /*enc_out=*/enc_frame,
        /*pred_out=*/pred_out_beam,
        /*joiner_session=*/model_sess_->joiner_session);
    auto log_probs = joiner_->GetJoinerOut(model_sess_->joiner_session);

    this->UpdateBeams(log_probs, on_start_);
    this->on_start_ = false;  // Once start, switch on_start decoding state;

    for (auto beam : this->beams_) {
      if (!beam->end_with_blank) {
        predictor_->StreamingStep({beam->decoded_tokens.back()},
                                  beam->pred_state,
                                  model_sess_->predictor_session);
        // Update beam state.
        beam->end_with_blank = true;
        memcpy(beam->pred_out->host<float>(),
               predictor_->GetPredOut(model_sess_->predictor_session)
                   ->host<float>(),
               sizeof(float) * beam->pred_out->elementSize());
        memcpy(beam->pred_state->host<int>(),
               predictor_->GetPredState(model_sess_->predictor_session)
                   ->host<int>(),
               sizeof(int) * beam->pred_state->elementSize());
      }
    }
    curr_time_step++;
  }
  mnn::Tensor::destroy(enc_frame);      // Release enc frame tensor.
  mnn::Tensor::destroy(pred_out_beam);  // Release pred_out_beam tensor.
}

std::string RnntBeamDecoding::GetResults() {
  return this->tokenizer_->Decode(
      beams_[0]->decoded_tokens);  // Return best beam.
};

void RnntBeamDecoding::BuildBeamPredOut(mnn::Tensor* pred_out_beam) {
  // Concat pred_out within beams.
  auto pred_out_dim =
      pred_out_beam
          ->shape()[2];  // pred_out_beam shape: {beam_size, 1, pred_out}
  CHECK_EQ(pred_out_beam->shape()[0], this->beams_.size());
  for (int beam_id = 0; beam_id < this->beam_size_; ++beam_id) {
    memcpy(/*dest=*/pred_out_beam->host<float>() + beam_id * pred_out_dim,
           /*src=*/this->beams_[beam_id]->pred_out->host<float>(),
           /*n_bytes=*/sizeof(float) * pred_out_dim);
  }
}

void RnntBeamDecoding::UpdateBeams(
    const std::vector<std::vector<float>>& log_probs, bool is_start) {
  // Update decoding state with log_probs of current time step.
  std::vector<DecodingBeam*> new_beams;

  int valid_beam_size = is_start ? 1 : this->beams_.size();
  for (int beam_id = 0; beam_id < valid_beam_size; ++beam_id) {
    auto beam = this->beams_[beam_id];
    std::vector<size_t> token_idxs(log_probs[beam_id].size());  // Num of tokens
    this->ArgSort(log_probs[beam_id], token_idxs);

    // Only iterate over top_k possible tokens at this timestep.
    for (auto token_id_it = token_idxs.begin();
         token_id_it != token_idxs.begin() + this->cutoff_top_k_;
         ++token_id_it) {
      if (IsBlank(*token_id_it)) {
        // If is <blank_id>, update beam with pred_out and pred_state indicating
        // this beam would not update at this timestep.
        auto updated_score =
            this->beams_[beam_id]->score + log_probs[beam_id][*token_id_it];
        new_beams.emplace_back(new DecodingBeam(
            /*updated_decoded_tokens=*/this->beams_[beam_id]->decoded_tokens,
            /*updated_end_with_blank=*/true,
            /*updated_score=*/updated_score,
            /*last_pred_out=*/this->beams_[beam_id]->pred_out,
            /*last_pred_state=*/this->beams_[beam_id]->pred_state));
      } else {
        // If not <blank_id>, update Beams with new beam, new beam will update
        // pred state only, cos at end of this iteration of this time step
        // predictor would move forward
        // one step on token direction, which will update pred_state and
        // pred_out.
        std::vector<int> updated_decoded_tokens(
            this->beams_[beam_id]->decoded_tokens);
        auto updated_score =
            this->beams_[beam_id]->score + log_probs[beam_id][*token_id_it];
        updated_decoded_tokens.push_back(*token_id_it);
        new_beams.emplace_back(new DecodingBeam(
            /*updated_decoded_tokens=*/updated_decoded_tokens,
            /*updated_end_with_blank=*/false,
            /*updated_score=*/updated_score,
            /*last_pred_out=*/this->beams_[beam_id]->pred_out,
            /*last_pred_state=*/this->beams_[beam_id]->pred_state));
      }
    }
  }

  // Sort new_beams.
  std::sort(new_beams.begin(), new_beams.end(),
            [](const DecodingBeam* beam_i, const DecodingBeam* beam_j) {
              return beam_i->score > beam_j->score;
            });
  // Keep top_beam_size beams while release others;
  if (new_beams.size() > this->beam_size_) {
    new_beams.resize(this->beam_size_);  // Delete pruned beam with resize.
  }
  this->ResetDecodingStates();
  new_beams.swap(this->beams_);
}

void RnntBeamDecoding::ResetDecodingStates() {
  for (auto beam : beams_) {
    delete beam;  // Release all beams
  }
  std::vector<DecodingBeam*>().swap(beams_);
}

}  // namespace decoding
}  // namespace s2t