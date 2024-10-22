// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.10
// Frontend pipeline impl of speech2text session.

#include "mnn-s2t/frontend/frontend-pipeline.h"

#include "glog/logging.h"

namespace s2t {
namespace frontend {

Frontend::Frontend(const FbankOptions& opts, const bool pcm_normalize)
    : pcm_normalize_(pcm_normalize), opts_(opts) {
  // Mandatorily using default setting in lhotes_feats, since it might not
  // changed In most cases
  CHECK_EQ(opts_.mel_opts.num_bins, 80);        // 80 dim fbank.
  CHECK_EQ(opts_.mel_opts.low_freq, 20.0f);     // Default setting in lhotes
  CHECK_EQ(opts_.mel_opts.high_freq, -400.0f);  // Default setting in lhotes
  CHECK_EQ(opts_.frame_opts.dither, 0.0f);
  CHECK_EQ(opts_.frame_opts.snip_edges, false);  // Same setting in lhotes.
  CHECK_EQ(opts_.energy_floor, 1e-10f);          // EPSILON = 1e-10 in lhotes.

  window_function_ = FeatureWindowFunction(opts_.frame_opts);
  feat_computer_ = std::make_shared<FbankComputer>(opts_);
}

void Frontend::AcceptPcms(const std::vector<float>& pcms) {
  this->Reset();  // Reset every time pcm flush in.
  if (this->pcm_normalize_) {
    for (auto sample : pcms) {
      pcms_ready_.push_back(sample * NORMALIZE_FACTOR);
    }
  } else {
    pcms_ready_.insert(pcms_ready_.end(), pcms.begin(), pcms.end());
  }
}

void Frontend::EmitFeats(std::vector<std::vector<float>>& feats, bool is_last) {
  feats.clear();
  auto num_frames = NumFrames(pcms_ready_.size(), opts_.frame_opts);
  CHECK_GT(num_frames, 0);  // should be larger than 0

  bool need_raw_log_energy =
      feat_computer_->NeedRawLogEnergy();  // Default is no need.
  float vtln_warp = 1.0;

  std::vector<float> window;
  for (int frame_id = 0; frame_id < num_frames; frame_id++) {
    std::fill(window.begin(), window.end(), 0);
    float raw_log_energy = 0.0;
    ExtractWindow(0, pcms_ready_, frame_id, opts_.frame_opts, window_function_,
                  &window, need_raw_log_energy ? &raw_log_energy : nullptr);

    // With default setting of NeedRawLogEnergy = false, feat_dim = num_bins,
    // otherwise feat_dim = num_bins + 1.
    std::vector<float> frame_feat(feat_computer_->Dim());
    feat_computer_->Compute(raw_log_energy /*=0.0 as deflaut*/,
                            vtln_warp /*=1.0 as default*/, &window,
                            frame_feat.data());
    feats.push_back(std::move(frame_feat));
  }
}

StreamingFrontend::StreamingFrontend(const FbankOptions& opts,
                                     int32_t feat_chunk_size,
                                     const bool pcm_normalize)
    : Frontend(opts, pcm_normalize), feat_chunk_size_(feat_chunk_size) {
  /* Framing strategy of kaldi-fbank-feats with default setting of lhotes:
      samp_freq = 16000
      frame_shift_ms = 10.0f
      frame_length_ms = 25.0f
      snip_edges = false

     first frame:  |------25ms-------|
                   |-10ms-|---15ms---|

     second frame: |-10ms-|------25ms------|
                   |-10ms-|---15ms---|
        residual pcm 15ms >= frame_shift / 2, enable padding.

     third frame:  |-10ms-|-10ms-|------25ms------|
                   |-10ms-|-10ms-|5ms|
        residual pcm 5ms >= frame_shift / 2, enable padding.
  */

  // pcm_chunk_size_ = (25ms + (feat_chunk_size - 1) * 10ms) * 16000
  pcm_chunk_size_ = ((feat_chunk_size_ - 1) * opts_.frame_opts.frame_shift_ms +
                     opts_.frame_opts.frame_length_ms) *
                    opts_.frame_opts.samp_freq / 1000;
  // pcm_cache_size_ = (25ms - 10ms) * 16000
  pcm_cache_size_ =
      (opts_.frame_opts.frame_length_ms - opts_.frame_opts.frame_shift_ms) *
      opts_.frame_opts.samp_freq / 1000;
  this->Reset();
}

void StreamingFrontend::Reset() {
  pcms_ready_.clear();
  pcms_pending_.clear();
  num_pending_pcms_ = 0;
  start_offset_of_front_ = 0;
  last_pcm_cache_.clear();
}

bool StreamingFrontend::IsReady() const {
  return num_pending_pcms_ + last_pcm_cache_.size() > pcm_chunk_size_;
}

void StreamingFrontend::AcceptPcms(const std::vector<float>& pcms) {
  std::vector<float> coming_pcm;
  if (this->pcm_normalize_) {
    for (auto sample : pcms) {
      coming_pcm.push_back(sample * NORMALIZE_FACTOR);
    }
  } else {
    coming_pcm.insert(coming_pcm.end(), pcms.begin(), pcms.end());
  }
  this->pcms_pending_.push_back(coming_pcm);
  this->num_pending_pcms_ += coming_pcm.size();
}

void StreamingFrontend::PreparePcms() {
  // Preppend last pcm cache.
  pcms_ready_.clear();
  pcms_ready_.insert(pcms_ready_.end(), last_pcm_cache_.begin(),
                     last_pcm_cache_.end());
  auto pcm_required_size =
      pcm_chunk_size_ -
      pcms_ready_.size();  // Residual required pcms for one chunk.

  this->ComsumePendingPcms(pcm_required_size);

  // Update last_pcm_cache.
  std::vector<float>().swap(last_pcm_cache_);
  last_pcm_cache_.insert(last_pcm_cache_.end(),
                         pcms_ready_.end() - pcm_cache_size_,
                         pcms_ready_.end());
  CHECK_EQ(last_pcm_cache_.size(), pcm_cache_size_);
  CHECK_EQ(pcms_ready_.size(), pcm_chunk_size_);
}

void StreamingFrontend::ComsumePendingPcms(size_t pcm_required_size) {
  while (pcm_required_size > 0) {
    if ((pcms_pending_.front().size() - start_offset_of_front_) <=
        pcm_required_size) {
      // Comsume the front slice of pcm if the front slice of pending pcm cannot
      // fullfill or just match the required residual, then pop out the front
      // slice.
      auto start_offset =
          pcms_pending_.front().begin() + start_offset_of_front_;
      auto end_offset = pcms_pending_.front().end();
      pcms_ready_.insert(pcms_ready_.end(), start_offset, end_offset);

      auto pcm_cosumed_size =
          pcms_pending_.front().size() - start_offset_of_front_;

      // Update num_pending_pcms_/pcm_required_size/start_offset_of_front_
      start_offset_of_front_ = 0;  // Reset to start of next pcm slice.
      num_pending_pcms_ -= pcm_cosumed_size;
      pcm_required_size -= pcm_cosumed_size;

      pcms_pending_.pop_front();

    } else {
      // Or only move forward start_offset of front pending pcm queue.
      auto start_offset =
          pcms_pending_.front().begin() + start_offset_of_front_;
      auto end_offset = pcms_pending_.front().begin() + start_offset_of_front_ +
                        pcm_required_size;
      pcms_ready_.insert(pcms_ready_.end(), start_offset, end_offset);

      auto pcm_cosumed_size = pcm_required_size;

      // Update num_pending_pcms_/pcm_required_size/start_offset_of_front_
      start_offset_of_front_ += pcm_cosumed_size;
      num_pending_pcms_ -= pcm_cosumed_size;
      pcm_required_size -= pcm_cosumed_size;
    }
  }

  CHECK_EQ(pcm_required_size, 0);
}

void StreamingFrontend::EmitFeats(std::vector<std::vector<float>>& feats,
                                  bool is_last) {
  feats.clear();
  if (not this->IsReady()) {
    LOG(FATAL) << "Pending pcm is not enought to emit feat chunks.";
  }
  this->PreparePcms();

  auto num_frames = NumFrames(pcms_ready_.size(), opts_.frame_opts);
  // Last 2 frames should be discarded since it involved with padding
  // with given setting frame_shift_ms = 10.0f / frame_length_ms = 25.0f
  // if not the last chunk.
  num_frames = is_last ? num_frames : num_frames - 2;

  CHECK_GT(num_frames, 0);  // should be larger than 0

  bool need_raw_log_energy =
      feat_computer_->NeedRawLogEnergy();  // Default is no need.
  float vtln_warp = 1.0;

  std::vector<float> window;
  for (int frame_id = 0; frame_id < num_frames; frame_id++) {
    std::fill(window.begin(), window.end(), 0);
    float raw_log_energy = 0.0;
    ExtractWindow(0, pcms_ready_, frame_id, opts_.frame_opts, window_function_,
                  &window, need_raw_log_energy ? &raw_log_energy : nullptr);

    // With default setting of NeedRawLogEnergy = false, feat_dim = num_bins,
    // otherwise feat_dim = num_bins + 1.
    std::vector<float> frame_feat(feat_computer_->Dim());
    feat_computer_->Compute(raw_log_energy /*=0.0 as deflaut*/,
                            vtln_warp /*=1.0 as default*/, &window,
                            frame_feat.data());
    feats.push_back(std::move(frame_feat));
  }
}

}  // namespace frontend
}  // namespace s2t