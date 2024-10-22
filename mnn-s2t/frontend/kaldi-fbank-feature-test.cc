// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.06
// Unittest of kaldi-fbank-feature

#include "mnn-s2t/frontend/kaldi-fbank-feature.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mnn-s2t/frontend/wav.h"

using namespace s2t::frontend;

class TestFeatureWindow : public ::testing::Test {
 protected:
  void SetUp() {
    wav_reader_ = std::make_shared<WavReader>(test_wav_);
    opts_ = FbankOptions();
    opts_.mel_opts.num_bins = 80;  // 80 dim fbank.
    opts_.frame_opts.dither = 0.0f;
    opts_.frame_opts.snip_edges = false;  // Same setting in lhotes.
  }

  std::string test_wav_ = "../sample_data/wavs/2086-149220-0019.wav";
  std::shared_ptr<WavReader> wav_reader_;
  FbankOptions opts_;
};

TEST_F(TestFeatureWindow, TestNumFrames) {
  ASSERT_TRUE(wav_reader_->data() != nullptr);
  ASSERT_TRUE(wav_reader_->num_samples() > 0);

  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());
  auto num_frames = NumFrames(wav_reader_->num_samples(), opts_.frame_opts);
  ASSERT_EQ(num_frames, 1318);  // Should be same as lhotes framing.
}

TEST_F(TestFeatureWindow, TestExtractWindow) {
  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());
  auto num_frames = NumFrames(wav_reader_->num_samples(), opts_.frame_opts);

  auto window_function = FeatureWindowFunction(opts_.frame_opts);
  std::vector<float> window;

  for (int frame_id = 0; frame_id < num_frames; frame_id++) {
    std::fill(window.begin(), window.end(), 0);
    ExtractWindow(0, pcm, frame_id, opts_.frame_opts, window_function, &window,
                  nullptr);
  }
}

class TestKaldiFbankFeature : public ::testing::Test {
 protected:
  void SetUp() {
    opts_.mel_opts.num_bins = 80;        // 80 dim fbank.
    opts_.mel_opts.low_freq = 20.0f;     // Default setting in lhotes
    opts_.mel_opts.high_freq = -400.0f;  // Default setting in lhotes

    opts_.frame_opts.dither = 0.0f;
    opts_.frame_opts.snip_edges = false;  // Same setting in lhotes.
    opts_.energy_floor = 1e-10f;          // EPSILON = 1e-10 in lhotes.

    feat_computer_ = std::make_shared<FbankComputer>(opts_);
    wav_reader_ = std::make_shared<WavReader>();
  }

  std::string test_wav_ = "../sample_data/wavs/2086-149220-0019.wav";
  std::shared_ptr<FbankComputer> feat_computer_;
  std::shared_ptr<WavReader> wav_reader_;
  FbankOptions opts_;
};

TEST_F(TestKaldiFbankFeature, TestComputeFeats) {
  wav_reader_->Open(test_wav_);
  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());

  // Normalize pcm in range (-1, 1)
  for (auto iter = pcm.begin(); iter != pcm.end(); iter++) {
    *iter = (*iter) / 32768;
  }

  ASSERT_EQ(NumFrames(12560, opts_.frame_opts), 79);
  auto num_frames = NumFrames(wav_reader_->num_samples(), opts_.frame_opts);

  auto window_function = FeatureWindowFunction(opts_.frame_opts);

  std::vector<std::vector<float>> feats;
  bool need_raw_log_energy =
      feat_computer_->NeedRawLogEnergy();  // Default is no need.
  // note: this online feature-extraction code does not support VTLN.
  float vtln_warp = 1.0;

  std::vector<float> window;
  for (int frame_id = 0; frame_id < num_frames; frame_id++) {
    std::fill(window.begin(), window.end(), 0);
    float raw_log_energy = 0.0;
    ExtractWindow(0, pcm, frame_id, opts_.frame_opts, window_function, &window,
                  need_raw_log_energy ? &raw_log_energy : nullptr);

    // With default setting of NeedRawLogEnergy = false, feat_dim = num_bins,
    // otherwise feat_dim = num_bins + 1.
    std::vector<float> frame_feat(feat_computer_->Dim());
    feat_computer_->Compute(raw_log_energy /*=0.0 as deflaut*/,
                            vtln_warp /*=1.0 as default*/, &window,
                            frame_feat.data());
    feats.push_back(std::move(frame_feat));
  }
}