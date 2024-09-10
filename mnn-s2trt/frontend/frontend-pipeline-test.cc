// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.10
// Unittest of frontend pipeline impl

#include "mnn-s2trt/frontend/frontend-pipeline.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mnn-s2trt/frontend/wav.h"

using namespace s2trt::frontend;

class TestNonStreamingFrontend : public ::testing::Test {
 protected:
  void SetUp() {
    wav_reader_ = std::make_shared<WavReader>();

    FbankOptions opts;
    opts.mel_opts.num_bins = 80;        // 80 dim fbank.
    opts.mel_opts.low_freq = 20.0f;     // Default setting in lhotes
    opts.mel_opts.high_freq = -400.0f;  // Default setting in lhotes

    opts.frame_opts.dither = 0.0f;
    opts.frame_opts.snip_edges = false;  // Same setting in lhotes.
    opts.energy_floor = 1e-10f;          // EPSILON = 1e-10 in lhotes.
    frontend_ = std::make_shared<Frontend>(opts, true);
  }

  std::string test_wav_ = "sample_data/wavs/2086-149220-0019.wav";
  std::shared_ptr<WavReader> wav_reader_;
  std::shared_ptr<Frontend> frontend_;
};

TEST_F(TestNonStreamingFrontend, TestPipelineProcess) {
  wav_reader_->Open(test_wav_);
  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());

  frontend_->AcceptPcms(pcm);
  std::vector<std::vector<float>> feats;
  frontend_->EmitFeats(feats);

  ASSERT_EQ(feats.size(), 1318);           // Num of frames.
  ASSERT_EQ((*feats.begin()).size(), 80);  // Num of feat_dims.
}

class TestStreamingFrontend : public ::testing::Test {
 protected:
  void SetUp() {
    wav_reader_ = std::make_shared<WavReader>();

    FbankOptions opts;
    opts.mel_opts.num_bins = 80;        // 80 dim fbank.
    opts.mel_opts.low_freq = 20.0f;     // Default setting in lhotes
    opts.mel_opts.high_freq = -400.0f;  // Default setting in lhotes

    opts.frame_opts.dither = 0.0f;
    opts.frame_opts.snip_edges = false;  // Same setting in lhotes.
    opts.energy_floor = 1e-10f;          // EPSILON = 1e-10 in lhotes.

    int32_t chunk_size = 77;
    frontend_ = std::make_shared<StreamingFrontend>(opts, chunk_size, true);
  }

  std::string test_wav_ = "sample_data/wavs/2086-149220-0019.wav";
  std::shared_ptr<WavReader> wav_reader_;
  std::shared_ptr<StreamingFrontend> frontend_;
};

TEST_F(TestStreamingFrontend, TestPipelineProcess) {
  wav_reader_->Open(test_wav_);
  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());

  frontend_->AcceptPcms(pcm);
  std::vector<std::vector<float>> feats;
  frontend_->EmitFeats(feats);

  ASSERT_EQ(feats.size(), 77);             // Num of frames.
  ASSERT_EQ((*feats.begin()).size(), 80);  // Num of feat_dims.
}