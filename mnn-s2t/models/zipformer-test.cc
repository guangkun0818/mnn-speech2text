// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Unittest of wrapped Zipformer.

#include "mnn-s2t/models/zipformer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mnn-s2t/frontend/frontend-pipeline.h"
#include "mnn-s2t/frontend/wav.h"

using namespace s2t;

class TestMnnZipformer : public ::testing::Test {
 protected:
  void SetUp() {
    wav_reader_ = std::make_shared<frontend::WavReader>();

    frontend::FbankOptions opts;
    opts.mel_opts.num_bins = 80;        // 80 dim fbank.
    opts.mel_opts.low_freq = 20.0f;     // Default setting in lhotes
    opts.mel_opts.high_freq = -400.0f;  // Default setting in lhotes

    opts.frame_opts.dither = 0.0f;
    opts.frame_opts.snip_edges =
        true;                    // Different with default setting of lhotes.
    opts.energy_floor = 1e-10f;  // EPSILON = 1e-10 in lhotes.
    frontend_ = std::make_shared<frontend::StreamingFrontend>(
        opts, /*chunk_size=*/77, /*pcm_normalize=*/true);

    const char* model = "../sample_data/models/streaming_zipformer.mnn";
    mnn_zipformer_ = std::make_shared<models::MnnZipformer>(
        model, /*feat_dim=*/80, /*chunk_size=*/77);
  }

  std::string test_wav_ = "../sample_data/wavs/2086-149220-0019.wav";
  std::shared_ptr<models::MnnZipformer> mnn_zipformer_;
  std::shared_ptr<frontend::StreamingFrontend> frontend_;
  std::shared_ptr<frontend::WavReader> wav_reader_;
};

TEST_F(TestMnnZipformer, TestModelInit) {
  mnn_zipformer_->Init(mnn_zipformer_->ChunkSize());
}

TEST_F(TestMnnZipformer, TestModelInference) {
  wav_reader_->Open(test_wav_);
  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());

  frontend_->AcceptPcms(pcm);
  std::vector<std::vector<float>> feats;
  frontend_->EmitFeats(feats);

  ASSERT_EQ(feats.size(), 77);             // Num of frames, chunk_size 77
  ASSERT_EQ((*feats.begin()).size(), 80);  // Num of feat_dims.

  mnn_zipformer_->Init(feats.size());
  mnn_zipformer_->StreamingStep(feats);
}