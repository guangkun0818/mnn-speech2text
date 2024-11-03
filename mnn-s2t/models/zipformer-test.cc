// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Unittest of wrapped Zipformer.

#include "mnn-s2t/models/zipformer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mnn-s2t/frontend/frontend-pipeline.h"
#include "mnn-s2t/frontend/wav.h"
#include "mnn-s2t/models/model-session.h"

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
        model, /*feat_dim=*/80, /*chunk_size=*/77,
        models::CPU_FORWARD_THREAD_8);
    model_sess_ = std::make_shared<models::RnntModelSession>();
  }

  std::string test_wav_ = "../sample_data/wavs/2086-149220-0019.wav";
  std::shared_ptr<models::MnnZipformer> mnn_zipformer_;
  std::shared_ptr<models::RnntModelSession> model_sess_;
  std::shared_ptr<frontend::StreamingFrontend> frontend_;
  std::shared_ptr<frontend::WavReader> wav_reader_;
};

TEST_F(TestMnnZipformer, TestModelInit) {
  // Unittest of model init/release.
  model_sess_->encoder_session =
      mnn_zipformer_->Init(mnn_zipformer_->ChunkSize());
  mnn_zipformer_->Reset(model_sess_->encoder_session);

  model_sess_->encoder_session =
      mnn_zipformer_->Init(mnn_zipformer_->ChunkSize());
  mnn_zipformer_->Reset(model_sess_->encoder_session);

  model_sess_->encoder_session =
      mnn_zipformer_->Init(mnn_zipformer_->ChunkSize());
  mnn_zipformer_->Reset(model_sess_->encoder_session);

  model_sess_->encoder_session =
      mnn_zipformer_->Init(mnn_zipformer_->ChunkSize());
  mnn_zipformer_->Reset(model_sess_->encoder_session);
}

TEST_F(TestMnnZipformer, TestModelInference) {
  wav_reader_->Open(test_wav_);
  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());
  mnn_zipformer_->Reset(model_sess_->encoder_session);
  model_sess_->encoder_session =
      mnn_zipformer_->Init(mnn_zipformer_->ChunkSize());

  frontend_->AcceptPcms(pcm);
  std::vector<std::vector<float>> feats;

  while (frontend_->IsReadyForFullChunk()) {
    frontend_->EmitFeats(feats);
    ASSERT_EQ(feats.size(), 77);             // Num of frames, chunk_size 77
    ASSERT_EQ((*feats.begin()).size(), 80);  // Num of feat_dims.

    mnn_zipformer_->StreamingStep(feats, model_sess_->encoder_session);
    // Demo test model streaming step output shape is {1, 16, 256}
    ASSERT_EQ(
        mnn_zipformer_->GetEncOut(model_sess_->encoder_session)->shape()[0], 1);
    ASSERT_EQ(
        mnn_zipformer_->GetEncOut(model_sess_->encoder_session)->shape()[1],
        16);
    ASSERT_EQ(
        mnn_zipformer_->GetEncOut(model_sess_->encoder_session)->shape()[2],
        256);
  }
}