// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.10
// Unittest of frontend pipeline impl

#include "mnn-s2t/frontend/frontend-pipeline.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mnn-s2t/frontend/wav.h"

using namespace s2t::frontend;

class TestNonStreamingFrontend : public ::testing::Test {
 protected:
  void SetUp() {
    wav_reader_ = std::make_shared<WavReader>();

    FbankOptions opts;
    opts.mel_opts.num_bins = 80;        // 80 dim fbank.
    opts.mel_opts.low_freq = 20.0f;     // Default setting in lhotes
    opts.mel_opts.high_freq = -400.0f;  // Default setting in lhotes

    opts.frame_opts.dither = 0.0f;
    opts.frame_opts.snip_edges =
        true;                    // Different with default setting of lhotes.
    opts.energy_floor = 1e-10f;  // EPSILON = 1e-10 in lhotes.
    frontend_ = std::make_shared<Frontend>(opts, true);
  }

  std::string test_wav_ = "../sample_data/wavs/2086-149220-0019.wav";
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

  ASSERT_EQ(feats.size(), 1316);           // Num of frames.
  ASSERT_EQ((*feats.begin()).size(), 80);  // Num of feat_dims.
}

class TestStreamingFrontend : public ::testing::Test {
 protected:
  void SetUp() {
    wav_reader_ = std::make_shared<WavReader>();
    feat_dim_ = 80;

    FbankOptions opts;

    opts.mel_opts.num_bins = feat_dim_;  // 80 dim fbank.
    opts.mel_opts.low_freq = 20.0f;      // Default setting in lhotes
    opts.mel_opts.high_freq = -400.0f;   // Default setting in lhotes

    opts.frame_opts.dither = 0.0f;
    opts.frame_opts.snip_edges =
        true;                    // Different with default setting of lhotes.
    opts.energy_floor = 1e-10f;  // EPSILON = 1e-10 in lhotes.
    feat_dim_ = opts.mel_opts.num_bins;

    chunk_size_ = 77;
    frontend_ = std::make_shared<Frontend>(opts, true);
    streaming_frontend_ =
        std::make_shared<StreamingFrontend>(opts, chunk_size_, true);
  }

  int32_t feat_dim_;
  int32_t chunk_size_;
  std::string test_wav_ = "../sample_data/wavs/2086-149220-0019.wav";
  std::shared_ptr<WavReader> wav_reader_;
  std::shared_ptr<Frontend> frontend_;
  std::shared_ptr<StreamingFrontend> streaming_frontend_;
};

TEST_F(TestStreamingFrontend, TestPipelineProcessOneChunk) {
  wav_reader_->Open(test_wav_);
  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());

  streaming_frontend_->AcceptPcms(pcm);
  std::vector<std::vector<float>> feats;
  streaming_frontend_->EmitFeats(feats);

  ASSERT_EQ(feats.size(), chunk_size_);           // Num of frames.
  ASSERT_EQ((*feats.begin()).size(), feat_dim_);  // Num of feat_dims.

  streaming_frontend_->EmitFeats(feats, true);
  ASSERT_EQ(feats.size(), chunk_size_);           // Num of frames.
  ASSERT_EQ((*feats.begin()).size(), feat_dim_);  // Num of feat_dims.
}

TEST_F(TestStreamingFrontend, TestPrecisionCheck) {
  wav_reader_->Open(test_wav_);
  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());

  frontend_->AcceptPcms(pcm);
  std::vector<std::vector<float>> offline_feats;
  frontend_->EmitFeats(offline_feats);

  streaming_frontend_->AcceptPcms(pcm);
  std::vector<std::vector<float>> feats_chunk;

  // Precision check with Non-streaming frontend
  int chunk_id = 0;
  int num_frames = 0;
  while (streaming_frontend_->IsReadyForFullChunk()) {
    streaming_frontend_->EmitFeats(feats_chunk);
    num_frames += feats_chunk.size();

    // Presicion check over all elems.
    for (int frame_id = 0; frame_id < chunk_size_; frame_id++) {
      for (int j = 0; j < feat_dim_; j++) {
        ASSERT_FLOAT_EQ(offline_feats[chunk_id * chunk_size_ + frame_id][j],
                        feats_chunk[frame_id][j]);
      }
    }
    chunk_id++;
  }

  // Process last chunk.
  streaming_frontend_->EmitFeats(feats_chunk, true);
  num_frames += feats_chunk.size();
  ASSERT_EQ(num_frames, offline_feats.size());
  for (int frame_id = 0; frame_id < feats_chunk.size(); frame_id++) {
    for (int j = 0; j < feat_dim_; j++) {
      ASSERT_FLOAT_EQ(offline_feats[chunk_id * chunk_size_ + frame_id][j],
                      feats_chunk[frame_id][j]);
    }
  }
  // Enable padding on last chunk.
  streaming_frontend_->PadIntoFullChunk(feats_chunk);
  ASSERT_EQ(chunk_size_, feats_chunk.size());
}