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
    frontend_ = std::make_shared<Frontend>(LHOTSE_FBANK_OPTIONS(), true);
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

    chunk_size_ = 77;
    frontend_ = std::make_shared<Frontend>(LHOTSE_FBANK_OPTIONS(), true);
    streaming_frontend_ = std::make_shared<StreamingFrontend>(
        LHOTSE_FBANK_OPTIONS(), chunk_size_, true);
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

  // Test streaming frontend for 3 times.
  int iter = 3;
  while (iter > 0) {
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

    streaming_frontend_->Reset();
    iter--;
  }
}