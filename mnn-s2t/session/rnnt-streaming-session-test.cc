// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.23
// Unittest of transducer Session.

#include "mnn-s2t/session/rnnt-streaming-session.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mnn-s2t/frontend/wav.h"

using namespace s2t;

class TestRnntStreamingSession : public ::testing::Test {
 protected:
  void SetUp() {
    this->SetUpRnntEncoderCfg();
    this->SetUpRnntPredictorCfg();
    this->SetUpJoinerCfg();
    this->SetUpSessionCfg();

    this->wav_reader_ = std::make_shared<frontend::WavReader>();
    // Build Rnnt rsrc shared among all session.
    this->rnnt_rsrc_ = std::make_shared<session::RnntRsrc>(
        enc_cfg_, pred_cfg_, joiner_cfg_,
        /*units_file=*/"../sample_data/units.txt");
    // Build transducer speech2text Session.
    this->session_ = std::make_shared<session::RnntStreamingSession>(
        rnnt_rsrc_, session_cfg_);
  }

  void SetUpRnntEncoderCfg() {
    enc_cfg_.chunk_size = 77;
    enc_cfg_.feat_dim = 80;
    enc_cfg_.enc_type = models::EncoderType::kZipformer;
    enc_cfg_.encoder_model = "../sample_data/models/streaming_zipformer.mnn";
  }

  void SetUpRnntPredictorCfg() {
    pred_cfg_.context_size = 5;
    pred_cfg_.predictor_model =
        "../sample_data/models/predictor_streaming_step.mnn";
  }

  void SetUpJoinerCfg() {
    joiner_cfg_.joiner_model =
        "../sample_data/models/joiner_streaming_step.mnn";
  }

  void SetUpSessionCfg() {
    session_cfg_.decoding_cfg.beam_size = 1;
    session_cfg_.decoding_cfg.cutoff_top_k = 1;
    session_cfg_.decoding_cfg.max_token_step = 1;
    session_cfg_.decoding_cfg.decoding_type =
        decoding::DecodingType::kRnntGreedyDecoding;
    session_cfg_.feat_chunk_size = 77;
    session_cfg_.pcm_normalize = true;
  }

  models::MnnEncoderCfg enc_cfg_;     // Encoder config
  models::MnnPredictorCfg pred_cfg_;  // Predictor config
  models::MnnJoinerCfg joiner_cfg_;   // Joiner config
  session::SessionCfg session_cfg_;   // Session config, frontend/decoding.
  std::shared_ptr<session::RnntRsrc> rnnt_rsrc_;
  std::shared_ptr<session::RnntStreamingSession> session_;
  std::shared_ptr<frontend::WavReader> wav_reader_;
};

TEST_F(TestRnntStreamingSession, TestRnntStreamingSessionInitRelease) {
  // Unittest of session init/release.
  this->session_->InitSession();
  this->session_->Reset();

  this->session_->InitSession();
  this->session_->Reset();

  this->session_->InitSession();
  this->session_->Reset();

  this->session_->InitSession();
  this->session_->Reset();
}

TEST_F(TestRnntStreamingSession, TestRnntStreamingSessionAcceptWaves) {
  this->session_->InitSession();
  std::string test_wav = "../sample_data/wavs/2086-149220-0019.wav";
  wav_reader_->Open(test_wav);
  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());
  this->session_->AcceptWaves(pcm);
  this->session_->Reset();
}

TEST_F(TestRnntStreamingSession, TestRnntStreamingSessionCompleteProcess) {
  this->session_->Reset();
  this->session_->InitSession();
  std::string test_wav = "../sample_data/wavs/2086-149220-0019.wav";
  wav_reader_->Open(test_wav);
  std::vector<float> pcm(wav_reader_->data(),
                         wav_reader_->data() + wav_reader_->num_samples());
  this->session_->AcceptWaves(pcm);
  this->session_->Process();
  this->session_->FinalizeSession();
  LOG(INFO) << this->session_->GetDecodedText();
  this->session_->Reset();
}