// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.23
// Transducer Session.

#include "mnn-s2t/session/rnnt-streaming-session.h"

namespace s2t {
namespace session {

RnntStreamingSession::RnntStreamingSession(
    const std::shared_ptr<RnntRsrc>& rnnt_rsrc, const SessionCfg& session_cfg)
    : rnnt_rsrc_(rnnt_rsrc) {
  // Session rsrc is exclusivly correspond to each RnntStreamingSession.
  session_rsrc_ = std::make_shared<SessionRsrc>();

  // Build Streaming frontend.
  session_rsrc_->frontend = std::make_shared<frontend::StreamingFrontend>(
      frontend::LHOTSE_FBANK_OPTIONS(), session_cfg.feat_chunk_size,
      session_cfg.pcm_normalize);

  // Build model session.
  session_rsrc_->model_session = std::make_shared<models::RnntModelSession>();
  // Build Decoding method.
  switch (session_cfg.decoding_cfg.decoding_type) {
    case decoding::DecodingType::kRnntGreedyDecoding:
      LOG(INFO) << "Decoding method : RnntGreedyDecoding setected.";
      session_rsrc_->decoding = std::make_shared<decoding::RnntGreedyDecoding>(
          rnnt_rsrc_->predictor, rnnt_rsrc_->joiner,
          session_rsrc_->model_session, rnnt_rsrc_->tokenizer,
          session_cfg.decoding_cfg);
      break;
    default:
      LOG(WARNING) << "Unsupported decoding type.";
      break;
  }
  this->InitSession();
}

RnntStreamingSession::~RnntStreamingSession() { this->Reset(); }

void RnntStreamingSession::InitSession() {
  // Reset Streaming frontend
  session_rsrc_->frontend->Reset();
  // Init Encoder Session.
  session_rsrc_->model_session->encoder_session =
      rnnt_rsrc_->encoder->Init(rnnt_rsrc_->encoder->ChunkSize());
  // Init Decoding and predictor/joiner;
  session_rsrc_->decoding->Init();
  this->decoded_text_.clear();  // Clear partial results
}

void RnntStreamingSession::Reset() {
  // Reset Streaming frontend
  session_rsrc_->frontend->Reset();
  // Release encoder session.
  session_rsrc_->model_session->encoder_session =
      rnnt_rsrc_->encoder->Reset(session_rsrc_->model_session->encoder_session);
  // Release predictor/joiner session, which prossessed by decoding method,
  // and decoding states.
  session_rsrc_->decoding->Reset();
  this->decoded_text_.clear();  // Clear partial results
}

void RnntStreamingSession::AcceptWaves(const std::vector<float>& pcms) {
  session_rsrc_->frontend->AcceptPcms(pcms);
}

void RnntStreamingSession::Process() {
  std::vector<std::vector<float>> feats;
  while (session_rsrc_->frontend->IsReadyForFullChunk()) {
    // Extract acoustic feats.
    session_rsrc_->frontend->EmitFeats(feats);
    // Encoder streaming forward.
    rnnt_rsrc_->encoder->StreamingStep(
        feats, session_rsrc_->model_session->encoder_session);  // Decoding.
    session_rsrc_->decoding->Decode(rnnt_rsrc_->encoder->GetEncOut(
        session_rsrc_->model_session->encoder_session));

    this->decoded_text_ = session_rsrc_->decoding->GetResults();
  }
}

void RnntStreamingSession::FinalizeSession() {
  // Finalize session by process remainder pcms, which is not full chunk.
  CHECK_NE(session_rsrc_->frontend->IsReadyForFullChunk(), true);
  std::vector<std::vector<float>> feats;
  session_rsrc_->frontend->EmitFeats(feats);
  session_rsrc_->frontend->PadIntoFullChunk(feats);  // Enable padding.
  // Encoder streaming forward.
  rnnt_rsrc_->encoder->StreamingStep(
      feats, session_rsrc_->model_session->encoder_session);
  // Decoding.
  session_rsrc_->decoding->Decode(rnnt_rsrc_->encoder->GetEncOut(
      session_rsrc_->model_session->encoder_session));
  this->decoded_text_ = session_rsrc_->decoding->GetResults();
}

std::string RnntStreamingSession::GetDecodedText() const {
  return this->decoded_text_;
}

}  // namespace session
}  // namespace s2t