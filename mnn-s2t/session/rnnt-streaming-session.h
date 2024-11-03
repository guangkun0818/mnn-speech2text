// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.23
// Transducer Session.

#ifndef _MNN_S2T_SESSION_RNNT_SESSION_H_
#define _MNN_S2T_SESSION_RNNT_SESSION_H_

#include <memory>
#include <string>

#include "mnn-s2t/decoding/decoding.h"
#include "mnn-s2t/decoding/rnnt-greedy-decoding.h"
#include "mnn-s2t/frontend/frontend-pipeline.h"
#include "mnn-s2t/models/model-session.h"
#include "mnn-s2t/session/rnnt-rsrc.h"

namespace s2t {
namespace session {

struct SessionCfg {
  // Frontend setting.
  int32_t feat_chunk_size;
  bool pcm_normalize = true;
  // Decoding setting.
  decoding::DecodingCfg decoding_cfg;
};

struct SessionRsrc {
  std::shared_ptr<frontend::StreamingFrontend> frontend;
  std::shared_ptr<models::RnntModelSession> model_session;
  std::shared_ptr<decoding::DecodingMethod> decoding;
};

class RnntStreamingSession {
 public:
  explicit RnntStreamingSession(const std::shared_ptr<RnntRsrc>& rnnt_rsrc,
                                const SessionCfg& session_cfg);

  ~RnntStreamingSession();

  // Init Session right after Session built.
  void InitSession();

  // Flush streaming audio data into session.
  void AcceptWaves(const std::vector<float>& pcms);

  // Process flushed pcms, which can be async with AcceptWaves()
  void Process();

  // Finalize session by process remainder audio data of frontend, which
  // is not full chunk.
  void FinalizeSession();

  // Reset session.
  void Reset();

  std::string GetDecodedText() const;

 private:
  std::shared_ptr<RnntRsrc> rnnt_rsrc_;
  std::shared_ptr<SessionRsrc> session_rsrc_;

  std::string decoded_text_;
};

}  // namespace session
}  // namespace s2t

#endif
