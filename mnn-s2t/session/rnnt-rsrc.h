// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.23
// Transducer Session.

#ifndef _MNN_S2T_SESSION_RNNT_RSRC_H_
#define _MNN_S2T_SESSION_RNNT_RSRC_H_

#include <memory>

#include "glog/logging.h"
#include "mnn-s2t/decoding/tokenizer.h"
#include "mnn-s2t/models/encoder.h"
#include "mnn-s2t/models/joiner.h"
#include "mnn-s2t/models/model-session.h"
#include "mnn-s2t/models/predictor.h"
#include "mnn-s2t/models/zipformer.h"

namespace s2t {
namespace session {

struct RnntRsrc {
  std::shared_ptr<models::MnnEncoder> encoder;
  std::shared_ptr<models::MnnPredictor> predictor;
  std::shared_ptr<models::MnnJoiner> joiner;
  std::shared_ptr<decoding::SubwordTokenizer> tokenizer;

  RnntRsrc(const models::MnnEncoderCfg& enc_cfg,
           const models::MnnPredictorCfg pred_cfg,
           const models::MnnJoinerCfg joiner_cfg, const char* units_file) {
    switch (enc_cfg.enc_type) {
      case models::EncoderType::kZipformer:
        LOG(INFO) << "Encoder: Zipformer selected.";
        encoder = std::make_shared<models::MnnZipformer>(
            enc_cfg, models::CPU_FORWARD_THREAD_8);
        break;
      default:
        LOG(WARNING) << "Unsupported encoder type.";
        break;
    }
    predictor = std::make_shared<models::MnnPredictor>(
        pred_cfg, models::CPU_FORWARD_THREAD_8);
    joiner = std::make_shared<models::MnnJoiner>(joiner_cfg,
                                                 models::CPU_FORWARD_THREAD_8);
    tokenizer = std::make_shared<decoding::SubwordTokenizer>(units_file);
  }
};

}  // namespace session
}  // namespace s2t

#endif