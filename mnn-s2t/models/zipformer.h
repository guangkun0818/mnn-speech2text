// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.23
// Streaming Zipformer Encoder.

#ifndef _MNN_S2T_MODEL_ZIPFORMER_H_
#define _MNN_S2T_MODEL_ZIPFORMER_H_

#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "mnn-s2t/models/encoder.h"

namespace s2t {
namespace models {

// Non streaming Zipformer.
// inputTensors : [ x, ...]
// outputTensors: [ encoder_out, ... ]
class MnnZipformer : public MnnEncoder {
 public:
  explicit MnnZipformer(const MnnEncoderCfg& cfg, mnn::ScheduleConfig config);

  ~MnnZipformer();

  const int ChunkSize() const;

  mnn::Session* Init(const int num_frames) override;

  void Reset(mnn::Session* session) override;

  void StreamingStep(const std::vector<std::vector<float>>& feats,
                     mnn::Session* session) override;

  void Inference(const std::vector<std::vector<float>>& feats,
                 mnn::Session* session) override;

  mnn::Tensor* GetEncOut(mnn::Session* session) override;

 private:
  void UpdateStates(mnn::Session* session);

  // Model resource.
  std::shared_ptr<mnn::Interpreter> model_;

  // Forward session.
  mnn::ScheduleConfig config_;

  int feat_dim_;
  int chunk_size_;
  int processed_lens_;
};

}  // namespace models
}  // namespace s2t

#endif