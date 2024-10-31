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
  explicit MnnZipformer(const char* zipformer_model, const int feat_dim,
                        const int chunk_size);

  ~MnnZipformer();

  const int ChunkSize() const;

  void Init(const int num_frames) override;

  void Reset() override;

  void StreamingStep(const std::vector<std::vector<float>>& feats) override;

  void Inference(const std::vector<std::vector<float>>& feats) override;

  mnn::Tensor* GetEncOut() override;

 private:
  void UpdateStates();

  // Model resource.
  std::shared_ptr<mnn::Interpreter> model_;

  // Forward session.
  mnn::ScheduleConfig config_;
  mnn::Session* session_;

  int feat_dim_;
  int chunk_size_;
  int processed_lens_;
};

}  // namespace models
}  // namespace s2t

#endif