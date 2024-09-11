// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.10
// Frontend pipeline impl of speech2text session.

#ifndef _MNN_S2TRT_FRONTEND_PIPELINE_H_
#define _MNN_S2TRT_FRONTEND_PIPELINE_H_

#include <deque>
#include <memory>
#include <vector>

#include "mnn-s2trt/frontend/kaldi-fbank-feature.h"

namespace s2trt {
namespace frontend {

const float NORMALIZE_FACTOR = 1 / 32768;

// Non streaming frontend pipeline.
class Frontend {
 public:
  // If True, normalize pcm with 32768.
  explicit Frontend(const FbankOptions& opts, const bool pcm_normalize = true);

  // Explictly disable copyable and movable.
  Frontend(const Frontend&) = delete;
  Frontend& operator=(const Frontend&) = delete;
  Frontend(const Frontend&&) = delete;
  Frontend& operator=(const Frontend&&) = delete;

  virtual ~Frontend(){};

  // Reset Non streaming frontend pipeline.
  void Reset() { pcms_ready_.clear(); }

  // Interface to accept flushed in pcms.
  virtual void AcceptPcms(const std::vector<float>& pcms);

  // Interface to extract feats.
  virtual void EmitFeats(std::vector<std::vector<float>>& feats,
                         bool is_last = true);

 protected:
  bool pcm_normalize_;
  FbankOptions opts_;
  FeatureWindowFunction window_function_;

  std::shared_ptr<FbankComputer> feat_computer_;
  std::vector<float> pcms_ready_;
};

class StreamingFrontend : public Frontend {
 public:
  explicit StreamingFrontend(const FbankOptions& opts, int32_t feat_chunk_size,
                             const bool pcm_normalize = true);

  // Explictly disable copyable and movable.
  StreamingFrontend(const StreamingFrontend&) = delete;
  StreamingFrontend& operator=(const StreamingFrontend&) = delete;
  StreamingFrontend(const StreamingFrontend&&) = delete;
  StreamingFrontend& operator=(const StreamingFrontend&&) = delete;

  virtual ~StreamingFrontend(){};

  // Reset Non streaming frontend pipeline.
  void Reset();

  // Specify whether pending pcm is enough to emit feats chunk.
  bool IsReady() const;

  // Interface to accept flushed in pcms.
  virtual void AcceptPcms(const std::vector<float>& pcms);

  // Interface to extract feats.
  virtual void EmitFeats(std::vector<std::vector<float>>& feats,
                         bool is_last = false);

 private:
  // Prepare pcm_to_extract from pending pcm.
  void PreparePcms();

  int32_t feat_chunk_size_;  // Chunk size of feature frame
  int32_t pcm_chunk_size_;   // Chunk size of pcm samples.
  int32_t pcm_cache_size_;   // Cache size of pcm samples after every emit.

  std::deque<std::vector<float>> pcms_pending_;
  int32_t num_pending_pcms_;
  int32_t start_offset_of_front_;  // Offset of front pcms slice over queue.

  std::vector<float> last_pcm_cache_;  // Cache for next chunk.
};

}  // namespace frontend
}  // namespace s2trt

#endif