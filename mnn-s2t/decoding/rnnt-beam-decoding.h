// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.11.08
// Beam decoding of Transducer.

#ifndef _MNN_S2T_DECODING_RNNT_BEAM_DECODING_H_
#define _MNN_S2T_DECODING_RNNT_BEAM_DECODING_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "mnn-s2t/decoding/decoding.h"
#include "mnn-s2t/decoding/tokenizer.h"
#include "mnn-s2t/models/joiner.h"
#include "mnn-s2t/models/model-session.h"
#include "mnn-s2t/models/predictor.h"

namespace s2t {
namespace decoding {

float LogAdd(float x, float y) { return std::log(std::exp(x) + std::exp(y)); }

struct DecodingBeam {
  std::vector<int> decoded_tokens;
  bool end_with_blank;
  float score;
  mnn::Tensor* pred_out = nullptr;
  mnn::Tensor* pred_state = nullptr;

  DecodingBeam(const std::vector<int>& updated_decoded_tokens,
               bool updated_end_with_blank, float updated_score,
               const mnn::Tensor* last_pred_out,
               const mnn::Tensor* last_pred_state);

  ~DecodingBeam();  // Release pred_state/pred_out tensor.
};

struct RnntBeamDecodingStates {
  std::vector<std::shared_ptr<DecodingBeam>>
      beams;  // where beams[0] is best beam.
};

class RnntBeamDecoding : public DecodingMethod {
 public:
  explicit RnntBeamDecoding(
      const std::shared_ptr<models::MnnPredictor>& predictor,
      const std::shared_ptr<models::MnnJoiner>& joiner,
      const std::shared_ptr<models::RnntModelSession>& model_sess,
      const std::shared_ptr<SubwordTokenizer>& tokenizer,
      const DecodingCfg& cfg);  // (TODO) support nnlm.

  void Init() override;

  void Reset() override;

  inline bool IsBlank(int token) const;

  void ArgSort(const std::vector<float>& log_probs,
               std::vector<size_t>& args) const;

  void Decode(mnn::Tensor* enc_out) override;

  std::string GetResults() override;

 private:
  void BuildBeamPredOut(mnn::Tensor* beamed_pred_out);

  void UpdateBeams(const std::vector<std::vector<float>>& log_probs,
                   bool is_start = false);

  void ResetDecodingStates();

  std::shared_ptr<models::MnnPredictor> predictor_;
  std::shared_ptr<models::MnnJoiner> joiner_;
  std::shared_ptr<models::RnntModelSession> model_sess_;
  std::shared_ptr<SubwordTokenizer> tokenizer_;
  std::vector<DecodingBeam*> beams_;

  int beam_size_;
  int cutoff_top_k_;
  bool on_start_ = true;
};

}  // namespace decoding
}  // namespace s2t

#endif