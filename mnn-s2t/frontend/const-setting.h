// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.10
// const setting for frontend, which would not changes in most cases.

#ifndef _MNN_S2TRT_FRONTEND_CONST_SETTING_H_
#define _MNN_S2TRT_FRONTEND_CONST_SETTING_H_

#include "mnn-s2t/frontend/kaldi-fbank-feature.h"

namespace s2t {
namespace frontend {

const float NORMALIZE_FACTOR = 1.0f / 32768;  // For 16k 16bit wavs.

// Const Setting refered to LhotseKaldiFeatFbank()
// https://github.com/guangkun0818/speech2text/blob/main/dataset/frontend/frontend.py
FbankOptions LHOTSE_FBANK_OPTIONS() {
  FbankOptions FBANK_OPTIONS;

  FBANK_OPTIONS.mel_opts.num_bins = 80;        // 80 dim fbank.
  FBANK_OPTIONS.mel_opts.low_freq = 20.0f;     // Default setting in lhotes
  FBANK_OPTIONS.mel_opts.high_freq = -400.0f;  // Default setting in lhotes

  FBANK_OPTIONS.frame_opts.dither = 0.0f;
  FBANK_OPTIONS.frame_opts.snip_edges =
      true;  // Different with default setting of lhotes.
  FBANK_OPTIONS.energy_floor = 1e-10f;  // EPSILON = 1e-10 in lhotes.
  return FBANK_OPTIONS;
}

}  // namespace frontend
}  // namespace s2t

#endif
