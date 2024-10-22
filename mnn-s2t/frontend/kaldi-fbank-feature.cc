// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.06
// This file is copied/modified from kaldi/src/feat/feature-fbank.cc

#include "mnn-s2trt/frontend/kaldi-fbank-feature.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "mnn-s2trt/frontend/feature-function.h"
#include "mnn-s2trt/frontend/kaldi-math.h"

namespace s2trt {
namespace frontend {

std::ostream &operator<<(std::ostream &os, const FbankOptions &opts) {
  os << opts.ToString();
  return os;
}

FbankComputer::FbankComputer(const FbankOptions &opts)
    : opts_(opts), rfft_(opts.frame_opts.PaddedWindowSize()) {
  if (opts.energy_floor > 0.0f) {
    log_energy_floor_ = logf(opts.energy_floor);
  }

  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]
  GetMelBanks(1.0f);
}

FbankComputer::~FbankComputer() {
  for (auto iter = mel_banks_.begin(); iter != mel_banks_.end(); ++iter)
    delete iter->second;
}

const MelBanks *FbankComputer::GetMelBanks(float vtln_warp) {
  MelBanks *this_mel_banks = nullptr;

  // std::map<float, MelBanks *>::iterator iter = mel_banks_.find(vtln_warp);
  auto iter = mel_banks_.find(vtln_warp);
  if (iter == mel_banks_.end()) {
    this_mel_banks = new MelBanks(opts_.mel_opts, opts_.frame_opts, vtln_warp);
    mel_banks_[vtln_warp] = this_mel_banks;
  } else {
    this_mel_banks = iter->second;
  }
  return this_mel_banks;
}

void FbankComputer::Compute(float signal_raw_log_energy, float vtln_warp,
                            std::vector<float> *signal_frame, float *feature) {
  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));

  CHECK_EQ(signal_frame->size(), opts_.frame_opts.PaddedWindowSize());

  // Compute energy after window function (not the raw one).
  if (opts_.use_energy && !opts_.raw_energy) {
    signal_raw_log_energy = std::log(
        std::max<float>(InnerProduct(signal_frame->data(), signal_frame->data(),
                                     signal_frame->size()),
                        std::numeric_limits<float>::epsilon()));
  }
  rfft_.Compute(signal_frame->data());  // signal_frame is modified in-place
  ComputePowerSpectrum(signal_frame);

  // Use magnitude instead of power if requested.
  if (!opts_.use_power) {
    Sqrt(signal_frame->data(), signal_frame->size() / 2 + 1);
  }

  int32_t mel_offset = ((opts_.use_energy && !opts_.htk_compat) ? 1 : 0);

  // Its length is opts_.mel_opts.num_bins
  float *mel_energies = feature + mel_offset;

  // Sum with mel filter banks over the power spectrum
  mel_banks.Compute(signal_frame->data(), mel_energies);

  if (opts_.use_log_fbank) {
    // Avoid log of zero (which should be prevented anyway by dithering).
    for (int32_t i = 0; i != opts_.mel_opts.num_bins; ++i) {
      auto t = std::max(mel_energies[i], std::numeric_limits<float>::epsilon());
      mel_energies[i] = std::log(t);
    }
  }

  // Copy energy as first value (or the last, if htk_compat == true).
  if (opts_.use_energy) {
    if (opts_.energy_floor > 0.0 && signal_raw_log_energy < log_energy_floor_) {
      signal_raw_log_energy = log_energy_floor_;
    }
    int32_t energy_index = opts_.htk_compat ? opts_.mel_opts.num_bins : 0;
    feature[energy_index] = signal_raw_log_energy;
  }
}

}  // namespace frontend
}  // namespace s2trt