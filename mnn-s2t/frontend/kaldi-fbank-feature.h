// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.06
// This file is copied/modified from kaldi/src/feat/feature-fbank.h

#ifndef _MNN_S2TRT_FEATURE_FBANK_H_
#define _MNN_S2TRT_FEATURE_FBANK_H_

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mnn-s2t/frontend/feature-window.h"
#include "mnn-s2t/frontend/mel-computations.h"
#include "mnn-s2t/frontend/rfft.h"

namespace s2t {
namespace frontend {

struct FbankOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;
  // append an extra dimension with energy to the filter banks
  bool use_energy = false;
  float energy_floor = 0.0f;  // active iff use_energy==true

  // If true, compute log_energy before preemphasis and windowing
  // If false, compute log_energy after preemphasis ans windowing
  bool raw_energy = true;  // active iff use_energy==true

  // If true, put energy last (if using energy)
  // If false, put energy first
  bool htk_compat = false;  // active iff use_energy==true

  // if true (default), produce log-filterbank, else linear
  bool use_log_fbank = true;

  // if true (default), use power in filterbank
  // analysis, else magnitude.
  bool use_power = true;

  FbankOptions() { mel_opts.num_bins = 23; }

  std::string ToString() const {
    std::ostringstream os;
    os << "frame_opts: \n";
    os << frame_opts << "\n";
    os << "\n";

    os << "mel_opts: \n";
    os << mel_opts << "\n";

    os << "use_energy: " << use_energy << "\n";
    os << "energy_floor: " << energy_floor << "\n";
    os << "raw_energy: " << raw_energy << "\n";
    os << "htk_compat: " << htk_compat << "\n";
    os << "use_log_fbank: " << use_log_fbank << "\n";
    os << "use_power: " << use_power << "\n";
    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const FbankOptions &opts);

class FbankComputer {
 public:
  using Options = FbankOptions;

  explicit FbankComputer(const FbankOptions &opts);
  ~FbankComputer();

  int32_t Dim() const {
    return opts_.mel_opts.num_bins + (opts_.use_energy ? 1 : 0);
  }

  // if true, compute log_energy_pre_window but after dithering and dc removal
  bool NeedRawLogEnergy() const { return opts_.use_energy && opts_.raw_energy; }

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  const FbankOptions &GetOptions() const { return opts_; }

  /**
     Function that computes one frame of features from
     one frame of signal.

     @param [in] signal_raw_log_energy The log-energy of the frame of the signal
         prior to windowing and pre-emphasis, or
         log(numeric_limits<float>::min()), whichever is greater.  Must be
         ignored by this function if this class returns false from
         this->NeedsRawLogEnergy().
     @param [in] vtln_warp  The VTLN warping factor that the user wants
         to be applied when computing features for this utterance.  Will
         normally be 1.0, meaning no warping is to be done.  The value will
         be ignored for feature types that don't support VLTN, such as
         spectrogram features.
     @param [in] signal_frame  One frame of the signal,
       as extracted using the function ExtractWindow() using the options
       returned by this->GetFrameOptions().  The function will use the
       vector as a workspace, which is why it's a non-const pointer.
     @param [out] feature  Pointer to a vector of size this->Dim(), to which
         the computed feature will be written. It should be pre-allocated.
  */
  void Compute(float signal_raw_log_energy, float vtln_warp,
               std::vector<float> *signal_frame, float *feature);

 private:
  const MelBanks *GetMelBanks(float vtln_warp);

  FbankOptions opts_;
  float log_energy_floor_;
  std::map<float, MelBanks *> mel_banks_;  // float is VTLN coefficient.
  Rfft rfft_;
};

}  // namespace frontend
}  // namespace s2t

#endif  // _MNN_S2TRT_FEATURE_FBANK_H_
