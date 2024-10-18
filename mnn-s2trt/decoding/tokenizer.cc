// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Subword Tokenizer.

#include "mnn-s2trt/decoding/tokenizer.h"

#include <regex>

namespace s2trt {
namespace decoding {

SubwordTokenzier::SubwordTokenzier(const char* units_file) {
  std::ifstream file(units_file);
  std::string line;
  while (std::getline(file, line)) {
    auto pos = line.find(' ', 0);  // Split line with space.
    auto token = line.substr(0, pos);
    auto token_idx = line.substr(pos + 1, line.size());
    vocab_.emplace(std::move(std::make_pair(std::stoi(token_idx), token)));
  }
}

std::string SubwordTokenzier::Decode(const std::vector<int>& token_idxs) {
  std::string decoded = "";
  for (auto idx : token_idxs) {
    auto found = vocab_.find(idx);
    CHECK(found != vocab_.end());
    decoded += found->second;
  }
  return this->RestoreSpace(decoded);
}

std::string SubwordTokenzier::RestoreSpace(const std::string& str) {
  // Replace "▁" in to space.
  std::string restored_str =
      std::regex_replace(str, std::regex(SUBWORD_SPACE), " ");
  // String strip space.
  restored_str.erase(0, restored_str.find_first_not_of(" "));
  restored_str.erase(restored_str.find_last_not_of(" ") + 1);

  return restored_str;
}

}  // namespace decoding
}  // namespace s2trt