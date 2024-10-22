// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Subword Tokenizer.

#include "mnn-s2trt/decoding/tokenizer.h"

#include <regex>

namespace s2trt {
namespace decoding {

SubwordTokenizer::SubwordTokenizer(const char* units_file) {
  std::ifstream file(units_file);
  std::string line;
  while (std::getline(file, line)) {
    auto pos = line.find(' ', 0);  // Split line with space.
    auto token = line.substr(0, pos);
    auto token_idx = line.substr(pos + 1, line.size());
    vocab_.emplace(std::move(std::make_pair(std::stoi(token_idx), token)));
  }
}

std::string SubwordTokenizer::Decode(const std::vector<int>& token_idxs) {
  std::string decoded = "";
  for (auto idx : token_idxs) {
    auto found = vocab_.find(idx);
    CHECK(found != vocab_.end());
    decoded += found->second;
  }
  return this->RestoreSpace(decoded);
}

std::string SubwordTokenizer::RestoreSpace(const std::string& str) {
  // Replace "▁" in to space.
  std::string restored_str =
      std::regex_replace(str, std::regex(SPM_DELIMITER), " ");

  return Rtrim(Ltrim(restored_str));
}

std::string SubwordTokenizer::Ltrim(const std::string& str) {
  size_t start = str.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : str.substr(start);
}

std::string SubwordTokenizer::Rtrim(const std::string& str) {
  size_t end = str.find_last_not_of(WHITESPACE);
  return (end == std::string::npos) ? "" : str.substr(0, end + 1);
}

}  // namespace decoding
}  // namespace s2trt