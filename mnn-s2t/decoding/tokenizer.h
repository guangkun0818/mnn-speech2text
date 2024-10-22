// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Subword Tokenizer.

#ifndef _MNN_S2TRT_DECODING_TOKENIZER_H_
#define _MNN_S2TRT_DECODING_TOKENIZER_H_

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

namespace s2trt {
namespace decoding {

const char SPM_DELIMITER[] = "▁";
const char WHITESPACE[] = " \n\r\t\f\v";

class SubwordTokenizer {
 public:
  explicit SubwordTokenizer(const char* units_file);

  ~SubwordTokenizer(){};

  // Decode token_idxs into strings.
  std::string Decode(const std::vector<int>& token_idxs);

 private:
  // Restore origin text by replacing "▁" into space.
  std::string RestoreSpace(const std::string& str);

  std::string Ltrim(const std::string& str);

  std::string Rtrim(const std::string& str);

  std::unordered_map<int, std::string> vocab_;
};

}  // namespace decoding
}  // namespace s2trt

#endif
