// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Unittest of Subword Tokenizer.

#include "mnn-s2trt/decoding/tokenizer.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace s2trt::decoding;

class TestTokenizer : public ::testing::Test {
 protected:
  void SetUp() {
    tokenizer_ = std::make_shared<SubwordTokenzier>("sample_data/units.txt");
  }
  std::shared_ptr<SubwordTokenzier> tokenizer_;
};

TEST_F(TestTokenizer, TestTokenizerDecode) {
  std::string decoded;
  decoded = tokenizer_->Decode(
      {63, 91, 20,  20, 47, 19, 9, 10, 66, 19, 2,  5,  99, 7,  94, 16, 8,  10,
       50, 15, 126, 31, 14, 4,  3, 88, 21, 2,  45, 27, 26, 19, 2,  89, 29, 6});
  ASSERT_STREQ(
      decoded.c_str(),
      "struggle warfare was the condition of private ownership it was fatal");

  decoded = tokenizer_->Decode({63, 56, 2,  16, 17, 15, 42, 35, 6,  14, 18,
                                7,  54, 4,  27, 27, 21, 18, 7,  11, 3,  6,
                                57, 14, 47, 63, 74, 93, 32, 3,  29, 112});
  ASSERT_STREQ(decoded.c_str(),
               "season with salt and pepper and a little sugar to taste");
  decoded =
      tokenizer_->Decode({30, 3,  7,   16, 23, 14, 33,  6,  15, 4,  22, 18, 12,
                          3,  39, 108, 38, 18, 13, 107, 20, 15, 59, 8,  9,  77,
                          23, 7,  49,  22, 81, 39, 34,  26, 33, 87, 4});
  ASSERT_STREQ(
      decoded.c_str(),
      "i don't believe ann knew any magic or she'd have worked it before");

  decoded = tokenizer_->Decode({11, 72, 6,  31, 61, 96,  95, 75, 62, 4,
                                79, 8,  22, 9,  5,  103, 6,  6,  33, 45,
                                40, 5,  17, 34, 7,  36,  20, 78, 13});
  ASSERT_STREQ(decoded.c_str(),
               "a black drove came up over the hill behind the wedding party");
}