# Author: guangkun0818 Email: 609946862@qq.com Created on 2024.10.31

# Build offline vad session
add_executable(s2t-transducer speech2text-rnnt.cc)

target_link_libraries(
  s2t-transducer
  frontend_shared
  encoder_shared
  predictor_shared
  joiner_shared
  decoding_shared
  glog
  gflags)
