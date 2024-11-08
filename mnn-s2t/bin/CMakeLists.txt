# Author: guangkun0818 Email: 609946862@qq.com Created on 2024.10.31

# Build offline vad session
add_executable(speech2text-rnnt speech2text-rnnt.cc)

target_link_libraries(speech2text-rnnt rnnt_session glog gflags)
