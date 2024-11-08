# `speech2text` mnn inference runtime

This repo offer a simple fast `MNN` inference runtime of [speech2text](https://github.com/guangkun0818/speech2text). Support `zipfromer stateless transducer` streaming ASR system inference currently，more system will be supported in the future (maybe ╮（╯＿╰）╭). ***Feel free to reach me 609946862@qq.com or issue if any problem encountered!***

## build and run
```bash
bash build.sh
```
Run inference with `thread_pool` like below. Please check configs from `configs/` and `sample_data/` for inference setting. 
```bash
./build/mnn-s2t/bin/speech2text-rnnt \
    --rnnt_rsrc_conf=configs/rnnt_rsrc_config.json \
    --session_conf=configs/session_config.json \
    --dataset_json=sample_data/test_data.json \
    --num_thread=4
```