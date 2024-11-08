// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.30
// Speech2text main with transducer system.

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "mnn-s2t/frontend/wav.h"
#include "mnn-s2t/session/rnnt-streaming-session.h"
#include "mnn-s2t/utils/json.h"
#include "mnn-s2t/utils/thread-pool.h"

DEFINE_string(rnnt_rsrc_conf, "configs/rnnt_rsrc_config.json",
              "Config of rnnt resource.");
DEFINE_string(session_conf, "configs/decoding_config.json",
              "Config of Rnnt Asr session.");
DEFINE_string(dataset_json, "runtime/config/test_data.json",
              "Dataset.json, Test dataset.");
DEFINE_int32(num_thread, 4, "Num of threads");

using Json = s2t::utils::json::JSON;

// Static transducer resource shared among all rnnt sessions.
static std::shared_ptr<s2t::session::RnntRsrc> rnnt_rsrc;
static s2t::models::MnnEncoderCfg enc_config;     // Encoder config
static s2t::models::MnnPredictorCfg pred_config;  // Predictor config
static s2t::models::MnnJoinerCfg joiner_config;   // Joiner config
static s2t::session::SessionCfg session_config;   // Session config

namespace {

void SetUpRnntEncoderConfig(Json& conf) {
  if (conf["type"].ToString() == "zipformer") {
    enc_config.enc_type = s2t::models::EncoderType::kZipformer;
    enc_config.chunk_size = conf["config"]["chunk_size"].ToInt();
    enc_config.feat_dim = conf["config"]["feats_dim"].ToInt();
    enc_config.encoder_model = conf["config"]["model_path"].ToString();
  } else {
    LOG(ERROR)
        << "Unsupported Encoder type, please check rnnt rsrc config setting.";
    std::abort();
  }
}

void SetUpRnntPredictorConfig(Json& conf) {
  pred_config.context_size = conf["context_size"].ToInt();
  pred_config.predictor_model = conf["model_path"].ToString();
}

void SetUpRnntJoinerConfig(Json& conf) {
  joiner_config.joiner_model = conf["model_path"].ToString();
}

}  // namespace

// Load Json from given json file.
void LoadJsonConf(const std::string& json_conf, /*output=*/Json& conf) {
  std::ifstream sess_conf_f(json_conf);
  std::string line, conf_infos;
  while (std::getline(sess_conf_f, line)) {
    conf_infos += line;
  }
  conf = Json::Load(conf_infos);
}

void BuildRnntRsrc(Json& conf) {
  CHECK(conf.hasKey("encoder"));
  CHECK(conf.hasKey("predictor"));
  CHECK(conf.hasKey("joiner"));
  CHECK(conf.hasKey("units_file"));

  SetUpRnntEncoderConfig(conf["encoder"]);
  SetUpRnntPredictorConfig(conf["predictor"]);
  SetUpRnntJoinerConfig(conf["joiner"]);

  rnnt_rsrc = std::make_shared<s2t::session::RnntRsrc>(
      enc_config, pred_config, joiner_config,
      conf["units_file"].ToString().c_str());
  LOG(INFO) << "Rnnt resource built.";
}

void SetUpSessionConfig(Json& conf) {
  CHECK(conf.hasKey("frontend"));
  CHECK(conf.hasKey("decoding"));

  session_config.feat_chunk_size = conf["frontend"]["feat_chunk_size"].ToInt();
  session_config.pcm_normalize = conf["frontend"]["pcm_normalize"].ToBool();

  if (conf["decoding"]["type"].ToString() == "rnnt-greedy-decoding") {
    session_config.decoding_cfg.decoding_type =
        s2t::decoding::DecodingType::kRnntGreedyDecoding;
    session_config.decoding_cfg.max_token_step =
        conf["decoding"]["config"]["max_token_step"].ToInt();
  } else {
    LOG(ERROR)
        << "Unsupported decoding type, please check session config setting.";
    std::abort();
  }
  LOG(INFO) << "Session config built.";
}

// Threading task by building each rnnt streaming session on every audio request
// flushed in.
void Speech2TextRnnt(const std::string& wave_file) {
  auto wave_reader = std::make_unique<s2t::frontend::WavReader>();
  auto session = std::make_unique<s2t::session::RnntStreamingSession>(
      rnnt_rsrc, session_config);
  LOG(INFO) << "Rnnt ASR Session built.";

  LOG(INFO) << "Request wave: " << wave_file;
  wave_reader->Open(wave_file);
  std::vector<float> pcm(wave_reader->data(),
                         wave_reader->data() + wave_reader->num_samples());
  // Session workflow
  session->AcceptWaves(pcm);
  session->Process();
  session->FinalizeSession();

  LOG(INFO) << "Decoded: " << session->GetDecodedText();
}

int main(int argc, char* argv[]) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Speech2text transducer starts...";

  Json rsrc_conf, sess_conf;
  LoadJsonConf(FLAGS_rnnt_rsrc_conf, rsrc_conf);
  LoadJsonConf(FLAGS_session_conf, sess_conf);
  BuildRnntRsrc(rsrc_conf);
  SetUpSessionConfig(sess_conf);

  // Create thread pool.
  s2t::utils::ThreadPool* thread_pool =
      new s2t::utils::ThreadPool(FLAGS_num_thread);

  std::ifstream datamap(FLAGS_dataset_json);
  std::string line;
  while (std::getline(datamap, line)) {
    auto test_entry = Json::Load(line);
    CHECK(test_entry.hasKey("audio_filepath"));

    thread_pool->enqueue(Speech2TextRnnt,
                         test_entry["audio_filepath"].ToString());
  }

  delete thread_pool;
  LOG(INFO) << "Done.";

  return 0;
}