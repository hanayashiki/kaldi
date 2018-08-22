// online2bin/online2-wav-nnet3-latgen-faster-api.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include "online2/online2-wav-nnet3-latgen-faster-api.h"
#include <ctime>

namespace kaldi {

void Online2WavNnet3LatgenDecoder::GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like,
                                  DecodingResult * result) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    result->good = false;
    result->message += "Empty lattice\n";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames.";
  result->text = "";
  if (word_syms != NULL) {
    std::cerr << utt << ' ';
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "") {
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
        result->message +=
          "Word-id " + std::to_string(words[i]) + " not in symbol table\n";
        result->good = false;
      }
      result->text += s;
      std::cerr << s << ' ';
    }
    std::cerr << std::endl;
  } else {
    result->good = false;
    result->message += "We got word_syms == nullptr\n";
  }
}

bool Online2WavNnet3LatgenDecoder::Decode(
    int argc, char *argv[], DecodingResult * result,
    std::string speech_id, std::istream & wave_stream) {

  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding with neural nets\n"
        "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
        "optional endpointing.  Note: some configuration values and inputs are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
        "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to decode utterance by utterance.\n";

    ParseOptions po(usage);

    std::string word_syms_rxfilename;

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    BaseFloat chunk_length_secs = config.chunk_length_secs;
    bool do_endpointing = config.do_endpointing;
    bool online = config.online;
    g_num_threads = config.num_threads_startup;

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);


    po.Read(argc, argv);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);


    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

    clock_t start = clock();
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        env.am_net);

    fst::Fst<fst::StdArc> *decode_fst = env.decode_fst;
    // fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);
    KALDI_LOG << "ReadFstKaldiGeneric used " << (clock() - start) / CLOCKS_PER_SEC;

    fst::SymbolTable * word_syms = env.word_syms;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;

    // This is modified to pass only one speech

    WaveHolder wave_holder;
    if (!wave_holder.Read(wave_stream)) {
        result->good = false;
        result->message += "the wav file is broken or not supported\n";
        throw std::runtime_error("bad wave");
    }

    if (!CheckWave(&wave_holder, result)) {
        throw std::runtime_error("bad wave attribute");
    }


    CompactLatticeWriter clat_writer(config.lattice_wspecifier);

    OnlineTimingStats timing_stats;

    OnlineIvectorExtractorAdaptationState adaptation_state(
        feature_info.ivector_extractor_info);
    const WaveData &wave_data = wave_holder.Value();
    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    SubVector<BaseFloat> data(wave_data.Data(), 0);

    OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
    feature_pipeline.SetAdaptationState(adaptation_state);

    OnlineSilenceWeighting silence_weighting(
        *(env.trans_model),
        feature_info.silence_weighting_config,
        decodable_opts.frame_subsampling_factor);

    SingleUtteranceNnet3Decoder decoder(decoder_opts, *(env.trans_model),
                                            decodable_info,
                                            *decode_fst, &feature_pipeline);
    OnlineTimer decoding_timer(speech_id);

    BaseFloat samp_freq = wave_data.SampFreq();
    int32 chunk_length;
    if (chunk_length_secs > 0) {
        chunk_length = int32(samp_freq * chunk_length_secs);
    if (chunk_length == 0) chunk_length = 1;
    } else {
        chunk_length = std::numeric_limits<int32>::max();
    }

    int32 samp_offset = 0;
    std::vector<std::pair<int32, BaseFloat> > delta_weights;

    start = clock();

    while (samp_offset < data.Dim()) {
        int32 samp_remaining = data.Dim() - samp_offset;
        int32 num_samp = chunk_length < samp_remaining ? chunk_length : samp_remaining;

        SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
        feature_pipeline.AcceptWaveform(samp_freq, wave_part);

        samp_offset += num_samp;
        decoding_timer.WaitUntil(samp_offset / samp_freq);
        if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
        }

        if (silence_weighting.Active() &&
            feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                              &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
        }
        clock_t start = clock();
        decoder.AdvanceDecoding();
		KALDI_LOG << "decoder.AdvanceDecoding used: " << 1.0 * (clock() - start) / CLOCKS_PER_SEC;

        if (do_endpointing && decoder.EndpointDetected(endpoint_opts)) {
            break;
        }
    }
    decoder.FinalizeDecoding();

    KALDI_LOG << "decoder used " << 1.0 * (clock() - start) / CLOCKS_PER_SEC;

    CompactLattice clat;
    bool end_of_utterance = true;
    decoder.GetLattice(end_of_utterance, &clat);

    GetDiagnosticsAndPrintOutput(speech_id, word_syms, clat,
                                     &num_frames, &tot_like, result);

    decoding_timer.OutputStats(&timing_stats);

    // In an application you might avoid updating the adaptation state if
    // you felt the utterance had low confidence.  See lat/confidence.h
    feature_pipeline.GetAdaptationState(&adaptation_state);

    // we want to output the lattice with un-scaled acoustics.
    BaseFloat inv_acoustic_scale =
            1.0 / decodable_opts.acoustic_scale;
    ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);

    clat_writer.Write(speech_id, clat);

    KALDI_LOG << "Decoded utterance " << speech_id;
    num_done++;

    timing_stats.Print(online);

    KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";

    KALDI_LOG << "Final result: " << result->text;

    return (num_done != 0 ? true : false);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    result->message += std::string(e.what()) + "\n";
    result->good = false;
    std::cerr << result->message << std::endl;
    return false;
  }

}

bool Online2WavNnet3LatgenDecoder::CheckWave(WaveHolder * wav, DecodingResult * result) {
  float sample_freq = wav->Value().SampFreq();
  if (std::abs(sample_freq - config.sample_freq) > 0.01) {
    result->good = false;
    result->message += "sample_freq is not matching required " + std::to_string(config.sample_freq)
            + ", get " + std::to_string(sample_freq) + "\n";
    return false;
  }
  return true;
}

Online2WavNnet3LatgenDecoder::Online2WavNnet3LatgenDecoder
  (const Online2WavNnet3LatgenDecoderConfig & config)
  : config(config) {
  std::string this_name = "Online2WavNnet3LatgenDecoder"; // new to delete faultlessly
  parameters.push_back(copy_cstr(this_name));
  std::string feature_type = "--feature_type=" + config.feature_type;
  parameters.push_back(copy_cstr(feature_type));
  std::string config_ = "--config=" + config.config;
  parameters.push_back(copy_cstr(config_));
  std::string max_active = "--max-active=" + std::to_string(config.max_active);
  parameters.push_back(copy_cstr(max_active));
  std::string beam = "--beam=" + std::to_string(config.beam);
  parameters.push_back(copy_cstr(beam));
  std::string lattice_beam = "--lattice-beam=" + std::to_string(config.lattice_beam);
  parameters.push_back(copy_cstr(lattice_beam));
  std::string frames_per_trunk = "--frames-per-chunk=" +
    std::to_string(config.frames_per_chunk);
  parameters.push_back(copy_cstr(frames_per_trunk));
  std::string acoustic_scale = "--acoustic-scale=" + std::to_string(config.acoustic_scale);
  parameters.push_back(copy_cstr(acoustic_scale));
  parameters.push_back("frame-subsampling-factor=3");


  KALDI_LOG << "Loading fst::Fst<fst::StdArc> decode_fst from " << config.fst_in;
  env.decode_fst = fst::ReadFstKaldiGeneric(config.fst_in);
  KALDI_LOG << "Loading nnet3::AmNnetSimple from" << config.nnet3_in;

  TransitionModel * trans_model = new TransitionModel();
  nnet3::AmNnetSimple * am_nnet = new nnet3::AmNnetSimple();

  {
    bool binary;
    kaldi::Input ki(config.nnet3_in, &binary);
    trans_model->Read(ki.Stream(), binary);
    am_nnet->Read(ki.Stream(), binary);
    SetBatchnormTestMode(true, &(am_nnet->GetNnet()));
    SetDropoutTestMode(true, &(am_nnet->GetNnet()));
    kaldi::nnet3::CollapseModel(kaldi::nnet3::CollapseModelConfig(),
    &(am_nnet->GetNnet()));
  }

  env.am_net = am_nnet;
  env.trans_model = trans_model;

  KALDI_LOG << "Loading fst::SymbolTable for word_syms from " << config.word_symbol_table;
  env.word_syms = fst::SymbolTable::ReadText(config.word_symbol_table);

}

Online2WavNnet3LatgenDecoder::~Online2WavNnet3LatgenDecoder() {
  delete env.decode_fst;
  delete env.am_net;
  delete env.trans_model;
  delete env.word_syms;
  for (auto a : parameters) {
    delete []a;
  }
}

DecodingResult Online2WavNnet3LatgenDecoder::GetResult(
	std::string speech_id, std::istream & wave_stream) {
  clock_t decode_start = clock();

  DecodingResult result;
  result.speech_id = speech_id;

  int argc = parameters.size();
  char ** argv = (char **)parameters.data();

  if (!Decode(argc, argv, &result, speech_id, wave_stream)) {
    KALDI_LOG << "GetResult: Something is wrong with speech " << speech_id;
    result.good = false;
  } else {
	result.good = true;
  }

  result.time_used = 1.0 * (clock() - decode_start) / CLOCKS_PER_SEC; 

  return result;

}	

}
    
