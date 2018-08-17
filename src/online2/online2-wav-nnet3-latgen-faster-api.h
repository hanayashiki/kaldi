#pragma once
#include <istream>
#include <iostream>
#include <string>
#include "fst/fstlib.h"
#include "online2/online-nnet3-decoding.h"
#include "feat/wave-reader.h"


namespace kaldi {

/* Results of Online2WavNnet3LatgenDecoder */
struct DecodingResult {
        bool good; /* Whether we got correct input and nothing's wrong during decoding */
        std::string speech_id; /* SpeechId from AsrService */
        std::string text; /* Final decoding result, if everything's correct  */
        std::string message; /* Accounts for why the decoding failed
                                or might be unreliable */
        double time_used; /* Time cost during the whole process */
public:
        std::string ToString() {
                return std::string("") +
                        "{\n"
                        "    " + "\"good\": " + std::to_string(good) + ",\n" +
                        "    " + "\"speech_id\": \"" + speech_id + "\",\n" +
                        "    " + "\"text\": \"" + text + "\",\n" +
                        "    " + "\"message\": \"" + message + "\",\n" +
                        "    " + "\"time_used\": " + std::to_string(time_used) + "\n" +
                        "}\n";
        }
};

/* Configuration for Online2WavNnet3LatgenDecoder */
struct Online2WavNnet3LatgenDecoderConfig {
        /* Length of chunk size in seconds, that we process.  Set to <= 0
         * to use all input in one chunk.
         * */
        float chunk_length_secs = -1.0;
        /* If true, apply endpoint detection
         * */
        bool do_endpointing = false;
        /* You can set this to false to disable online iVector estimation
         * and have all the data for each utterance used, even at
         * utterance start.  This is useful where you just want the best
         * results and don't care about online operation.
         * */
        bool online = false;
        /* Number of threads used when initializing iVector extractor. */
        int num_threads_startup = 4;

        std::string feature_type = "fbank";
        std::string config;
        bool add_pitch = false;
        int max_active = 7000;
        float beam = 10.0;
        float lattice_beam = 6.0;
        int frames_per_chunk = 50;
        float acoustic_scale = 1.0;

        std::string word_symbol_table;
        std::string nnet3_in;
        std::string fst_in;
        std::string lattice_wspecifier;
        float sample_freq = 16000.0;

};

class Online2WavNnet3LatgenDecoder {
//private:
public:
        /* just borrows pointers for large objs used by Online2WavNnet3LatgenDecoder */
        struct DecodingEnv {
                fst::Fst<fst::StdArc>* decode_fst;
                TransitionModel* trans_model;
                nnet3::AmNnetSimple* am_net;
                fst::SymbolTable * word_syms;
        };

        Online2WavNnet3LatgenDecoderConfig config;
        std::vector<const char *> parameters;
        char * copy_cstr(std::string & str) {
                char * c_str = new char[str.length() + 1];
                strncpy(c_str, str.c_str(), str.length() + 1);
                return c_str;
        }

        DecodingEnv env;

        void GetDiagnosticsAndPrintOutput(
                const std::string &utt,
                const fst::SymbolTable *word_syms,
                const CompactLattice &clat,
                int64 *tot_num_frames,
                double *tot_like,
                DecodingResult * result);
        bool Decode(int argc, char *argv[], DecodingResult * result,
                        std::string speech_id, std::istream & wave_stream);
        bool CheckWave(WaveHolder * wav, DecodingResult * result);
public:
        Online2WavNnet3LatgenDecoder(const Online2WavNnet3LatgenDecoderConfig & config);
        ~Online2WavNnet3LatgenDecoder();
        DecodingResult GetResult(std::string speech_id, std::istream & wave_stream);
};

}


 
