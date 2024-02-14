#include "common.h"
#include "llama.h"
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <signal.h>
#include <unistd.h>


struct ostream_beam_view {
    llama_context *ctx;
    llama_beam_view beam_view;
};

static std::ostream & operator<<(std::ostream & os, const ostream_beam_view & obv) {
    os << "p(" << obv.beam_view.p << ") eob(" << std::boolalpha << obv.beam_view.eob << ") tokens(";
    for (size_t i = 0 ; i < obv.beam_view.n_tokens ; ++i) {
        os << llama_token_to_piece(obv.ctx, obv.beam_view.tokens[i]);
    }
    return os << ')';
}

struct beam_search_callback_data {
    llama_context *ctx;
    std::vector<llama_token> response;
};

static bool is_at_eob(const beam_search_callback_data &callback_data, const llama_token *tokens, size_t n_tokens) {
    return n_tokens && tokens[n_tokens-1] == llama_token_eos(llama_get_model(callback_data.ctx));
}

static void beam_search_callback(void * callback_data_ptr, llama_beams_state beams_state) {
    auto& callback_data = *static_cast<beam_search_callback_data*>(callback_data_ptr);
    for (size_t i = 0 ; i < beams_state.n_beams ; ++i) {
        llama_beam_view& beam_view = beams_state.beam_views[i];
        if (!beam_view.eob && is_at_eob(callback_data, beam_view.tokens, beam_view.n_tokens)) {
            beam_view.eob = true;
        }
    }
    printf(",");  // Show progress
    if (const size_t n = beams_state.common_prefix_length) {
        callback_data.response.resize(callback_data.response.size() + n);
        assert(0u < beams_state.n_beams);
        const llama_token * tokens = beams_state.beam_views[0].tokens;
        std::copy(tokens, tokens + n, callback_data.response.end() - n);
        printf("%zu", n);
    }
    fflush(stdout);
#if 1 // DEBUG: print current beams for this iteration
    std::cout << "\n\nCurrent beams (last_call=" << beams_state.last_call << "):\n";
    for (size_t i = 0 ; i < beams_state.n_beams ; ++i) {
        std::cout << "beams[" << i << "]: " << ostream_beam_view{callback_data.ctx,beams_state.beam_views[i]} << std::endl;
    }
#endif
}


int main(int argc, char ** argv) {
    if ( argc < 2 || argv[1][0] == '-' ) {
        printf( "Usage: %s MODEL_PATH [BEAM_WIDTH=2] [PROMPT]\n" , argv[0] );
        return 1 ;
    }
    gpt_params params;
    params.model = argv[1];
    params.n_beams = 2 < argc ? std::stoi(argv[2]) : 2;

    if ( argc > 3 ) {
        params.prompt = argv[3];
    }
    if ( params.prompt.empty() ) {
        params.prompt = "### Request:\nHow many countries are there?\n\n### Response:\n";
    }

    llama_backend_init(params.numa);
    llama_model *model;
    llama_context *ctx;
    std::tie(model, ctx) = llama_init_from_gpt_params( params );

    if ( model == NULL ) {
        fprintf( stderr , "%s: error: unable to load model\n" , __func__ );
        return 1;
    }

    std::vector<llama_token> tokens_list = llama_tokenize(ctx, params.prompt, true);
    const size_t max_context_size     = llama_n_ctx( ctx );
    const size_t max_tokens_list_size = max_context_size - 4 ;

    //bm
    std::cout << "\nmax_context_size = " << max_context_size << std::endl;
    std::cout << "max_tokens_list_size = " << max_tokens_list_size << std::endl;
    std::cout << "tokens_list.size() = " << tokens_list.size() << std::endl;
    for(size_t i = 0; i < tokens_list.size(); ++i) {
        std::cout << tokens_list[i] << " ";
    }
    std::cout << "\n";


    if (tokens_list.size() > max_tokens_list_size) {
        fprintf( stderr , "%s: error: prompt too long (%zu tokens, max %zu)\n" ,
             __func__ , tokens_list.size() , max_tokens_list_size );
        return 1;
    }

    fprintf( stderr, "\n\n" );

    // Print the tokens from the prompt :
    for( auto id : tokens_list ) {
        std::cout << llama_token_to_piece(ctx, id);
    }
    std::cout << std::flush;

    int n_past = 0;
    if (llama_decode(ctx, llama_batch_get_one(tokens_list.data(), tokens_list.size(), n_past, 0))) {
        fprintf(stderr, "%s : failed to eval prompt.\n" , __func__ );
        return 1;
    }
    n_past += tokens_list.size();
    //bm
    std::cout << "n_past = " << n_past << std::endl;
    for(size_t i = 0; i < tokens_list.size(); ++i) {
        std::cout << tokens_list[i] << " ";
    }
    std::cout << "\n";

    {
        beam_search_callback_data callback_data{ctx, {}};
        size_t const beam_width = static_cast<size_t>(params.n_beams);
        int const n_predict = 256;

        //计算量大
        std::cout << "llama_beam_search\n";
        llama_beam_search(ctx, beam_search_callback, &callback_data, beam_width, n_past, n_predict);
        std::cout << "llama_beam_search done\n";

        std::cout << "\n\n";
        for (llama_token const token_id : callback_data.response) {
            std::cout << llama_token_to_piece(ctx, token_id);
        }
        std::cout << std::endl;
    }

    llama_free( ctx );
    llama_free_model( model );
    llama_backend_free();

    return 0;
}
