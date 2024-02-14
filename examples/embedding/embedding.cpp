#include "common.h"
#include "llama.h"
#include <ctime>
#include <iostream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }
    std::cout << params.model << std::endl;
    std::cout << params.model_alias << std::endl;
    std::cout << params.numa << std::endl;

    params.embedding = true;

    print_build_info();

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }
    std::cout << "params.prompt: " << params.prompt << std::endl;

    llama_backend_init(params.numa);

    llama_model *model;
    llama_context *ctx;

    // load the model
    std::cout << "\nllama_init_from_gpt_params\n\n";
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    std::cout << "\nllama_init_from_gpt_params done\n\n";

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    std::cout << "n_ctx=" << n_ctx << ", n_ctx_train=" << n_ctx_train << std::endl;

    if (n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    std::cout << "\n\nprint system information:\n";
    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n\n", get_system_info(params).c_str());
    }

    int n_past = 0;

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, params.prompt, true);
    std::cout << "params.prompt: " << params.prompt << std::endl;
    std::cout << "params.verbose_prompt: " << params.verbose_prompt << std::endl;
    std::cout << "embd_inp.data(): " << embd_inp.data() << std::endl;
    std::cout << "\n\n";

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
        }
        fprintf(stderr, "\n");
    }

    if (embd_inp.size() > (size_t)n_ctx) {
        fprintf(stderr, "%s: error: prompt is longer than the context window (%zu tokens, n_ctx = %d)\n",
                __func__, embd_inp.size(), n_ctx);
        return 1;
    }

    while (!embd_inp.empty()) {
        int n_tokens = std::min(params.n_batch, (int) embd_inp.size());
        std::cout << "n_tokens=" << n_tokens << std::endl;
        if (llama_decode(ctx, llama_batch_get_one(embd_inp.data(), n_tokens, n_past, 0))) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }
        n_past += n_tokens;
        embd_inp.erase(embd_inp.begin(), embd_inp.begin() + n_tokens);
    }

    const int n_embd = llama_n_embd(model);
    const float *embeddings = llama_get_embeddings(ctx);

    std::cout << "\nn_embd=" << n_embd << ", print embedding:" << std::endl;
    for (int i = 0; i < n_embd; i++) {
        printf("%f ", embeddings[i]);
    }
    printf("\n");

    llama_print_timings(ctx);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
