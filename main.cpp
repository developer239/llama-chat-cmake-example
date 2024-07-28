#include "llama-wrapper.h"
#include <iostream>

int main() {
  LlamaWrapper llama;

  ModelParams model_params;
  model_params.n_gpu_layers = 32; // default on MacPro M1

  if (!llama.InitializeModel("../models/Meta-Llama-3.1-8B-Instruct-Q3_K_S.gguf", model_params)) {
    std::cerr << "Failed to initialize the model." << std::endl;
    return 1;
  }

  ContextParams ctx_params;
  ctx_params.n_ctx = 2048;  // Set context size to 2048

  if (!llama.InitializeContext(ctx_params)) {
    std::cerr << "Failed to initialize the context." << std::endl;
    return 1;
  }

  SamplingParams sampling_params;
  sampling_params.temperature = 0.2f;
  sampling_params.max_tokens = 1000;
  sampling_params.frequency_penalty = 1.0f;
  sampling_params.repeat_penalty = 1.0f;

  llama.RunQueryStream("What is capital of France? Stop when you resolve the prompt.", sampling_params, [](const std::string& piece) {
    std::cout << piece << std::flush;
  });

  return 0;
}
