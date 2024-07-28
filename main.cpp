#include <iostream>
#include <sstream>

#include "llama-wrapper.h"

int main() {
  LlamaWrapper llama;

  ModelParams model_params;
  model_params.n_gpu_layers = 32; // default on MacPro M1

  if (!llama.InitializeModel(
          "../models/Meta-Llama-3-8B-Instruct.Q5_1.gguf",
          model_params
      )) {
    std::cerr << "Failed to initialize the model." << std::endl;
    return 1;
  }

  ContextParams ctx_params;
  ctx_params.n_ctx = 2048;

  if (!llama.InitializeContext(ctx_params)) {
    std::cerr << "Failed to initialize the context." << std::endl;
    return 1;
  }

  SamplingParams sampling_params;
  sampling_params.temperature = 0.2f;
  sampling_params.max_tokens = 1000;
  sampling_params.frequency_penalty = 1.0f;
  sampling_params.repeat_penalty = 1.0f;

  std::string system_prompt =
      "You are a helpful assistant. "
      "Always provide clear, concise, and accurate answers. "
      "Avoid unnecessary details and focus on the main point. "
      "If you don't know the answer, simply say you don't know.";

  std::string user_msg = "How do I properly pass prompts to self managed llama 3 instructor LLM?";

  // Construct the prompt in the desired format
  // https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
  //  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  //  {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
  //  {{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  //  {{ model_answer_1 }}<|eot_id|>
  std::ostringstream prompt;

  prompt << "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
         << system_prompt << "<|eot_id|>"
         << "<|start_header_id|>user<|end_header_id|>"
         << user_msg << "<|eot_id|><|start_header_id|>assistant<|end_header_id|>";

  llama.RunQueryStream(
      prompt.str(),
      sampling_params,
      [](const std::string& piece) { std::cout << piece << std::flush; }
  );

  return 0;
}
