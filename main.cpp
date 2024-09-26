#include <iostream>
#include "llama-chat.h"

int main() {
  LlamaChat llama;

  ModelParams modelParams;
  modelParams.nGpuLayers = 32;

  if (!llama.InitializeModel("../models/Meta-Llama-3.1-8B-Instruct-Q3_K_S.gguf", modelParams)) {
    std::cerr << "Failed to initialize the model." << std::endl;
    return 1;
  }

  ContextParams ctxParams;
  ctxParams.nContext = 2048;

  if (!llama.InitializeContext(ctxParams)) {
    std::cerr << "Failed to initialize the context." << std::endl;
    return 1;
  }

  std::string systemPrompt = "You are a helpful assistant. Always provide clear, concise, and accurate answers. "
      "Avoid unnecessary details and focus on the main point. "
      "If you don't know the answer, simply say you don't know.";
  llama.SetSystemPrompt(systemPrompt);

  std::vector<std::string> questions = {
      "What is the capital of France?",
      "What's the population of that city?",
      "What country is the city in?"
  };

  for (const auto& question : questions) {
    std::cout << "User: " << question << std::endl;
    std::cout << "Assistant: ";
    llama.Prompt(question, [](const std::string& piece) {
      std::cout << piece << std::flush;
    });
    std::cout << std::endl << std::endl;
  }

  return 0;
}
