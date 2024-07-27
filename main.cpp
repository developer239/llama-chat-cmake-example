#include "llama-wrapper.h"
#include <iostream>

int main() {
  LlamaWrapper llama;
  if (!llama.Initialize("../models/Meta-Llama-3.1-8B-Instruct-Q3_K_S.gguf", 1000)) {
    std::cerr << "Failed to initialize the model." << std::endl;
    return 1;
  }

  std::string response = llama.RunQuery("Tell me a story.", 100);
  std::cout << "Response: " << response << std::endl;

  return 0;
}
