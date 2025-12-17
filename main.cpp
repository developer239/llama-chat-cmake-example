#include <iostream>

#include "llama-chat.h"

int main() {
  const auto IMAGE_PATH = "../images/kapybara.jpg";

  LlamaChat llama;

  ModelParams modelParams;
  modelParams.gpuLayerCount = 32;
  modelParams.multiModalProjectorPath = "../models/mmproj-model-f16.gguf";
  modelParams.offloadMultiModalToGPU = true;

  if (!llama.InitializeModel("../models/gemma-3-4b-it-f16.gguf", modelParams)) {
    std::cerr << "Failed to initialize the model." << std::endl;
    return 1;
  }

  ContextParams contextParams;
  contextParams.contextSize = 4096;
  contextParams.batchSize = 512;

  if (!llama.InitializeContext(contextParams)) {
    std::cerr << "Failed to initialize the context." << std::endl;
    return 1;
  }

  llama.SetSystemPrompt(
      "You are a helpful assistant that can analyze images and answer "
      "questions about them. "
      "Provide clear, to the point descriptions when asked about images."
  );

  std::cout << "\n=== Example 1: Image Description ===" << std::endl;
  std::cout << "User: What do you see in this image?" << std::endl;
  std::cout << "Assistant: ";

  ImageInput image = ImageInput::FromPath(IMAGE_PATH);
  llama.Prompt(
      "What do you see in this image?",
      [](const std::string& piece) { std::cout << piece << std::flush; },
      image
  );
  std::cout << std::endl;

  std::cout << "\n=== Example 2: Follow-up Question ===" << std::endl;
  std::cout << "User: What is your favorite animal and why is it capybara?" << std::endl;
  std::cout << "Assistant: ";

  llama.Prompt(
      "What is your favorite animal and why is it capybara?",
      [](const std::string& piece) { std::cout << piece << std::flush; }
  );
  std::cout << std::endl;

  return 0;
}