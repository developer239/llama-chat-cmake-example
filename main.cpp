#include <iostream>

#include "llama-chat.h"

/**
 * Example: Using LlamaChat with multimodal (vision) support
 *
 * Requirements:
 * - A vision-capable model (e.g., Gemma 3, LLaVA, MiniCPM-V, etc.)
 * - The corresponding mmproj (multimodal projector) file
 *
 * You can get compatible models from:
 * -
 * https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc
 *
 * Example models:
 * - Gemma 3: gemma-3-4b-it-Q4_K_M.gguf + mmproj-gemma-3-4b-it-Q4_K_M.gguf
 * - LLaVA 1.5: llava-v1.5-7b-Q4_K_M.gguf + mmproj-model-f16.gguf
 */

int main() {
  LlamaChat llama;

  // Configure model parameters with multimodal support
  ModelParams modelParams;
  modelParams.nGpuLayers = 32;

  // Specify the multimodal projector path
  // This is required for vision support
  modelParams.multiModalPath = "../models/gemma-3-4b-it-f16";
  modelParams.offloadMultiModalToGPU = true;

  // Load the main model
  if (!llama.InitializeModel(
          "../models/gemma-3-4b-it-Q4_K_M.gguf",
          modelParams
      )) {
    std::cerr << "Failed to initialize the model." << std::endl;
    return 1;
  }

  // Initialize context
  ContextParams ctxParams;
  ctxParams.nContext = 4096;  // Vision models often need larger context
  ctxParams.nBatch = 512;

  if (!llama.InitializeContext(ctxParams)) {
    std::cerr << "Failed to initialize the context." << std::endl;
    return 1;
  }

  // Check if vision support is available
  if (llama.HasVisionSupport()) {
    std::cout << "Vision support is available!" << std::endl;
  } else {
    std::cerr << "Warning: Vision support is not available. "
              << "Make sure the mmproj file is correct." << std::endl;
  }

  // Set system prompt
  std::string systemPrompt =
      "You are a helpful assistant that can analyze images and answer "
      "questions about them. "
      "Provide clear, detailed descriptions when asked about images.";
  llama.SetSystemPrompt(systemPrompt);

  // Example 1: Describe an image
  std::cout << "\n=== Example 1: Image Description ===" << std::endl;
  std::cout << "User: What do you see in this image?" << std::endl;
  std::cout << "Assistant: ";

  ImageInput image = ImageInput::FromPath("../images/test.png");
  llama.PromptWithImage(
      "What do you see in this image? Please describe it in detail.",
      image,
      [](const std::string& piece) { std::cout << piece << std::flush; }
  );
  std::cout << std::endl;

  // Example 2: Ask a follow-up question about the same image
  std::cout << "\n=== Example 2: Follow-up Question ===" << std::endl;
  std::cout << "User: What colors are predominant in the image?" << std::endl;
  std::cout << "Assistant: ";

  llama.Prompt(
      "What colors are predominant in the image?",
      [](const std::string& piece) { std::cout << piece << std::flush; }
  );
  std::cout << std::endl;

  // Example 3: Analyze a different image
  std::cout << "\n=== Example 3: Analyze Another Image ===" << std::endl;
  llama.ResetConversation();  // Reset to start fresh
  llama.SetSystemPrompt(systemPrompt);

  std::cout << "User: Is there any text in this image? If so, what does it say?"
            << std::endl;
  std::cout << "Assistant: ";

  ImageInput documentImage = ImageInput::FromPath("../images/document.png");
  llama.PromptWithImage(
      "Is there any text in this image? If so, what does it say?",
      documentImage,
      [](const std::string& piece) { std::cout << piece << std::flush; }
  );
  std::cout << std::endl;

  // Example 4: Text-only prompt (still works without image)
  std::cout << "\n=== Example 4: Text-only Prompt ===" << std::endl;
  llama.ResetConversation();
  llama.SetSystemPrompt("You are a helpful assistant.");

  std::cout << "User: What is the capital of France?" << std::endl;
  std::cout << "Assistant: ";

  llama.Prompt("What is the capital of France?", [](const std::string& piece) {
    std::cout << piece << std::flush;
  });
  std::cout << std::endl;

  return 0;
}