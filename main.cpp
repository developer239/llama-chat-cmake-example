// LlamaVision integration example.
//
// A guided tour of the v2 API with hard-coded targets (no CLI params):
// one-time model load (the expensive step), then three tests in order -
// a simple text-only prompt, image vision, and video summarization via
// frame sampling.
//
// Run it from the build directory (paths below are relative to it):
//   ./llama_vision_example

#include <chrono>
#include <iostream>
#include <string>

#include "llama-vision.h"
#include "video-frames.h"

namespace {

constexpr auto kModelPath = "../models/gemma-3-4b-it-f16.gguf";
constexpr auto kProjectorPath = "../models/mmproj-model-f16.gguf";
constexpr auto kImagePath = "../images/input.jpg";
constexpr auto kVideoPath = "../images/input.mp4";

double SecondsSince(std::chrono::steady_clock::time_point start) {
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration<double>(elapsed).count();
}

void PrintPiece(const std::string& piece) {
  std::cout << piece << std::flush;
}

}  // namespace

int main() {
  LlamaVision llama;

  // ---- Load once: this is the only expensive step. ----
  VisionModelParams modelParams;
  modelParams.modelPath = kModelPath;
  modelParams.projectorPath = kProjectorPath;
  modelParams.systemPrompt =
      "You are a helpful assistant. Answer clearly and to the point.";

  auto startedAt = std::chrono::steady_clock::now();
  if (!llama.Load(modelParams)) {
    std::cerr << "Load failed: " << llama.LoadError() << std::endl;
    return 1;
  }
  std::cout << "Model loaded in " << SecondsSince(startedAt)
            << "s - every call below reuses it.\n";

  // ---- 1. Simple text-only prompt: the library doubles as a local LLM. ----
  std::cout << "\n=== 1. Simple text prompt ===\n";
  PromptParams textOnly;
  textOnly.prompt = "In one sentence, what is a capybara?";
  textOnly.maxTokens = 100;

  PromptResult text = llama.Prompt(textOnly, PrintPiece);
  std::cout << "\n";
  if (!text.ok) {
    std::cerr << "Prompt failed: " << text.error << std::endl;
    return 1;
  }

  // ---- 2. Image vision. ----
  std::cout << "\n=== 2. Image vision (" << kImagePath << ") ===\n";
  startedAt = std::chrono::steady_clock::now();

  PromptResult described =
      llama.DescribeImage(kImagePath, "Describe this image.", PrintPiece);
  std::cout << "\n";
  if (!described.ok) {
    std::cerr << "Prompt failed: " << described.error << std::endl;
    return 1;
  }
  std::cout << "(" << SecondsSince(startedAt) << "s, "
            << described.promptTokenCount << " prompt tokens, "
            << described.generatedTokenCount << " generated)\n";

  // ---- 3. Video summarization via frame sampling. ----
  std::cout << "\n=== 3. Video (" << kVideoPath << ") ===\n";
  VideoFrameParams frameParams;
  frameParams.maxFrames = 6;

  VideoFrameResult frames = ExtractVideoFrames(kVideoPath, frameParams);
  if (!frames.ok) {
    std::cerr << "Frame extraction failed: " << frames.error << std::endl;
    return 1;
  }
  std::cout << "(sampled " << frames.framePaths.size() << " frames)\n";

  PromptParams video;
  video.prompt =
      "These images are frames sampled from a single video, in order. "
      "Study them and reason about the sequence: is this one continuous "
      "scene or several distinct scenes? What is happening in each scene? "
      "How are the frames connected to one another - what changes from one "
      "to the next, and what stays the same, including between the first "
      "frame and the last? Be as precise as possible.";
  video.imagePaths = frames.framePaths;

  PromptResult summary = llama.Prompt(video, PrintPiece);
  std::cout << "\n";
  CleanupVideoFrames(frames);  // frames are read inside Prompt()

  if (!summary.ok) {
    std::cerr << "Prompt failed: " << summary.error << std::endl;
    return 1;
  }
  std::cout << "(" << summary.promptTokenCount << " prompt tokens for "
            << frames.framePaths.size() << " frames)\n";

  return 0;
}
