// LlamaVision integration example.
//
// A guided tour of the v2 API: one-time model load (the expensive step),
// the single-image convenience call, an independent follow-up question,
// a text-only prompt, and - optionally - video summarization via frame
// sampling.
//
// Run it from the build directory (paths below are relative to it):
//   ./llama_vision_example                    image + text examples
//   ./llama_vision_example path/to/clip.mp4   additionally summarizes video

#include <chrono>
#include <iostream>
#include <string>

#include "llama-vision.h"
#include "video-frames.h"

namespace {

constexpr auto kModelPath = "../models/gemma-3-4b-it-f16.gguf";
constexpr auto kProjectorPath = "../models/mmproj-model-f16.gguf";
constexpr auto kImagePath = "../images/kapybara.jpg";

double SecondsSince(std::chrono::steady_clock::time_point start) {
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration<double>(elapsed).count();
}

void PrintPiece(const std::string& piece) {
  std::cout << piece << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
  LlamaVision llama;

  // ---- Load once: this is the only expensive step. ----
  VisionModelParams modelParams;
  modelParams.modelPath = kModelPath;
  modelParams.projectorPath = kProjectorPath;
  modelParams.systemPrompt =
      "You are a helpful assistant that analyzes images. "
      "Answer clearly and to the point.";

  auto startedAt = std::chrono::steady_clock::now();
  if (!llama.Load(modelParams)) {
    std::cerr << "Load failed: " << llama.LoadError() << std::endl;
    return 1;
  }
  std::cout << "Model loaded in " << SecondsSince(startedAt)
            << "s - every call below reuses it.\n";

  // ---- 1. The one-liner. ----
  std::cout << "\n=== 1. DescribeImage ===\n";
  startedAt = std::chrono::steady_clock::now();

  GenerateResult described =
      llama.DescribeImage(kImagePath, "Describe this image.", PrintPiece);
  std::cout << "\n";
  if (!described.ok) {
    std::cerr << "Generate failed: " << described.error << std::endl;
    return 1;
  }
  std::cout << "(" << SecondsSince(startedAt) << "s, "
            << described.promptTokenCount << " prompt tokens, "
            << described.generatedTokenCount << " generated)\n";

  // ---- 2. An independent follow-up about the same image. ----
  // There is no conversation history: to ask something else about an
  // image, send the image again. Each call starts fresh.
  std::cout << "\n=== 2. Second, independent question ===\n";
  GenerateParams question;
  question.prompt = "What animal is shown, and what is it known for?";
  question.imagePaths = {kImagePath};
  question.maxTokens = 200;

  GenerateResult answer = llama.Generate(question, PrintPiece);
  std::cout << "\n";
  if (!answer.ok) {
    std::cerr << "Generate failed: " << answer.error << std::endl;
    return 1;
  }

  // ---- 3. Text-only: the library doubles as a local LLM. ----
  std::cout << "\n=== 3. Text-only prompt ===\n";
  GenerateParams textOnly;
  textOnly.prompt =
      "In one sentence: why do capybaras get along with other animals?";
  textOnly.maxTokens = 100;
  textOnly.systemPromptOverride = "You are a concise zoologist.";

  GenerateResult smallTalk = llama.Generate(textOnly, PrintPiece);
  std::cout << "\n";
  if (!smallTalk.ok) {
    std::cerr << "Generate failed: " << smallTalk.error << std::endl;
    return 1;
  }

  // ---- 4. Optional: summarize a video via frame sampling. ----
  if (argc > 1) {
    std::cout << "\n=== 4. Video: " << argv[1] << " ===\n";
    VideoFrameParams frameParams;
    frameParams.maxFrames = 6;

    VideoFrameResult frames = ExtractVideoFrames(argv[1], frameParams);
    if (!frames.ok) {
      std::cerr << "Frame extraction failed: " << frames.error << std::endl;
      return 1;
    }
    std::cout << "(sampled " << frames.framePaths.size() << " frames)\n";

    GenerateParams video;
    video.prompt =
        "These images are frames sampled from one video, in order. "
        "Describe what happens.";
    video.imagePaths = frames.framePaths;

    GenerateResult summary = llama.Generate(video, PrintPiece);
    std::cout << "\n";
    CleanupVideoFrames(frames);  // frames are read inside Generate()

    if (!summary.ok) {
      std::cerr << "Generate failed: " << summary.error << std::endl;
      return 1;
    }
    std::cout << "(" << summary.promptTokenCount << " prompt tokens for "
              << frames.framePaths.size() << " frames)\n";
  }

  return 0;
}
