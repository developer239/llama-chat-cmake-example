// llameworker integration example.
//
// A guided tour of the API with hard-coded targets (no CLI params):
// one-time model load (the expensive step), then three tests in order -
// a simple text-only prompt, image vision, and video summarization via
// frame sampling.
//
// Run it from the build directory (paths below are relative to it):
//   ./llameworker_example

#include <chrono>
#include <print>
#include <string>
#include <string_view>

#include "llameworker.h"
#include "video-frames.h"

namespace lw = llameworker;

namespace {

constexpr auto kModelPath = "../models/gemma-3-4b-it-f16.gguf";
constexpr auto kProjectorPath = "../models/mmproj-model-f16.gguf";
constexpr auto kImagePath = "../images/input.jpg";
constexpr auto kVideoPath = "../images/input.mp4";

double secondsSince(std::chrono::steady_clock::time_point start) {
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration<double>(elapsed).count();
}

void printPiece(std::string_view piece) { std::print("{}", piece); }

}  // namespace

int main() {
  lw::LlameWorker worker;

  // ---- Load once: this is the only expensive step. ----
  lw::VisionModelParams modelParams;
  modelParams.modelPath = kModelPath;
  modelParams.projectorPath = kProjectorPath;
  modelParams.systemPrompt =
      "You are a helpful assistant. Answer clearly and to the point.";

  auto startedAt = std::chrono::steady_clock::now();
  if (!worker.load(modelParams)) {
    std::println(stderr, "Load failed: {}", worker.loadError());
    return 1;
  }
  std::println(
      "Model loaded in {}s - every call below reuses it.",
      secondsSince(startedAt));

  // ---- 1. Simple text-only prompt: the library doubles as a local LLM. ----
  std::println("\n=== 1. Simple text prompt ===");
  lw::PromptParams textOnly;
  textOnly.prompt = "In one sentence, what is a capybara?";
  textOnly.maxTokens = 100;

  lw::PromptResult text = worker.prompt(textOnly, printPiece);
  std::print("\n");
  if (!text.ok) {
    std::println(stderr, "Prompt failed: {}", text.error);
    return 1;
  }

  // ---- 2. Image vision. ----
  std::println("\n=== 2. Image vision ({}) ===", kImagePath);
  startedAt = std::chrono::steady_clock::now();

  lw::PromptResult described =
      worker.describeImage(kImagePath, "Describe this image.", printPiece);
  std::print("\n");
  if (!described.ok) {
    std::println(stderr, "Prompt failed: {}", described.error);
    return 1;
  }
  std::println(
      "({}s, {} prompt tokens, {} generated)", secondsSince(startedAt),
      described.promptTokenCount, described.generatedTokenCount);

  // ---- 3. Video summarization via frame sampling. ----
  std::println("\n=== 3. Video ({}) ===", kVideoPath);
  lw::VideoFrameParams frameParams;
  frameParams.maxFrames = 6;

  lw::VideoFrameResult frames = lw::extractVideoFrames(kVideoPath, frameParams);
  if (!frames.ok) {
    std::println(stderr, "Frame extraction failed: {}", frames.error);
    return 1;
  }
  std::println("(sampled {} frames)", frames.framePaths.size());

  lw::PromptParams video;
  video.prompt =
      "These images are frames sampled from a single video, in order. "
      "Study them and reason about the sequence: is this one continuous "
      "scene or several distinct scenes? What is happening in each scene? "
      "How are the frames connected to one another - what changes from one "
      "to the next, and what stays the same, including between the first "
      "frame and the last? Be as precise as possible.";
  video.imagePaths = frames.framePaths;

  lw::PromptResult summary = worker.prompt(video, printPiece);
  std::print("\n");
  lw::cleanupVideoFrames(frames);  // frames are read inside prompt()

  if (!summary.ok) {
    std::println(stderr, "Prompt failed: {}", summary.error);
    return 1;
  }
  std::println(
      "({} prompt tokens for {} frames)", summary.promptTokenCount,
      frames.framePaths.size());

  return 0;
}
