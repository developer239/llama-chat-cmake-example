#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "llama-chat.h"

int main() {
  const auto IMAGE_PATH = "../images/kapybara.jpg";

  // Load image using OpenCV
  cv::Mat image = cv::imread(IMAGE_PATH);
  if (image.empty()) {
    std::cerr << "Failed to load image: " << IMAGE_PATH << std::endl;
    return 1;
  }

  std::cout << "Original image size: " << image.cols << "x" << image.rows << std::endl;

  // Preprocessing: resize to max 512px on longest side (preserving aspect ratio)
  int maxSize = 512;
  double scale = static_cast<double>(maxSize) / std::max(image.cols, image.rows);
  if (scale < 1.0) {
    cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);
    std::cout << "Resized image to: " << image.cols << "x" << image.rows << std::endl;
  }

  // Preprocessing: apply slight Gaussian blur for noise reduction
  cv::Mat processed;
  cv::GaussianBlur(image, processed, cv::Size(3, 3), 0);

  // Show preview
  cv::imshow("Preprocessed Image (Press any key to continue)", processed);
  cv::waitKey(0);
  cv::destroyAllWindows();

  // Convert BGR to RGB for the LLM
  cv::Mat rgb;
  cv::cvtColor(processed, rgb, cv::COLOR_BGR2RGB);

  // Ensure continuous memory layout
  if (!rgb.isContinuous()) {
    rgb = rgb.clone();
  }

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

  ImageInput imageInput = ImageInput::FromRGBData(
      static_cast<uint32_t>(rgb.cols),
      static_cast<uint32_t>(rgb.rows),
      rgb.data
  );

  llama.Prompt(
      "What do you see in this image?",
      [](const std::string& piece) { std::cout << piece << std::flush; },
      imageInput
  );
  std::cout << std::endl;

  std::cout << "\n=== Example 2: Follow-up Question ===" << std::endl;
  std::cout << "User: What is your favorite animal and why is it capybara?"
            << std::endl;
  std::cout << "Assistant: ";

  llama.Prompt(
      "What is your favorite animal and why is it capybara?",
      [](const std::string& piece) { std::cout << piece << std::flush; }
  );
  std::cout << std::endl;

  return 0;
}