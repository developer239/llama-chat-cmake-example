# Llame Worker Example

Shows how to consume [llame-worker](https://github.com/developer239/llame-worker)
(the `LlamaVision` library) from a plain CMake project. One-off multimodal
prompts against a local GGUF model: text, image, and video.

The entire integration is two CMake lines:

```cmake
add_subdirectory(externals/llame-worker)
target_link_libraries(llama_vision_example PRIVATE LlamaVision)
```

## Requirements

A vision-capable GGUF model **and** its matching multimodal projector
(`mmproj-*.gguf`) in `models/`, plus `input.jpg` and `input.mp4` in
`images/` (both paths are hard-coded in `main.cpp`). For video, `ffmpeg`
(with `ffprobe`) must be on PATH.

Ready-to-use vision GGUFs (model + mmproj) live in the ggml-org
[multimodal collection](https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc)
on Hugging Face. Gemma 3 is the safe default; for stronger text/UI reading
try Qwen2.5-VL, InternVL3, or Pixtral. Download with the `hf` CLI, then
point the constants at the top of `main.cpp` at the files:

```bash
hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF \
  Qwen2.5-VL-7B-Instruct-Q8_0.gguf mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf \
  --local-dir models
```

## Setup

```bash
git submodule update --init --recursive
```

## Build & run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build
./llama_vision_example
```

Build **Release**: a Debug build of llama.cpp is an order of magnitude
slower at inference. On macOS, Metal is enabled by default; for NVIDIA
GPUs, add `-DGGML_CUDA=ON` to the configure step. The binary takes no
arguments and runs three sections: a text-only prompt, image description,
and a frame-sampled video summary.
