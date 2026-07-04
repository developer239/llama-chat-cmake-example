# LlamaVision CMake Example

Shows how to consume [llame-worker](https://github.com/developer239/llame-worker)
(the `LlamaVision` library, a wrapper around
[llama.cpp](https://github.com/ggml-org/llama.cpp)) from a plain CMake
project: one-off multimodal prompts against a **local** GGUF model - image
description, independent follow-up questions, text-only prompts, and
optional video summarization. No OpenCV, no manual preprocessing: images
go in as file paths.

The entire integration is two CMake lines:

```cmake
add_subdirectory(externals/llame-worker)
target_link_libraries(llama_vision_example PRIVATE LlamaVision)
```

## Project structure

```
llame-worker-example/
├── CMakeLists.txt
├── main.cpp
├── images/                  (gitignored - put your input.jpg and input.mp4 here)
│   ├── input.jpg
│   └── input.mp4
├── models/                  (gitignored - download model + mmproj here)
│   ├── gemma-3-4b-it-f16.gguf
│   └── mmproj-model-f16.gguf
└── externals/
    └── llame-worker/        (submodule - brings llama.cpp with it)
```

## Setup

Clone with submodules, or initialize them in an existing clone:

```bash
git clone --recurse-submodules <this-repo>
# or:
git submodule update --init --recursive
```

Download a vision model **and its matching multimodal projector** into
`models/`. The example expects `gemma-3-4b-it-f16.gguf` plus
`mmproj-model-f16.gguf` (available from the ggml-org repositories on
Hugging Face); any vision model + mmproj pair works if you update the
constants at the top of `main.cpp`. Put your test inputs at
`images/input.jpg` and `images/input.mp4` (the paths are hard-coded in
`main.cpp`) - WebP is not supported by the bundled decoder.

For the video test, install ffmpeg (with ffprobe):
`brew install ffmpeg`, `sudo apt install ffmpeg`, or
`choco install ffmpeg`.

## Build & run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build
./llama_vision_example
```

The binary takes no arguments - the model, image, and video paths are
hard-coded at the top of `main.cpp`.

Build **Release**: a Debug build of llama.cpp is an order of magnitude
slower at inference, which makes the model look broken when it's only
unoptimized. On macOS, Metal is enabled by default; on machines with an
NVIDIA GPU, add `-DGGML_CUDA=ON` to the first command.

Paths in `main.cpp` are relative to the build directory, so run the binary
from there (CLion's `cmake-build-debug`/`cmake-build-release` working
directories satisfy this too).

## What you should see

A load-time line first (this is the only expensive step - seconds), then
three sections: a one-sentence text-only reply, a description of
`input.jpg` with its prompt/output token counts, and a frame-sampled
summary of `input.mp4` with the total prompt-token cost of the sampled
frames. Every generation after load takes on the order of a second or two
on a modern machine; that gap between load cost and call cost is exactly
why a long-lived process (an MCP server, a daemon) is the natural home for
this library.
