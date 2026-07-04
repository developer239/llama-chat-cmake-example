# LlamaVision CMake Example

Shows how to consume [llama-chat](https://github.com/developer239/llama-chat)
(the `LlamaVision` library, a wrapper around
[llama.cpp](https://github.com/ggml-org/llama.cpp)) from a plain CMake
project: one-off multimodal prompts against a **local** GGUF model - image
description, independent follow-up questions, text-only prompts, and
optional video summarization. No OpenCV, no manual preprocessing: images
go in as file paths.

The entire integration is two CMake lines:

```cmake
add_subdirectory(externals/llama-chat)
target_link_libraries(llama_vision_example PRIVATE LlamaVision)
```

## Project structure

```
llama-chat-cmake-example/
├── CMakeLists.txt
├── main.cpp
├── images/                  (gitignored - put any JPEG/PNG here)
│   └── kapybara.jpg
├── models/                  (gitignored - download model + mmproj here)
│   ├── gemma-3-4b-it-f16.gguf
│   └── mmproj-model-f16.gguf
└── externals/
    └── llama-chat/          (submodule - brings llama.cpp with it)
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
constants at the top of `main.cpp`. Put any JPEG or PNG at
`images/kapybara.jpg` - WebP is not supported by the bundled decoder.

For the optional video example, install ffmpeg (with ffprobe):
`brew install ffmpeg`, `sudo apt install ffmpeg`, or
`choco install ffmpeg`.

## Build & run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build
./llama_vision_example                    # image + text examples
./llama_vision_example path/to/clip.mp4   # additionally summarizes a video
```

Build **Release**: a Debug build of llama.cpp is an order of magnitude
slower at inference, which makes the model look broken when it's only
unoptimized. On macOS, Metal is enabled by default; on machines with an
NVIDIA GPU, add `-DGGML_CUDA=ON` to the first command.

Paths in `main.cpp` are relative to the build directory, so run the binary
from there (CLion's `cmake-build-debug`/`cmake-build-release` working
directories satisfy this too).

## What you should see

A load-time line first (this is the only expensive step - seconds), then
four sections: a streamed description of your image with its prompt/output
token counts, an independent second answer about the same image, a
one-sentence text-only reply, and - if you passed a video - a frame count,
a streamed summary, and the total prompt-token cost of the sampled frames.
Every generation after load takes on the order of a second or two on a
modern machine; that gap between load cost and call cost is exactly why a
long-lived process (an MCP server, a daemon) is the natural home for this
library.
