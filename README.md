# LlamaChat CMake Example

This project demonstrates how to use the [LlamaChat](https://github.com/developer239/llama-chat) library in a C++ application. It shows how to initialize the model, set up the context, and run multiple queries using the LlamaChat interface.

## Project Structure

```
llamachat-example/
│
├── CMakeLists.txt
├── src/
│   └── main.cpp
├── models/
│   └── Meta-Llama-3.1-8B-Instruct-Q3_K_S.gguf
└── externals/
    └── llama-chat/
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/your-username/llamachat-example.git
   cd llamachat-example
   ```

2. Initialize and update the LlamaChat submodule:
   ```
   git submodule update --init --recursive
   ```

3. Place your language model file (e.g., Meta-Llama-3.1-8B-Instruct-Q3_K_S.gguf) in the `models/` directory.

4. Create a build directory and run CMake:
   ```
   mkdir build
   cd build
   cmake ..
   ```

5. Build the project:
   ```
   cmake --build .
   ```

## Usage

After building the project, you can run the example application:

```
./llamachat_example
```

The application will initialize the LlamaChat model, set a system prompt, and then ask a series of predefined questions. The responses will be printed to the console.
