cmake_minimum_required(VERSION 3.28)
project(llama_wrapped_cmake_example)

set(CMAKE_CXX_STANDARD 17)

add_executable(llama_wrapped_cmake_example main.cpp)

add_subdirectory(externals/llama-chat)
target_link_libraries(llama_wrapped_cmake_example PRIVATE LlamaChat)
