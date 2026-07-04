cmake_minimum_required(VERSION 3.24)
project(llameworker_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The entire integration: pull in the library (which brings llama.cpp
# along) and link the target.
add_subdirectory(externals/llame-worker)

add_executable(llameworker_example main.cpp)
target_link_libraries(llameworker_example PRIVATE llameworker)
