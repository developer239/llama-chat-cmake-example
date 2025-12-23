cmake_minimum_required(VERSION 3.28)
project(llama_wrapped_cmake_example)

set(CMAKE_CXX_STANDARD 17)

add_executable(llama_wrapped_cmake_example main.cpp)

# OpenCV
if(APPLE)
    find_package(OpenCV REQUIRED)
    target_link_libraries(llama_wrapped_cmake_example PRIVATE "-framework Carbon")
    target_link_libraries(llama_wrapped_cmake_example PRIVATE ${OpenCV_LIBS})
elseif(WIN32)
    set(OpenCV_DIR "C:/tools/opencv/build/x64/vc15/lib")
    find_package(OpenCV REQUIRED)
    target_link_libraries(llama_wrapped_cmake_example PRIVATE ${OpenCV_LIBS})
endif()

add_subdirectory(externals/llama-chat)
target_link_libraries(llama_wrapped_cmake_example PRIVATE LlamaChat)
