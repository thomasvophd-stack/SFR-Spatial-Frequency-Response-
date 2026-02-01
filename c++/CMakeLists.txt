cmake_minimum_required(VERSION 3.10)

# Project name
project(SFR_Spatial_Frequency_Response)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/opt/homebrew/opt/eigen/include/eigen3/")
include_directories("/opt/homebrew/opt/jsoncpp/include/")
include_directories(third_party/kiss_fft130) # Correct path to KISS FFT include directory

# Find Eigen3 package
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add all source files
add_executable(SFR_Spatial_Frequency_Response 
    src/main.cpp 
    src/sfr_utility.cpp 
    third_party/kiss_fft130/kiss_fft.c # Add KISS FFT source file
)

# Link libraries
target_link_libraries(SFR_Spatial_Frequency_Response 
    ${OpenCV_LIBS} 
    /opt/homebrew/opt/jsoncpp/lib/libjsoncpp.dylib
    nlohmann_json::nlohmann_json
)