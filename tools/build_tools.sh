#!/bin/bash

# 编译提取RGB工具
g++ -std=c++11 -I/opt/homebrew/opt/opencv/include/opencv4 -L/opt/homebrew/opt/opencv/lib \
    -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
    /Users/travis.zhao/ultrahdr/tools/extract_rgb_from_raw.cpp \
    -o /Users/travis.zhao/ultrahdr/tools/extract_rgb_from_raw

# 编译JPEG转RGBA TIFF工具
g++ -std=c++11 -I/opt/homebrew/opt/opencv/include/opencv4 -L/opt/homebrew/opt/opencv/lib \
    -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
    /Users/travis.zhao/ultrahdr/tools/jpg_to_rgba_tif.cpp \
    -o /Users/travis.zhao/ultrahdr/tools/jpg_to_rgba_tif

echo "Tools built successfully"