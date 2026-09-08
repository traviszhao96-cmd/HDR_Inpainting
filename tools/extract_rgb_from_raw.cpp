#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

// 辅助函数：从2101010 ABGR格式解析像素
void convert2101010ABGRToRGB(const uint32_t* input, uint8_t* output, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        uint32_t pixel = input[i];
        
        // 从2101010格式提取BGR通道 (忽略Alpha通道)
        // ABGR格式中，BGR顺序为：B=低位, G=中位, R=高位
        int b = pixel & 0x3FF;          // 10位B通道 (最低10位)
        int g = (pixel >> 10) & 0x3FF;  // 10位G通道 (中间10位)
        int r = (pixel >> 20) & 0x3FF;  // 10位R通道 (高10位)
        
        // 将10位值转换为8位，并按RGB顺序存储
        output[i*3]     = (uint8_t)(r * 255 / 1023);  // R
        output[i*3 + 1] = (uint8_t)(g * 255 / 1023);  // G
        output[i*3 + 2] = (uint8_t)(b * 255 / 1023);  // B
    }
}

// 新增函数：从RGBA8888格式解析像素
void convertRGBA8888ToRGB(const uint32_t* input, uint8_t* output, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        uint32_t pixel = input[i];
        
        // 从RGBA8888格式提取RGB通道 (忽略Alpha通道)
        uint8_t r = (pixel >> 0) & 0xFF;   // R通道 (最低8位)
        uint8_t g = (pixel >> 8) & 0xFF;   // G通道 (次低8位)
        uint8_t b = (pixel >> 16) & 0xFF;  // B通道 (次高8位)
        // Alpha通道在最高8位，被忽略
        
        // 按RGB顺序存储
        output[i*3]     = b;  // B
        output[i*3 + 1] = g;  // G
        output[i*3 + 2] = r;  // R
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_raw> <output_jpg> [width] [height] [format]\n";
        std::cerr << "Format: 0=RGBA8888, 1=ABGR2101010 (default: 1)\n";
        return 1;
    }

    // 获取图像尺寸，默认为2464x3280
    int width = (argc > 3) ? std::stoi(argv[3]) : 2464;
    int height = (argc > 4) ? std::stoi(argv[4]) : 3280;
    // 获取格式，默认为ABGR2101010
    int format = (argc > 5) ? std::stoi(argv[5]) : 1;
    
    std::cout << "Processing image with dimensions: " << width << "x" << height 
              << ", format: " << (format == 0 ? "RGBA8888" : "ABGR2101010") << std::endl;
    
    // 读取RAW文件
    std::ifstream file(argv[1], std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open input file: " << argv[1] << std::endl;
        return 1;
    }
    
    // 分配内存
    size_t pixelCount = width * height;
    std::vector<uint32_t> rawData(pixelCount);
    std::vector<uint8_t> rgbData(pixelCount * 3);
    
    // 读取数据
    if (!file.read(reinterpret_cast<char*>(rawData.data()), pixelCount * sizeof(uint32_t))) {
        std::cerr << "Error: Could not read " << pixelCount * sizeof(uint32_t) 
                  << " bytes from file. Only read " << file.gcount() << " bytes." << std::endl;
        return 1;
    }
    
    // 根据格式选择转换函数
    if (format == 0) {
        // RGBA8888格式
        convertRGBA8888ToRGB(rawData.data(), rgbData.data(), width, height);
    } else {
        // ABGR2101010格式
        convert2101010ABGRToRGB(rawData.data(), rgbData.data(), width, height);
    }
    
    // 创建OpenCV图像
    cv::Mat rgb(height, width, CV_8UC3, rgbData.data());
    
    // 保存为无压缩JPEG
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(100); // 最高质量，无压缩
    params.push_back(cv::IMWRITE_JPEG_PROGRESSIVE);
    params.push_back(0);
    params.push_back(cv::IMWRITE_JPEG_OPTIMIZE);
    params.push_back(0);

    if (!cv::imwrite(argv[2], rgb, params)) {
        std::cerr << "Error: Could not write output file\n";
        return 1;
    }

    std::cout << "Successfully converted " << argv[1] << " to " << argv[2] << std::endl;
    return 0;
}