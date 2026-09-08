#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// 生成所有可能的通道排列组合
void generateAllChannelPermutations(const uint32_t* input, int width, int height, const std::string& baseOutputPath, int format) {
    std::vector<uint8_t> rgbData(width * height * 3);
    
    // 定义通道排列组合
    struct ChannelOrder {
        int rShift, gShift, bShift;
        std::string name;
    };
    
    // 根据格式选择不同的通道排列
    std::vector<ChannelOrder> orders;
    
    if (format == 0) { // RGBA8888
        orders = {
            {0, 8, 16, "RGB"},   // R=0-7, G=8-15, B=16-23
            {0, 16, 8, "RBG"},    // R=0-7, B=8-15, G=16-23
            {8, 0, 16, "GRB"},    // G=0-7, R=8-15, B=16-23
            {8, 16, 0, "GBR"},    // G=0-7, B=8-15, R=16-23
            {16, 0, 8, "BRG"},    // B=0-7, R=8-15, G=16-23
            {16, 8, 0, "BGR"}     // B=0-7, G=8-15, R=16-23
        };
    } else { // ABGR2101010
        orders = {
            {20, 10, 0, "RGB"},   // R=20-29, G=10-19, B=0-9
            {20, 0, 10, "RBG"},   // R=20-29, B=10-19, G=0-9
            {10, 20, 0, "GRB"},   // G=20-29, R=10-19, B=0-9
            {10, 0, 20, "GBR"},   // G=20-29, B=10-19, R=0-9
            {0, 20, 10, "BRG"},   // B=20-29, R=10-19, G=0-9
            {0, 10, 20, "BGR"}    // B=20-29, G=10-19, R=0-9
        };
    }
    
    for (const auto& order : orders) {
        // 转换像素
        for (int i = 0; i < width * height; i++) {
            uint32_t pixel = input[i];
            
            uint8_t r = (pixel >> order.rShift) & 0xFF;
            uint8_t g = (pixel >> order.gShift) & 0xFF;
            uint8_t b = (pixel >> order.bShift) & 0xFF;
            
            rgbData[i*3]     = r;
            rgbData[i*3 + 1] = g;
            rgbData[i*3 + 2] = b;
        }
        
        // 创建OpenCV图像
        cv::Mat rgb(height, width, CV_8UC3, rgbData.data());
        
        // 保存为JPEG
        std::string outputPath = baseOutputPath + "_" + order.name + ".jpg";
        std::vector<int> params;
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(100);
        
        cv::imwrite(outputPath, rgb, params);
        std::cout << "Generated " << outputPath << std::endl;
    }
}

// 修改后的通道排列组合生成函数
void generateRGBChannelPermutations(const uint32_t* input, int width, int height, const std::string& baseOutputPath) {
    std::vector<uint8_t> rgbData(width * height * 3);
    
    // 只定义RGB通道的排列组合，保持Alpha通道不变
    struct RGBOrder {
        int rShift, gShift, bShift;
        std::string name;
    };
    
    RGBOrder orders[] = {
        {0, 8, 16, "RGB"},   // 标准顺序
        {0, 16, 8, "RBG"},   // 交换G和B
        {8, 0, 16, "GRB"},   // 交换R和G
        {8, 16, 0, "GBR"},   // R和B交换位置
        {16, 0, 8, "BRG"},   // G和B交换位置
        {16, 8, 0, "BGR"}    // 完全反转顺序
    };
    
    for (const auto& order : orders) {
        // 转换像素
        for (int i = 0; i < width * height; i++) {
            uint32_t pixel = input[i];
            
            // 提取RGB通道（忽略Alpha通道）
            uint8_t r = (pixel >> order.rShift) & 0xFF;
            uint8_t g = (pixel >> order.gShift) & 0xFF;
            uint8_t b = (pixel >> order.bShift) & 0xFF;
            
            rgbData[i*3]     = r;
            rgbData[i*3 + 1] = g;
            rgbData[i*3 + 2] = b;
        }
        
        // 创建OpenCV图像
        cv::Mat rgb(height, width, CV_8UC3, rgbData.data());
        
        // 保存为JPEG
        std::string outputPath = baseOutputPath + "_" + order.name + ".jpg";
        std::vector<int> params;
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(100);
        
        cv::imwrite(outputPath, rgb, params);
        std::cout << "Generated " << outputPath << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_raw> <output_base_path> [width] [height] [format]\n";
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
    
    // 读取数据
    if (!file.read(reinterpret_cast<char*>(rawData.data()), pixelCount * sizeof(uint32_t))) {
        std::cerr << "Error: Could not read " << pixelCount * sizeof(uint32_t) 
                  << " bytes from file. Only read " << file.gcount() << " bytes." << std::endl;
        return 1;
    }
    
    // 生成所有可能的通道排列
    generateAllChannelPermutations(rawData.data(), width, height, argv[2], format);
    
    return 0;
}