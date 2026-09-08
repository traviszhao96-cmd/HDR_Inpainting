#include <opencv2/opencv.hpp>
#include <iostream>

// 转换为RGBA8888格式并保存为TIFF
bool convertToRGBA8888(const cv::Mat& rgb, const std::string& outputPath) {
    cv::Mat rgba;
    cv::cvtColor(rgb, rgba, cv::COLOR_RGB2RGBA);

    // 设置Alpha通道为255
    for (int i = 0; i < rgba.rows; ++i) {
        for (int j = 0; j < rgba.cols; ++j) {
            rgba.at<cv::Vec4b>(i, j)[3] = 255;
        }
    }

    std::vector<int> params;
    params.push_back(cv::IMWRITE_TIFF_COMPRESSION);
    params.push_back(1); // 无压缩
    params.push_back(cv::IMWRITE_TIFF_XDPI);
    params.push_back(300);
    params.push_back(cv::IMWRITE_TIFF_YDPI);
    params.push_back(300);

    return cv::imwrite(outputPath, rgba, params);
}

// 转换为ABGR2101010格式并保存为TIFF
bool convertToABGR2101010(const cv::Mat& rgb, const std::string& outputPath) {
    cv::Mat rgba1010102(rgb.rows, rgb.cols, CV_32SC1); // 使用32位有符号整数存储

    for (int i = 0; i < rgb.rows; ++i) {
        for (int j = 0; j < rgb.cols; ++j) {
            cv::Vec3b pixel = rgb.at<cv::Vec3b>(i, j);
            // 将8位RGB转换为10位ABGR2101010格式
            // A(2位) = 3 (全不透明)
            // B(10位) = pixel[0] << 2
            // G(10位) = pixel[1] << 2
            // R(10位) = pixel[2] << 2
            int32_t abgr = (3 << 30) |               // Alpha (2位)
                          ((pixel[0] << 2) << 20) | // Blue (10位)
                          ((pixel[1] << 2) << 10) |  // Green (10位)
                          (pixel[2] << 2);           // Red (10位)
            rgba1010102.at<int32_t>(i, j) = abgr;
        }
    }

    std::vector<int> params;
    params.push_back(cv::IMWRITE_TIFF_COMPRESSION);
    params.push_back(1); // 无压缩
    params.push_back(cv::IMWRITE_TIFF_XDPI);
    params.push_back(300);
    params.push_back(cv::IMWRITE_TIFF_YDPI);
    params.push_back(300);

    return cv::imwrite(outputPath, rgba1010102, params);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_jpg> <output_tif> <format>\n";
        std::cerr << "Format: 0=RGBA8888, 1=ABGR2101010\n";
        return 1;
    }

    cv::Mat rgb = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (rgb.empty()) {
        std::cerr << "Error: Could not read input file\n";
        return 1;
    }

    int format = std::stoi(argv[3]);
    bool success = false;
    
    if (format == 0) {
        success = convertToRGBA8888(rgb, argv[2]);
    } else if (format == 1) {
        success = convertToABGR2101010(rgb, argv[2]);
    } else {
        std::cerr << "Error: Invalid format specified\n";
        return 1;
    }

    if (!success) {
        std::cerr << "Error: Could not write output file\n";
        return 1;
    }

    return 0;
}