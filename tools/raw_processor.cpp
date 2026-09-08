#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

struct RawConfig {
    int width;
    int height;
    int channels;
    int bitDepth;
    std::string byteOrder;
    std::string colorSpace;
    std::string fileType;
    std::string creationTime;
    std::string sourceFile;
};

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void writeConfigFile(const std::string& configPath, const RawConfig& config) {
    std::ofstream configFile(configPath);
    configFile << "[Header]\n"
               << "Width=" << config.width << "\n"
               << "Height=" << config.height << "\n"
               << "Channels=" << config.channels << "\n"
               << "BitDepth=" << config.bitDepth << "\n"
               << "ByteOrder=" << config.byteOrder << "\n"
               << "ColorSpace=" << config.colorSpace << "\n"
               << "FileType=" << config.fileType << "\n\n"
               << "[Metadata]\n"
               << "CreationTime=" << config.creationTime << "\n"
               << "SourceFile=" << config.sourceFile << "\n";
}

RawConfig readConfigFile(const std::string& configPath) {
    RawConfig config;
    std::ifstream file(configPath);
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream is_line(line);
        std::string key;
        if (std::getline(is_line, key, '=')) {
            std::string value;
            if (std::getline(is_line, value)) {
                if (key == "Width") config.width = std::stoi(value);
                else if (key == "Height") config.height = std::stoi(value);
                else if (key == "Channels") config.channels = std::stoi(value);
                else if (key == "BitDepth") config.bitDepth = std::stoi(value);
                else if (key == "ByteOrder") config.byteOrder = value;
                else if (key == "ColorSpace") config.colorSpace = value;
                else if (key == "FileType") config.fileType = value;
                else if (key == "CreationTime") config.creationTime = value;
                else if (key == "SourceFile") config.sourceFile = value;
            }
        }
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_raw_file> <output_config_file>\n";
        return 1;
    }

    std::string rawFilePath = argv[1];
    std::string configFilePath = argv[2];

    // 假设这些参数是已知的或从其他地方获取
    RawConfig config = {
        .width = 2464,
        .height = 3280,
        .channels = 4,
        .bitDepth = 32,
        .byteOrder = "BigEndian",
        .colorSpace = "HLG",
        .fileType = "RAW",
        .creationTime = getCurrentTimestamp(),
        .sourceFile = rawFilePath
    };

    writeConfigFile(configFilePath, config);
    std::cout << "Config file created successfully: " << configFilePath << "\n";

    // 读取示例
    RawConfig readConfig = readConfigFile(configFilePath);
    std::cout << "Read config: Width=" << readConfig.width << ", Height=" << readConfig.height << "\n";

    return 0;
}