#include "ultrahdr_api.h"
#include <cstdio>

void decode_to_hlg(const char* input_path, const char* output_path) {
    uhdr_compressed_image compressed_image;
    // 读取图像数据到compressed_image

    uhdr_raw_image raw_image;
    // 调用libultrahdr API进行解码
    uhdr_error_info_t error_info = uhdr_decode(&compressed_image, &raw_image, UHDR_CODEC_JPG);

    if (error_info.error_code == UHDR_CODEC_OK) {
        // 保存解码后的图像
        uhdr_write_jpeg(output_path, &raw_image);
    } else {
        printf("Decode failed with error code: %d\n", error_info.error_code);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <input_image>\n", argv[0]);
        return 1;
    }

    char output_path[256];
    snprintf(output_path, sizeof(output_path), "%s_hlg", argv[1]);

    decode_to_hlg(argv[1], output_path);
    return 0;
}