#include "image.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <algorithm>

namespace image {

float* load(const std::string& filename, int* W, int* H, int* C, int C0)
{
    const unsigned char* bytes = stbi_load(filename.c_str(), W, H, C, C0);
    const int C1 = C0 == 0 ? (*C) : C0;
    const int size = (*W) * (*H) * (C1);
    float* data = (float*)malloc(size * sizeof(float));

    for(auto i = 0; i < size; ++i) {
        const unsigned char value_uchar = bytes[i];
        const unsigned int value_uint = uint32_t(value_uchar);
        const float value_float = float(value_uint);
        data[i] = value_float / 255;
    }

    *C = C1;

    std::cout << "Image of size " << *W << "x" << *H << " with " << *C << " channels loaded from '" << filename << std::endl;

    return data;
}

void save(const std::string& filename, int W, int H, int C, const float* data)
{
    constexpr int quality = 100;
    const auto size = W * H * C;
    unsigned char* bytes = (unsigned char*)malloc(size * sizeof(unsigned char));

    for(auto i = 0; i < size; ++i) {
        bytes[i] = (unsigned char)(data[i]*255);
    }
    stbi_write_jpg(filename.c_str(), W, H, C, bytes, quality);
    free(bytes);
    std::cout << "Image of size " << W << "x" << H << " with " << C << " channels saved to '" << filename << std::endl;
}


} // namespace image
