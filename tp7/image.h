#include "stb_image.h"
#include "stb_image_write.h"

#include <string>

namespace image {

float* load(const std::string& filename, int* N, int* M, int* C, int C0 = 0);
void save(const std::string& filename, int N, int M, int C, const float* data);

} // namespace image




