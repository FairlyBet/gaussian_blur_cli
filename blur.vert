#version 430 core

// layout(local_size_x = 2, local_size_y = 2) in;
layout(binding = 0, rgba8) restrict readonly uniform imageBuffer input_image;
layout(binding = 1, rgba8) restrict writeonly uniform imageBuffer output_image;
layout(std140, binding = 2) restrict readonly uniform ImageData {
    int offset;
    int width;
    int height;
};
const int KERNEL_SIZE = 43;
const float[KERNEL_SIZE] KERNEL = float[KERNEL_SIZE](0.000634461, 0.00096405006, 0.0014352618, 0.0020936271, 0.002992296, 0.004190314, 0.0057494384, 0.0077293175, 0.010181077, 0.01313963, 0.016615346, 0.020586025, 0.024990356, 0.029724136, 0.034640398, 0.03955427, 0.044252794, 0.04850929, 0.052100986, 0.05482818, 0.056532543, 0.057112362, 0.056532543, 0.05482818, 0.052100986, 0.04850929, 0.044252794, 0.03955427, 0.034640398, 
0.029724136, 0.024990356, 0.020586025, 0.016615346, 0.01313963, 0.010181077, 0.0077293175, 0.0057494384, 0.004190314, 0.002992296, 0.0020936271, 0.0014352618, 0.00096405006, 0.000634461);

uniform ivec2 direction;

const int RGBA = 4;

vec4 fetch_pixel(ivec2 pos) {
    int x = pos.x * RGBA;
    int y = pos.y * width * RGBA;
    return imageLoad(input_image, x + y);
}

void write_pixel(ivec2 pos, vec4 pixel) {
    int x = pos.x * RGBA;
    int y = pos.y * width * RGBA;
    imageStore(output_image, x + y, pixel);
}

void main() {
    // ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 pos = ivec2(0, 0);

    if (pos.x >= width || pos.y >= height) return;

    vec4 sum = vec4(0.0);
    for (int i = 0; i < KERNEL_SIZE; ++i)
    {
        ivec2 npos = pos + direction * (i - KERNEL_SIZE / 2);
        if (npos.x < 0) npos.x = 0;
        if (npos.y < 0) npos.y = 0;
        if (npos.x >= width) npos.x = width - 1;
        if (npos.y >= height) npos.y = height - 1;
        sum += KERNEL[i] * fetch_pixel(npos);
    }
    write_pixel(pos, sum);
}