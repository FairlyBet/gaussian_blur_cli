#version 450 core

// layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 0, r8) readonly uniform imageBuffer input_image;
layout(binding = 1, r8) writeonly uniform imageBuffer output_image;
layout(std140, binding = 2) uniform ImageData {
int offset;
int width;
int height;
};

const int RGB = 3;
const int KERNEL_SIZE = 33;
const float[KERNEL_SIZE] KERNEL = float[KERNEL_SIZE](0.012308157, 0.014371717, 0.016614275, 0.019015647, 0.021547552, 0.024173625, 0.026849903, 0.029525734, 0.03214517, 0.03464877, 0.03697575, 0.039066378, 0.04086452, 0.042320102, 0.04339144, 0.044047218, 0.044268005, 0.044047218, 0.04339144, 0.042320102, 0.04086452, 0.039066378, 0.03697575, 0.03464877, 0.03214517, 0.029525734, 0.026849903, 0.024173625, 0.021547552, 0.019015647, 0.016614275, 0.014371717, 0.012308157);

ivec3 get_indecies(ivec2 pos) {
int x = pos.x * RGB;
int y = pos.y * width * RGB;

int r_index = x + y;
int g_index = x + y + 1;
int b_index = x + y + 2;

return ivec3(r_index, g_index, b_index);
}

vec3 fetch_pixel(ivec2 pos) {
ivec3 indecies = get_indecies(pos);

float r = imageLoad(input_image, indecies.r).r;
float g = imageLoad(input_image, indecies.g).r;
float b = imageLoad(input_image, indecies.b).r;

return vec3(r, g, b);
}

void write_pixel(ivec2 pos, vec3 pixel) {
ivec3 indecies = get_indecies(pos);

imageStore(output_image, indecies.r, vec4(pixel.r));
imageStore(output_image, indecies.g, vec4(pixel.g));
imageStore(output_image, indecies.b, vec4(pixel.b));
}

void main() {
    // ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    // vec3 pixel = fetch_pixel(pos);
    // write_pixel(pos, pixel);
}