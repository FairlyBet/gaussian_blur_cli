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
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec3 pixel = fetch_pixel(pos);
    write_pixel(pos, pixel);
}