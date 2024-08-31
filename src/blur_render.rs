use crate::buffer::{ReadableImageBuffer, UniformBuffer, WritableImageBuffer};
use std::{ffi::CString, marker::PhantomData};

#[derive(Debug)]
pub struct Renderer {}

impl Renderer {
    pub fn new() -> Option<Self> {
        let blur_program = BlurProgram::new()?;
        blur_program.use_();
        let mut input_buffer = WritableImageBuffer::new(64)?;
        let output_buffer = ReadableImageBuffer::new(64)?;

        #[derive(Debug, Clone, Copy)]
        #[repr(C)]
        struct ImageData {
            offset: i32,
            width: i32,
            height: i32,
        }

        let mut uniform_buffer = UniformBuffer::<ImageData>::new()?;
        uniform_buffer.update(ImageData {
            offset: 0,
            width: 4,
            height: 4,
        });
        unsafe {
            input_buffer
                .data()
                .iter_mut()
                .enumerate()
                .for_each(|(i, item)| *item = i as u8);
            input_buffer.bind_image_texture(0, gl::READ_ONLY);
            output_buffer.bind_image_texture(1, gl::WRITE_ONLY);
            uniform_buffer.bind_buffer_base(2);
            gl::DispatchCompute(1, 1, 1);
            gl::Finish();
            println!("{:?}", output_buffer.data());
        }

        Some(Self {})
    }
}

struct BlurShader {
    shader: u32,
    _marker: PhantomData<*const ()>, // Neither Send and Sync
}

impl BlurShader {
    const SRC: &'static str = r#"
#version 450 core

layout(local_size_x = 8, local_size_y = 8) in;
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
"#;

    pub const GROUP_SIZE: (usize, usize) = (8, 8);

    fn new() -> Option<Self> {
        unsafe {
            let shader = gl::CreateShader(gl::COMPUTE_SHADER);

            if shader == 0 {
                return None;
            }

            let src = CString::new(Self::SRC).ok()?;
            gl::ShaderSource(shader, 1, &src.as_ptr(), &(src.count_bytes() as i32));
            gl::CompileShader(shader);
            let mut compiled = 0;
            gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut compiled);

            if compiled == gl::FALSE as i32 {
                return None;
            }

            Some(Self {
                shader,
                _marker: PhantomData,
            })
        }
    }
}

impl Drop for BlurShader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteShader(self.shader);
        }
    }
}

struct BlurProgram {
    program: u32,
    _marker: PhantomData<*const ()>, // Neither Send and Sync
}

impl BlurProgram {
    fn new() -> Option<Self> {
        let shader = BlurShader::new()?;
        unsafe {
            let program = gl::CreateProgram();
            gl::AttachShader(program, shader.shader);
            gl::LinkProgram(program);
            let mut success = 0;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);

            if success == gl::FALSE as i32 {
                return None;
            }

            Some(Self {
                program,
                _marker: PhantomData,
            })
        }
    }

    fn use_(&self) {
        unsafe {
            gl::UseProgram(self.program);
        }
    }
}

impl Drop for BlurProgram {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.program);
        }
    }
}
