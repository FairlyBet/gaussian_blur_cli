use crate::blur::RGBA_SIZE;
use glfw::Version;
use std::{ffi::CString, marker::PhantomData};

#[derive(Debug)]
#[repr(C)]
pub struct ImageData {
    pub offset: i32,
    pub width: i32,
    pub height: i32,
}

pub struct BlurProgram {
    program: u32,
    direction_location: i32,
    group_size: (u32, u32),
    _marker: PhantomData<*const ()>, // Neither Send nor Sync
}

impl BlurProgram {
    pub const MAX_BUFFER_SIZE: usize = i32::MAX as usize * RGBA_SIZE;
    pub const INPUT_BINDING_UNIT: u32 = 0;
    pub const OUTPUT_BINDING_UNIT: u32 = 1;
    pub const IMAGE_DATA_BINDING_POINT: u32 = 2;

    pub fn new(context_version: Version, group_size: (u32, u32), kernel: &[f32]) -> Option<Self> {
        let src = Self::src(context_version, group_size, kernel);
        let shader = ComputeShader::new(&src)?;
        // SAFETY:
        //
        unsafe {
            let program = gl::CreateProgram();
            gl::AttachShader(program, shader.shader);
            gl::LinkProgram(program);
            let mut is_linked = 0;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut is_linked);
            if is_linked == gl::FALSE as i32 {
                return None;
            }

            gl::UseProgram(program);
            let name = CString::new("direction").ok()?;
            let direction_location = gl::GetUniformLocation(program, name.as_ptr());

            Some(Self {
                program,
                direction_location,
                group_size,
                _marker: PhantomData,
            })
        }
    }

    pub fn src(context_version: Version, group_size: (u32, u32), kernel: &[f32]) -> String {
        const SRC: &str = r#"
uniform ivec2 direction;

vec4 fetch_pixel(ivec2 pos) {
    int x = pos.x;
    int y = pos.y * width;
    return imageLoad(input_image, offset + x + y);
}

void write_pixel(ivec2 pos, vec4 pixel) {
    int x = pos.x;
    int y = pos.y * width;
    imageStore(output_image, offset + x + y, pixel);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

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
"#;

        let version = format!(
            "#version {}{}{} core\n",
            context_version.major, context_version.minor, context_version.patch
        );
        let layout_data = format!(
            r#"
layout(local_size_x = {}, local_size_y = {}) in;
layout(binding = {}, rgba8) restrict readonly uniform imageBuffer input_image;
layout(binding = {}, rgba8) restrict writeonly uniform imageBuffer output_image;
layout(std140, binding = {}) uniform ImageData {{
    int offset;
    int width;
    int height;
}};
"#,
            group_size.0,
            group_size.1,
            Self::INPUT_BINDING_UNIT,
            Self::OUTPUT_BINDING_UNIT,
            Self::IMAGE_DATA_BINDING_POINT,
        );
        let kernel = Self::kernel_to_glsl(kernel);

        version + &layout_data + &kernel + SRC
    }

    fn kernel_to_glsl(kernel: &[f32]) -> String {
        let kernel_size = format!("const int KERNEL_SIZE = {};\n", kernel.len());
        let kernel_values: String = kernel
            .iter()
            .map(|item| {
                let mut str = item.to_string();
                str.push_str(", ");
                str
            })
            .collect();
        let kernel_values = kernel_values.trim_end_matches(", ");
        let kernel = format!(
            "const float[KERNEL_SIZE] KERNEL = float[KERNEL_SIZE]({});\n",
            kernel_values
        );

        kernel_size + &kernel
    }

    pub fn group_size(&self) -> (u32, u32) {
        self.group_size
    }

    pub fn use_(&self) {
        unsafe {
            gl::UseProgram(self.program);
        }
    }

    pub fn set_horizontal(&mut self) {
        self.use_();
        unsafe {
            gl::ProgramUniform2i(self.program, self.direction_location, 1, 0);
        }
    }

    pub fn set_vertical(&mut self) {
        self.use_();
        unsafe {
            gl::ProgramUniform2i(self.program, self.direction_location, 0, 1);
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

struct ComputeShader {
    shader: u32,
    _marker: PhantomData<*const ()>, // Neither Send nor Sync
}

impl ComputeShader {
    fn new(src: &str) -> Option<Self> {
        unsafe {
            let shader = gl::CreateShader(gl::COMPUTE_SHADER);
            if shader == 0 {
                return None;
            }
            let src = CString::new(src).ok()?;
            let src_len: i32 = src.count_bytes().try_into().ok()?;
            gl::ShaderSource(shader, 1, &src.as_ptr(), &src_len);
            gl::CompileShader(shader);
            let mut is_compiled = 0;
            gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut is_compiled);
            if is_compiled == gl::FALSE as i32 {
                let mut len = 0;
                gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
                let mut v: Vec<u8> = Vec::with_capacity(len.try_into().unwrap());
                let mut len_written = 0;
                gl::GetShaderInfoLog(
                    shader,
                    v.capacity().try_into().unwrap(),
                    &mut len_written,
                    v.as_mut_ptr().cast(),
                );
                v.set_len(len_written.try_into().unwrap());

                println!("{}", String::from_utf8_lossy(&v).into_owned());
                return None;
            }

            Some(Self {
                shader,
                _marker: PhantomData,
            })
        }
    }
}

impl Drop for ComputeShader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteShader(self.shader);
        }
    }
}
