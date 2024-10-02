use crate::{
    buffers::unifrom_buffer::UniformBuffer,
    renderer::{ImageInfo, RGBA_SIZE},
};
use glfw::Version;
use std::{ffi::CString, marker::PhantomData};
use tracing::error;

#[derive(Debug)]
#[repr(C)]
pub struct ImageData {
    pub pixel_offset: i32,
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
    // Multiplied by `RGBA_SIZE` because every pixel is 4 byte long
    pub const MAX_BUFFER_SIZE: usize = i32::MAX as usize * RGBA_SIZE;
    pub const INPUT_BINDING_UNIT: u32 = 0;
    pub const OUTPUT_BINDING_UNIT: u32 = 1;
    pub const IMAGE_DATA_BINDING_POINT: u32 = 0;

    pub fn new(context_version: Version, group_size: (u32, u32), kernel: &[f32]) -> Option<Self> {
        let src = Self::src(context_version, group_size, kernel);
        let shader = ComputeShader::new(&src)?;
        // SAFETY:
        // This code is valid OpenGL calls
        // Returned from OpenGL values are checked to be
        // valid.
        unsafe {
            let program = gl::CreateProgram();
            if program == 0 {
                return None;
            }
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
            // if this panics then the shader src is incorrect
            assert_ne!(direction_location, -1);

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

    pub fn use_(&self) {
        // SAFETY:
        // It is safe as `self.program` is
        // valid program object id
        unsafe {
            gl::UseProgram(self.program);
        }
    }

    pub fn set_horizontal(&self) {
        self.use_();
        // SAFETY:
        // It is safe as location of `direction` uniform
        // and providing values are valid and corresponds to
        // shader program
        unsafe {
            gl::ProgramUniform2i(self.program, self.direction_location, 1, 0);
        }
    }

    pub fn set_vertical(&self) {
        self.use_();
        // SAFETY:
        // It is safe as location of `direction` uniform
        // and providing values are valid and corresponds to
        // shader program
        unsafe {
            gl::ProgramUniform2i(self.program, self.direction_location, 0, 1);
        }
    }

    /// ### SAFETY:
    ///
    /// The `use` method must be called before this method.
    /// Caller must ensure to correctly bind buffers
    /// with data that corresponds to information provided
    /// in `loaded_images`
    pub unsafe fn dispatch_compute(
        &self,
        loaded_images: &Vec<(ImageInfo, usize)>,
        image_data_buffer: &mut UniformBuffer<ImageData>,
    ) {
        for (img, offset) in loaded_images {
            image_data_buffer.update(ImageData {
                // Here we're dividing by RGBA_SIZE because we need offset
                // of the whole RGBA pixel and not the particular byte
                pixel_offset: (*offset / RGBA_SIZE) as i32,
                width: img.width as i32,
                height: img.height as i32,
            });

            // SAFETY:
            // Assumes that the program is being used
            // and the data used in shader is valid and provided correctly
            unsafe {
                gl::DispatchCompute(
                    img.width.div_ceil(self.group_size.0),
                    img.height.div_ceil(self.group_size.1),
                    1,
                );
            }
        }
    }
}

impl Drop for BlurProgram {
    fn drop(&mut self) {
        // SAFETY:
        // It is safe because `self.program` is
        // valid shader program object id
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
        // SAFETY:
        // This code is valid OpenGL calls
        // Returned from OpenGL values are checked to be
        // valid.
        let shader = unsafe {
            let shader = gl::CreateShader(gl::COMPUTE_SHADER);
            if shader == 0 {
                return None;
            }
            let src = CString::new(src).ok()?;
            let src_len = src.count_bytes().try_into().ok()?;
            gl::ShaderSource(shader, 1, &src.as_ptr(), &src_len);
            gl::CompileShader(shader);
            shader
        };

        // SAFETY:
        // This code is valid OpenGL calls.
        // It tries to receive info log about
        // shader compilation in case of unsuccessful
        // compilation. In order to do it `gl::GetShaderInfoLog`
        // has to receive pointer to buffer where info text will be written.
        // Amount of written bytes is set to `written_len` variable. All numeric values
        // are converted with `try_into` and also `written_len` is checked to be equal or less
        // than char buffer capacity. Then `chars` len is set to amount of written chars
        unsafe {
            let mut is_compiled = 0;
            gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut is_compiled);

            if is_compiled == gl::FALSE as i32 {
                let mut len = 0;
                gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
                let mut chars: Vec<u8> = Vec::with_capacity(len.try_into().unwrap());
                let mut written_len = 0;
                gl::GetShaderInfoLog(
                    shader,
                    chars.capacity().try_into().unwrap(),
                    &mut written_len,
                    chars.as_mut_ptr().cast(),
                );
                assert!(written_len <= len);
                chars.set_len(written_len.try_into().unwrap());
                error!("{}", String::from_utf8_lossy(&chars));

                return None;
            }
        }

        Some(Self {
            shader,
            _marker: PhantomData,
        })
    }
}

impl Drop for ComputeShader {
    fn drop(&mut self) {
        // SAFETY:
        // It is safe because `self.shader` is
        // valid shader object id
        unsafe {
            gl::DeleteShader(self.shader);
        }
    }
}
