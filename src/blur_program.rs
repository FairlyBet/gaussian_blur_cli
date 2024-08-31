use glfw::Version;
use std::{ffi::CString, marker::PhantomData};

pub struct BlurProgram {
    program: u32,
    direction_location: i32,
    _marker: PhantomData<*const ()>, // Neither Send and Sync
}

impl BlurProgram {
    pub const GROUP_SIZE: (u32, u32) = (32, 32);
    pub const INPUT_BINDING_UNIT: u32 = 0;
    pub const OUPUT_BINDING_UNIT: u32 = 1;
    pub const UNIFORM_BINDING_POINT: u32 = 2;

    const SRC: &'static str = r#"
uniform ivec2 direction;

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

    if (pos.x >= width || pos.y >= height) return;

    vec3 sum = vec3(0.0);
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

    pub fn new(kernel: &str, context_version: Version) -> Option<Self> {
        let src = Self::src(context_version, kernel);
        let shader = ComputeShader::new(&src)?;
        unsafe {
            let program = gl::CreateProgram();
            gl::AttachShader(program, shader.shader);
            gl::LinkProgram(program);
            let mut success = 0;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);

            if success == gl::FALSE as i32 {
                return None;
            }

            gl::UseProgram(program);
            let direction = CString::new("direction").ok()?;
            let direction_location = gl::GetUniformLocation(program, direction.as_ptr());

            Some(Self {
                program,
                direction_location,
                _marker: PhantomData,
            })
        }
    }

    pub fn src(context_version: Version, kernel: &str) -> String {
        let version = format!(
            "#version {}{}{} core\n",
            context_version.major, context_version.minor, context_version.patch
        );
        let layout_data = format!(
            r#"
layout(local_size_x = {}, local_size_y = {}) in;
layout(binding = {}, r8) readonly uniform imageBuffer input_image;
layout(binding = {}, r8) writeonly uniform imageBuffer output_image;
layout(std140, binding = {}) uniform ImageData {{
    int offset;
    int width;
    int height;
}};
"#,
            Self::GROUP_SIZE.0,
            Self::GROUP_SIZE.1,
            Self::INPUT_BINDING_UNIT,
            Self::OUPUT_BINDING_UNIT,
            Self::UNIFORM_BINDING_POINT,
        );

        version + &layout_data + kernel + Self::SRC
    }

    pub fn use_(&self) {
        unsafe {
            gl::UseProgram(self.program);
        }
    }

    pub fn set_horizontal(&self) {
        self.use_();
        unsafe {
            gl::ProgramUniform2i(self.program, self.direction_location, 1, 0);
        }
    }

    pub fn set_vertical(&self) {
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
    _marker: PhantomData<*const ()>, // Neither Send and Sync
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

impl Drop for ComputeShader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteShader(self.shader);
        }
    }
}
