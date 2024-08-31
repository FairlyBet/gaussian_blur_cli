use crate::{
    blur_program::BlurProgram,
    buffer::{ImageBuffer, ReadableImageBuffer, UniformBuffer, WritableImageBuffer},
};
use glfw::{fail_on_errors, ClientApiHint, OpenGlProfileHint, WindowHint, WindowMode};
use image::ExtendedColorType;

#[derive(Debug)]
pub struct Renderer {}

impl Renderer {
    pub fn new(sigma: f32) -> Option<Self> {
        let mut glfw = glfw::init(fail_on_errors!()).ok()?;
        glfw.window_hint(WindowHint::ClientApi(ClientApiHint::OpenGl));
        glfw.window_hint(WindowHint::OpenGlProfile(OpenGlProfileHint::Core));
        glfw.window_hint(WindowHint::ContextVersion(4, 5));
        glfw.window_hint(WindowHint::Visible(false));
        let (mut w, _) = glfw.create_window(1, 1, "", WindowMode::Windowed)?;
        gl::load_with(|symbol| w.get_proc_address(symbol));

        let t = std::time::Instant::now();
        let image = image::open("sample.jpg").unwrap();
        println!("Image open: {}", t.elapsed().as_millis());
        let b = image.blur(sigma);
        println!("CPU Finished: {}", t.elapsed().as_millis());
        b.save("test2.jpg").unwrap();
        
        // let bytes = image.as_rgb8().unwrap().as_raw();

        // let mut input_buffer = WritableImageBuffer::new(bytes.len())?;
        // let intermadiate_buffer = ImageBuffer::new(bytes.len())?;
        // let output_buffer = ReadableImageBuffer::new(bytes.len())?;
        // let mut uniform_buffer = UniformBuffer::<ImageData>::new()?;
        // uniform_buffer.update(ImageData {
        //     offset: 0,
        //     width: image.width() as i32,
        //     height: image.height() as i32,
        // });

        // let kernel = gaussian_kernel(6 * sigma as usize + 1, sigma);
        // let kernel = kernel_to_glsl(kernel);
        // let program = BlurProgram::new(&kernel, w.get_context_version())?;
        // program.use_();

        // unsafe {
        //     input_buffer.data().copy_from_slice(bytes);
        //     println!("Ready for render: {}", t.elapsed().as_millis());
        //     // .iter_mut()
        //     // .enumerate()
        //     // .for_each(|(i, item)| *item = i as u8);
        //     uniform_buffer.bind_buffer_base(BlurProgram::UNIFORM_BINDING_POINT);

        //     input_buffer.bind_image_texture(BlurProgram::INPUT_BINDING_UNIT, gl::READ_ONLY);
        //     intermadiate_buffer.bind_image_texture(BlurProgram::OUPUT_BINDING_UNIT, gl::WRITE_ONLY);
        //     program.set_horizontal();
        //     gl::DispatchCompute(
        //         image.width().div_ceil(BlurProgram::GROUP_SIZE.0),
        //         image.height().div_ceil(BlurProgram::GROUP_SIZE.1),
        //         1,
        //     );

        //     intermadiate_buffer.bind_image_texture(BlurProgram::INPUT_BINDING_UNIT, gl::READ_ONLY);
        //     output_buffer.bind_image_texture(BlurProgram::OUPUT_BINDING_UNIT, gl::WRITE_ONLY);
        //     program.set_vertical();
        //     gl::DispatchCompute(
        //         image.width().div_ceil(BlurProgram::GROUP_SIZE.0),
        //         image.height().div_ceil(BlurProgram::GROUP_SIZE.1),
        //         1,
        //     );
        //     gl::Finish();
        //     println!("Finished: {}", t.elapsed().as_millis());
        //     let error = gl::GetError();
        //     assert_eq!(error, gl::NO_ERROR);

        //     image::save_buffer(
        //         "test.jpg",
        //         output_buffer.data(),
        //         image.width(),
        //         image.height(),
        //         ExtendedColorType::Rgb8,
        //     )
        //     .unwrap();
        //     println!("Image saved: {}", t.elapsed().as_millis());
        // }

        Some(Self {})
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct ImageData {
    offset: i32,
    width: i32,
    height: i32,
}

fn gaussian_kernel(size: usize, sigma: f32) -> Vec<f32> {
    assert!(size % 2 == 1, "Size of the kernel should be odd.");

    let mut kernel = Vec::with_capacity(size);
    let half_size = (size / 2) as isize;
    let sigma2 = sigma * sigma;
    let normalization_factor = 1.0 / (2.0 * std::f32::consts::PI * sigma2).sqrt();
    let mut sum = 0.0;

    for i in -half_size..=half_size {
        let value = normalization_factor * (-((i * i) as f32) / (2.0 * sigma2)).exp();
        kernel.push(value);
        sum += value;
    }

    (0..size).for_each(|i| {
        kernel[i] /= sum;
    });

    kernel
}

fn kernel_to_glsl(kernel: Vec<f32>) -> String {
    let size = format!("const int KERNEL_SIZE = {};\n", kernel.len());
    let str: String = kernel
        .iter()
        .map(|item| {
            let mut str = item.to_string();
            str.push_str(", ");
            str
        })
        .collect();
    let str = str.trim_end_matches(", ");
    let array = format!(
        "const float[KERNEL_SIZE] KERNEL = float[KERNEL_SIZE]({});\n",
        str
    );
    size + &array
}
