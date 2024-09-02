use crate::{
    blur_program::BlurProgram,
    buffer::{ImageBuffer, UniformBuffer},
};
use glfw::{fail_on_errors, ClientApiHint, OpenGlProfileHint, WindowHint, WindowMode};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageFormat, Limits};
use std::{fs::File, io::BufReader};

#[derive(Debug)]
pub struct Renderer {}

impl Renderer {
    pub fn new(sigma: f32) -> Option<Self> {
        let mut glfw = glfw::init(fail_on_errors!()).ok()?;
        glfw.window_hint(WindowHint::ClientApi(ClientApiHint::OpenGl));
        glfw.window_hint(WindowHint::OpenGlProfile(OpenGlProfileHint::Core));
        glfw.window_hint(WindowHint::ContextVersion(4, 3));
        glfw.window_hint(WindowHint::Visible(false));
        let (mut w, _) = glfw.create_window(1, 1, "", WindowMode::Windowed)?;
        gl::load_with(|symbol| w.get_proc_address(symbol));
        let t = std::time::Instant::now();

        let decoder = get_decoder("sample.jpg").unwrap();
        let (width, height) = decoder.dimensions();
        let size = decoder.total_bytes() as usize;
        let original_color_type = decoder.original_color_type();

        let mut input_buffer = ImageBuffer::new_writable(size)?;
        let intermadiate_buffer = ImageBuffer::new(size)?;
        let output_buffer = ImageBuffer::new_readable(size)?;
        let mut uniform_buffer = UniformBuffer::<ImageData>::new()?;
        uniform_buffer.update(ImageData {
            offset: 0,
            width: width as i32,
            height: height as i32,
        });

        let kernel = gaussian_kernel(6 * sigma as usize + 1, sigma);
        let program = BlurProgram::new(w.get_context_version(), (2, 2), &kernel)?;
        program.use_();

        unsafe {
            decoder.read_image(input_buffer.data()).unwrap();
            println!("Ready for render: {}", t.elapsed().as_millis());
            uniform_buffer.bind_buffer_base(BlurProgram::UNIFORM_BINDING_POINT);

            input_buffer.bind_image_texture(BlurProgram::INPUT_BINDING_UNIT, gl::READ_ONLY, gl::R8);
            intermadiate_buffer.bind_image_texture(
                BlurProgram::OUTPUT_BINDING_UNIT,
                gl::WRITE_ONLY,
                gl::R8,
            );
            program.set_horizontal();
            gl::DispatchCompute(
                width.div_ceil(program.group_size().0),
                height.div_ceil(program.group_size().1),
                1,
            );

            intermadiate_buffer.bind_image_texture(
                BlurProgram::INPUT_BINDING_UNIT,
                gl::READ_ONLY,
                gl::R8,
            );
            output_buffer.bind_image_texture(
                BlurProgram::OUTPUT_BINDING_UNIT,
                gl::WRITE_ONLY,
                gl::R8,
            );
            program.set_vertical();
            gl::DispatchCompute(
                width.div_ceil(program.group_size().0),
                height.div_ceil(program.group_size().1),
                1,
            );

            gl::Finish();
            println!("Finished: {}", t.elapsed().as_millis());

            image::save_buffer(
                "test.jpg",
                output_buffer.data(),
                width,
                height,
                original_color_type,
            )
            .unwrap();
            println!("Image saved: {}", t.elapsed().as_millis());
        }

        Some(Self {})
    }

    fn fill_buffer_from_ohter_thread() {
        //
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

pub type Decoder = Box<dyn ImageDecoder>;

fn get_decoder(path: &str) -> anyhow::Result<Decoder> {
    use image::codecs::{
        bmp::BmpDecoder, jpeg::JpegDecoder, png::PngDecoder, tga::TgaDecoder, tiff::TiffDecoder,
        webp::WebPDecoder,
    };

    let format = ImageFormat::from_path(path)?;
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let decoder: Box<dyn ImageDecoder> = match format {
        ImageFormat::Png => Box::new(PngDecoder::with_limits(reader, Limits::default())?),
        ImageFormat::Jpeg => Box::new(JpegDecoder::new(reader)?),
        ImageFormat::WebP => Box::new(WebPDecoder::new(reader)?),
        ImageFormat::Tiff => Box::new(TiffDecoder::new(reader)?),
        ImageFormat::Tga => Box::new(TgaDecoder::new(reader)?),
        ImageFormat::Bmp => Box::new(BmpDecoder::new(reader)?),
        _ => return Err(anyhow::anyhow!("Unsupported image format")),
    };

    match decoder.color_type() {
        ColorType::Rgb8 | ColorType::Rgba8 => Ok(decoder),
        _ => Err(anyhow::anyhow!("Unsuppoted color format")),
    }
}

pub struct ImageInfo<'a> {
    path: &'a str,
    decoder: Decoder,
    width: u32,
    height: u32,
    size: u64,
    original_color_type: ExtendedColorType,
    color_type: ColorType,
}

impl<'a> ImageInfo<'a> {
    pub fn new(path: &'a str) -> anyhow::Result<Self> {
        let decoder = get_decoder(path)?;
        Ok(Self {
            path,
            width: decoder.dimensions().0,
            height: decoder.dimensions().1,
            size: decoder.total_bytes(),
            original_color_type: decoder.original_color_type(),
            color_type: decoder.color_type(),
            decoder,
        })
    }

    pub fn path(&self) -> &str {
        self.path
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn original_color_type(&self) -> ExtendedColorType {
        self.original_color_type
    }

    pub fn color_type(&self) -> ColorType {
        self.color_type
    }
}
