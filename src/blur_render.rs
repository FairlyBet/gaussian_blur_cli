use crate::{
    blur_program::BlurProgram,
    buffer::{ImageBuffer, PersistentRead, PersistentWrite, Regular, UniformBuffer},
};
use anyhow::anyhow;
use glfw::{
    fail_on_errors, ClientApiHint, GlfwReceiver, OpenGlProfileHint, PWindow, WindowEvent,
    WindowHint, WindowMode,
};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageFormat, Limits, Rgb};
use std::{fs::File, io::BufReader};

#[derive(Debug)]
pub struct Renderer {
    window: PWindow,
    receiver: GlfwReceiver<(f64, WindowEvent)>,
    max_image_size: usize,
}

impl Renderer {
    pub fn new() -> Option<Self> {
        let mut glfw = glfw::init(fail_on_errors!()).ok()?;
        glfw.window_hint(WindowHint::ClientApi(ClientApiHint::OpenGl));
        glfw.window_hint(WindowHint::OpenGlProfile(OpenGlProfileHint::Core));
        glfw.window_hint(WindowHint::ContextVersion(4, 3));
        glfw.window_hint(WindowHint::Visible(false));
        let (mut window, receiver) = glfw.create_window(1, 1, "", WindowMode::Windowed)?;
        gl::load_with(|symbol| window.get_proc_address(symbol));

        let mut max_texture_buffer_size = 0;
        // SAFETY:
        // The subsequent code is valid and therefore safe
        unsafe {
            gl::GetInteger64v(gl::MAX_TEXTURE_BUFFER_SIZE, &mut max_texture_buffer_size);
        }
        let max_image_size = (max_texture_buffer_size as usize).min(BlurProgram::MAX_BUFFER_SIZE);

        Some(Self {
            window,
            receiver,
            max_image_size,
        })

        // let image = image::open("sample.webp").unwrap();
        // let image = image.as_rgb8().unwrap();
        // let (width, height) = image.dimensions();
        // let bytes: &[u8] = image.as_raw();
        // // let size = bytes.len();
        // let size = (width * height * 4) as usize;

        // let decoder = get_decoder("sample.webp").unwrap();
        // let (width, height) = decoder.dimensions();
        // let size = decoder.total_bytes() as usize;
        // let original_color_type = decoder.original_color_type();

        // let mut input_buffer = ImageBuffer::new_writable(size)?;
        // let intermadiate_buffer = ImageBuffer::new(size)?;
        // let output_buffer = ImageBuffer::new_readable(size)?;
        // let mut uniform_buffer = UniformBuffer::<ImageData>::new()?;
        // uniform_buffer.update(ImageData {
        //     offset: 0,
        //     width: width as i32,
        //     height: height as i32,
        // });

        // let kernel = gaussian_kernel(6 * sigma as usize + 1, sigma);
        // let program = BlurProgram::new(w.get_context_version(), (16, 16), &kernel)?;
        // program.use_();

        // unsafe {
        //     let mut i = 0;
        //     bytes.chunks_exact(3).for_each(|pixel| {
        //         input_buffer.data()[i..i + 3].copy_from_slice(pixel);
        //         i += 4;
        //     });
        //     println!("Ready for render: {}", t.elapsed().as_millis());

        //     uniform_buffer.bind_buffer_base(BlurProgram::UNIFORM_BINDING_POINT);

        //     input_buffer.bind_image_texture(
        //         BlurProgram::INPUT_BINDING_UNIT,
        //         gl::READ_ONLY,
        //         gl::RGBA8,
        //     );
        //     intermadiate_buffer.bind_image_texture(
        //         BlurProgram::OUTPUT_BINDING_UNIT,
        //         gl::WRITE_ONLY,
        //         gl::RGBA8,
        //     );
        //     program.set_horizontal();
        //     gl::DispatchCompute(
        //         width.div_ceil(program.group_size().0),
        //         height.div_ceil(program.group_size().1),
        //         1,
        //     );

        //     intermadiate_buffer.bind_image_texture(
        //         BlurProgram::INPUT_BINDING_UNIT,
        //         gl::READ_ONLY,
        //         gl::RGBA8,
        //     );
        //     output_buffer.bind_image_texture(
        //         BlurProgram::OUTPUT_BINDING_UNIT,
        //         gl::WRITE_ONLY,
        //         gl::RGBA8,
        //     );
        //     program.set_vertical();
        //     gl::DispatchCompute(
        //         width.div_ceil(program.group_size().0),
        //         height.div_ceil(program.group_size().1),
        //         1,
        //     );

        //     gl::Finish();
        //     println!("Finished: {}", t.elapsed().as_millis());

        //     let image = image::RgbImage::from_fn(width, height, |x, y| {
        //         let index = (x * 4 + y * width * 4) as usize;
        //         let pixel = &output_buffer.data()[index..index + 3];
        //         Rgb([pixel[0], pixel[1], pixel[2]])
        //     });
        //     image.save("test.webp").unwrap();
        //     println!("Image saved: {}", t.elapsed().as_millis());
        // }
    }

    pub fn max_image_size(&self) -> usize {
        self.max_image_size
    }

    pub fn process(&self, images: &[&str], config: &Config) -> anyhow::Result<()> {
        let mut infos = Vec::new();
        for image in images {
            match ImageInfo::new(image, self) {
                Ok(image_info) => {
                    infos.push(image_info);
                }
                Err(e) => {
                    // log error
                    eprintln!("{e}");
                }
            }
        }

        let buf1 = WorkingBuffer::new(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"));
        let buf2 = WorkingBuffer::new(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"));
        let intermadiate_buffer = ImageBuffer::new(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"));

        // let slice = &mut [1, 2, 3, 4, 5];
        // slice.split_at_mut(mid)
        // let v = vec![&mut slice[0..1], &mut slice[1..2], &mut slice[2..]];
        // std::thread::scope(|scope| {
        // for s in v {
        //         scope.spawn(|| {
        //             s[0] = 10;
        //         });
        //     }
        // });

        Ok(())
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
    fn get_decoder(path: &str) -> anyhow::Result<Decoder> {
        use image::codecs::{
            bmp::BmpDecoder, jpeg::JpegDecoder, png::PngDecoder, tga::TgaDecoder,
            tiff::TiffDecoder, webp::WebPDecoder,
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
            _ => return Err(anyhow!("Unsupported image format")),
        };

        match decoder.color_type() {
            ColorType::Rgb8 | ColorType::Rgba8 => Ok(decoder),
            _ => Err(anyhow!("Unsuppoted color format")),
        }
    }

    pub fn new(path: &'a str, renderer: &Renderer) -> anyhow::Result<Self> {
        let decoder = Self::get_decoder(path)?;
        let color_type = decoder.color_type();

        let err = format!("{} doesn't fit into working buffer", path);
        let mut size: usize = decoder
            .total_bytes()
            .try_into()
            .map_err(|_| anyhow!(err.clone()))?;
        if color_type == ColorType::Rgb8 {
            size = size.checked_add(size / 3).ok_or(anyhow!(err.clone()))?;
        }

        if size > renderer.max_image_size {
            return Err(anyhow!(err));
        }

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

pub struct Config {
    working_buffer_size: usize,
    group_size: (u32, u32),
    sigma: f32,
}

pub enum GroupSize {
    G2x2,
    G4x4,
    G8x8,
    G16x16,
    G32x32,
}

impl GroupSize {
    fn group_size(&self) -> (u32, u32) {
        match self {
            GroupSize::G2x2 => (2, 2),
            GroupSize::G4x4 => (4, 4),
            GroupSize::G8x8 => (8, 8),
            GroupSize::G16x16 => (16, 16),
            GroupSize::G32x32 => (32, 32),
        }
    }
}

struct WorkingBuffer {
    input_buffer: ImageBuffer<PersistentWrite>,
    output_buffer: ImageBuffer<PersistentRead>,
}

impl WorkingBuffer {
    fn new(size: usize) -> Option<Self> {
        let input_buffer = ImageBuffer::new_writable(size)?;
        let output_buffer = ImageBuffer::new_readable(size)?;
        Some(Self {
            input_buffer,
            output_buffer,
        })
    }
}
