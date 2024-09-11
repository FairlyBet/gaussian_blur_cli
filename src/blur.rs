use crate::{
    blur_program::{BlurProgram, ImageData},
    buffer::{ImageBuffer, UniformBuffer},
};
use anyhow::{anyhow, Result};
use clap::Parser;
use glfw::{
    fail_on_errors, ClientApiHint, GlfwReceiver, OpenGlProfileHint, PWindow, WindowEvent,
    WindowHint, WindowMode,
};
use image::{
    codecs::{
        bmp::BmpDecoder, jpeg::JpegDecoder, png::PngDecoder, tga::TgaDecoder, tiff::TiffDecoder,
        webp::WebPDecoder,
    },
    ColorType, ImageDecoder, ImageFormat, Limits, Rgb, RgbImage, Rgba, RgbaImage,
};
use rayon::{
    iter::{IndexedParallelIterator as _, ParallelIterator as _},
    slice::ParallelSliceMut as _,
};
use std::{
    error::Error,
    f32,
    fmt::Display,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    result,
    sync::Arc,
};

const RGB_SIZE: usize = 3;
const RGBA_SIZE: usize = 4;

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
        // The subsequent code is valid and safe
        unsafe {
            gl::GetInteger64v(gl::MAX_TEXTURE_BUFFER_SIZE, &mut max_texture_buffer_size);
        }
        let max_image_size = (max_texture_buffer_size as usize).min(BlurProgram::MAX_BUFFER_SIZE);

        Some(Self {
            window,
            receiver,
            max_image_size,
        })
    }

    pub fn max_image_size(&self) -> usize {
        self.max_image_size
    }

    pub fn process(&self, mut images: Vec<Arc<str>>, config: &Config) -> Result<()> {
        let kernel = gaussian_kernel(config.sigma);
        let program = BlurProgram::new(
            self.window.get_context_version(),
            (config.group_size, config.group_size),
            &kernel,
        )
        .ok_or(anyhow!("Cannot create shader program"))?;
        program.use_();

        let mut image_data_buffer =
            UniformBuffer::<ImageData>::new().ok_or(anyhow!("Cannot create uniform buffer"))?;
        image_data_buffer.bind_buffer_base(BlurProgram::IMAGE_DATA_BINDING_POINT);

        let mut input_buffer = ImageBuffer::new_writable(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"))?;
        let intermediate_buffer = ImageBuffer::new(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"))?;
        let output_buffer = ImageBuffer::new_readable(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"))?;

        while !images.is_empty() {
            // SAFETY:
            // It is safe because buffer usage is only writing
            // and it is not being used by GPU at this moment
            let input_slice = unsafe { input_buffer.data() };
            let loaded_images = self.load_images(&mut images, input_slice);
            for (img, offset) in &loaded_images {
                image_data_buffer.update(ImageData {
                    offset: (*offset / RGBA_SIZE) as i32,
                    width: img.width as i32,
                    height: img.height as i32,
                });

                // SAFETY:
                unsafe {
                    input_buffer.bind_image_texture(
                        BlurProgram::INPUT_BINDING_UNIT,
                        gl::READ_ONLY,
                        gl::RGBA8,
                    );
                    intermediate_buffer.bind_image_texture(
                        BlurProgram::OUTPUT_BINDING_UNIT,
                        gl::WRITE_ONLY,
                        gl::RGBA8,
                    );
                    program.set_horizontal();
                    gl::DispatchCompute(
                        img.width.div_ceil(program.group_size().0),
                        img.height.div_ceil(program.group_size().1),
                        1,
                    );

                    intermediate_buffer.bind_image_texture(
                        BlurProgram::INPUT_BINDING_UNIT,
                        gl::READ_ONLY,
                        gl::RGBA8,
                    );
                    output_buffer.bind_image_texture(
                        BlurProgram::OUTPUT_BINDING_UNIT,
                        gl::WRITE_ONLY,
                        gl::RGBA8,
                    );
                    program.set_vertical();
                    gl::DispatchCompute(
                        img.width.div_ceil(program.group_size().0),
                        img.height.div_ceil(program.group_size().1),
                        1,
                    );
                }
            }
            unsafe {
                let t = std::time::Instant::now();
                gl::Finish();
                println!("Finished: {}", t.elapsed().as_millis());
            }
            let t = std::time::Instant::now();
            Self::save_images(
                unsafe { output_buffer.data() },
                &loaded_images,
                &config.output_dir,
            );
            println!("Saved: {}", t.elapsed().as_millis());
            _ = glfw::flush_messages(&self.receiver);
        }

        Ok(())
    }

    fn load_images(
        &self,
        images: &mut Vec<Arc<str>>,
        mut input_slice: &mut [u8],
    ) -> Vec<(ImageInfo, usize)> {
        let mut loaded_images = vec![];
        let mut offset = 0;
        let mut i = 0;
        while i < images.len() {
            let path = images[i].clone();
            match self.try_load(path.clone(), input_slice) {
                Ok(info) => {
                    input_slice = input_slice.split_at_mut(info.rgba_size).1;
                    let size = info.rgba_size;
                    loaded_images.push((info, offset));
                    offset += size;
                    _ = images.remove(i);
                }
                Err(err) => match err {
                    LoadError::TooLargeImage => {
                        // replace with log
                        eprintln!("{} does not fit into working buffer", path);
                        _ = images.remove(i);
                    }
                    LoadError::DecoderError(e) => {
                        // replace with log
                        eprintln!("{e}");
                        _ = images.remove(i);
                    }
                    LoadError::NoSpaceLeft => i += 1,
                },
            }
        }
        loaded_images
    }

    fn try_load(&self, path: Arc<str>, buffer: &mut [u8]) -> result::Result<ImageInfo, LoadError> {
        match get_decoder(&path) {
            Ok(decoder) => {
                let size = if decoder.color_type() == ColorType::Rgb8 {
                    self::rgb_size_to_rgba_size(decoder.total_bytes())
                        .ok_or(LoadError::TooLargeImage)?
                } else {
                    decoder.total_bytes()
                };
                let size: usize = size.try_into().map_err(|_| LoadError::TooLargeImage)?;
                if size > self.max_image_size {
                    return Err(LoadError::TooLargeImage);
                }
                if size > buffer.len() {
                    return Err(LoadError::NoSpaceLeft);
                }

                let image_info = ImageInfo {
                    path,
                    width: decoder.dimensions().0,
                    height: decoder.dimensions().1,
                    color_type: decoder.color_type(),
                    rgba_size: size,
                };

                if let Err(e) = Self::read_image(decoder, &mut buffer[..size]) {
                    return Err(LoadError::DecoderError(e));
                }
                Ok(image_info)
            }
            Err(e) => Err(LoadError::DecoderError(e)),
        }
    }

    fn read_image(decoder: Decoder, buffer: &mut [u8]) -> Result<()> {
        match decoder.color_type() {
            ColorType::Rgb8 => {
                let size = decoder.total_bytes() as usize;
                let mut image_buf = Vec::with_capacity(size);
                // SAFETY:
                // This is safe as `image_buf` is created with capacity of `size`
                unsafe { image_buf.set_len(size) }

                decoder.read_image(&mut image_buf)?;

                // This code copies image to GPU memory converting RGB to RGBA
                // without setting the Alpha component, so it is
                // some random value from memory
                buffer
                    .par_chunks_exact_mut(RGBA_SIZE)
                    .enumerate()
                    .for_each(|(i, pixel)| {
                        let pos = i * RGB_SIZE;
                        pixel[..RGB_SIZE].copy_from_slice(&image_buf[pos..pos + RGB_SIZE]);
                    });
            }
            ColorType::Rgba8 => decoder.read_image(buffer)?,
            _ => unreachable!(),
        }
        Ok(())
    }

    fn save_images(buffer: &[u8], infos: &[(ImageInfo, usize)], output_dir: &Path) {
        for (info, offset) in infos {
            // Replace with `filename()`
            let filename = sanitize_filename::sanitize(&*info.path);
            let mut path = output_dir.to_path_buf();
            path.push(filename);
            match info.color_type {
                ColorType::Rgb8 => {
                    let img = RgbImage::from_par_fn(info.width, info.height, |x, y| {
                        let pos = offset + (x + y * info.width) as usize * RGBA_SIZE;
                        Rgb([buffer[pos], buffer[pos + 1], buffer[pos + 2]])
                    });
                    if let Err(e) = img.save(&path) {
                        eprintln!("{e}");
                    }
                }
                ColorType::Rgba8 => {
                    let img = RgbaImage::from_par_fn(info.width, info.height, |x, y| {
                        let pos = offset + (x + y * info.width) as usize * RGBA_SIZE;
                        Rgba([
                            buffer[pos],
                            buffer[pos + 1],
                            buffer[pos + 2],
                            buffer[pos + 3],
                        ])
                    });
                    if let Err(e) = img.save(&path) {
                        eprintln!("{e}");
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

#[derive(Debug)]
pub enum LoadError {
    TooLargeImage,
    NoSpaceLeft,
    DecoderError(anyhow::Error),
}

impl Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            LoadError::TooLargeImage => "Image is too large",
            LoadError::NoSpaceLeft => "No space left in buffer",
            LoadError::DecoderError(e) => &e.to_string(),
        };
        write!(f, "{str}")
    }
}

impl Error for LoadError {}

fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let size = 6 * sigma as usize + 1;
    let mut kernel = Vec::with_capacity(size);
    let half_size = (size / 2) as isize;
    let sigma2 = sigma * sigma;
    let normalization_factor = 1.0 / (2.0 * f32::consts::PI * sigma2).sqrt();
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

fn rgb_size_to_rgba_size(rbg_size: u64) -> Option<u64> {
    rbg_size.checked_add(rbg_size / 3)
}

fn get_decoder(path: &str) -> Result<Decoder> {
    let format = ImageFormat::from_path(path)?;
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let decoder: Decoder = match format {
        ImageFormat::Png => Box::new(PngDecoder::with_limits(reader, Limits::default())?),
        ImageFormat::Jpeg => Box::new(JpegDecoder::new(reader)?),
        ImageFormat::WebP => Box::new(WebPDecoder::new(reader)?),
        ImageFormat::Tiff => Box::new(TiffDecoder::new(reader)?),
        ImageFormat::Tga => Box::new(TgaDecoder::new(reader)?),
        ImageFormat::Bmp => Box::new(BmpDecoder::new(reader)?),
        _ => return Err(anyhow!("{path} has unsupported image format")),
    };

    match decoder.color_type() {
        ColorType::Rgb8 | ColorType::Rgba8 => Ok(decoder),
        _ => Err(anyhow!("{path} has unsupported color type")),
    }
}

pub type Decoder = Box<dyn ImageDecoder + Send + Sync>;

#[derive(Debug)]
pub struct ImageInfo {
    // Replace with `Arc<Path>`
    path: Arc<str>,
    width: u32,
    height: u32,
    color_type: ColorType,
    rgba_size: usize,
}

#[derive(Debug, Parser)]
pub struct Config {
    #[arg(short)]
    pub working_buffer_size: usize,
    #[arg(short)]
    pub group_size: u32,
    #[arg(short)]
    pub sigma: f32,
    #[arg(short)]
    pub output_dir: PathBuf,
}
