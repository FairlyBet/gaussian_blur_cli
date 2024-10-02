use crate::{
    blur_program::{BlurProgram, ImageData},
    buffer::{ImageBuffer, UniformBuffer},
    Args,
};
use anyhow::{anyhow, Result};
use clap::Parser;
use config::Environment;
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
use serde::Deserialize;
use std::{
    env,
    error::Error,
    f32,
    ffi::OsStr,
    fmt::{Display, Formatter},
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    result,
    sync::Arc,
};
use tracing::error;

pub const RGB_SIZE: usize = 3;
pub const RGBA_SIZE: usize = 4;

#[derive(Debug)]
pub struct Renderer {
    window: PWindow,
    receiver: GlfwReceiver<(f64, WindowEvent)>,
    max_buffer_size: usize,
}

impl Renderer {
    pub fn new() -> Option<Self> {
        let mut glfw = glfw::init(fail_on_errors!()).ok()?;
        glfw.window_hint(WindowHint::ClientApi(ClientApiHint::OpenGl));
        glfw.window_hint(WindowHint::OpenGlProfile(OpenGlProfileHint::Core));
        glfw.window_hint(WindowHint::ContextVersion(4, 4));
        glfw.window_hint(WindowHint::Visible(false));
        let (mut window, receiver) = glfw.create_window(1, 1, "", WindowMode::Windowed)?;
        gl::load_with(|symbol| window.get_proc_address(symbol));

        let mut max_texture_buffer_size = 0;
        // SAFETY:
        // The subsequent code is safe as a providing pointer is created from
        // a valid reference to a local variable
        unsafe {
            gl::GetInteger64v(gl::MAX_TEXTURE_BUFFER_SIZE, &mut max_texture_buffer_size);
        }
        let max_buffer_size = (max_texture_buffer_size as usize).min(BlurProgram::MAX_BUFFER_SIZE);

        Some(Self {
            window,
            receiver,
            max_buffer_size,
        })
    }

    pub fn max_image_resolution(&self) -> usize {
        self.max_buffer_size / RGBA_SIZE
    }

    pub fn max_buffer_size(&self) -> usize {
        self.max_buffer_size
    }

    pub fn process(&self, mut images: Vec<Arc<Path>>, config: &Config) -> Result<()> {
        let kernel = gaussian_kernel(config.sigma);
        let program = BlurProgram::new(
            self.window.get_context_version(),
            (config.group_size, config.group_size),
            &kernel,
        )
        .ok_or(anyhow!("Cannot create shader program"))?;
        program.use_();

        let mut image_data =
            UniformBuffer::<ImageData>::new().ok_or(anyhow!("Cannot create uniform buffer"))?;
        image_data.bind_buffer_base(BlurProgram::IMAGE_DATA_BINDING_POINT);

        let mut input_buffer = ImageBuffer::new_writable(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"))?;
        let intermediate_buffer = ImageBuffer::new(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"))?;
        let output_buffer = ImageBuffer::new_readable(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"))?;

        while !images.is_empty() {
            // SAFETY:
            // It is safe because buffer is used only for writing,
            // and it is not being used by OpenGL at this moment
            let buffer = unsafe { input_buffer.data() };
            let loaded_images = self.load_images(&mut images, buffer);

            // SAFETY:
            // It is safe as providing `access` and `format`
            // values are correct and corresponds to buffers usage
            // as intended in the shader program.
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
            }
            program.set_horizontal();
            // SAFETY:
            // It is safe as all the data used by the
            // shader is valid and set correctly and the `use`
            // method is called. Also `loaded_images` corresponds
            // to data in image buffers that are bound at the moment
            unsafe {
                program.dispatch_compute(&loaded_images, &mut image_data);
            }

            // SAFETY:
            // It is safe as providing `access` and `format`
            // values are correct and corresponds to buffers usage
            // as intended in the shader program.
            unsafe {
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
            }
            program.set_vertical();
            // SAFETY:
            // It is safe as all the data used by the
            // shader is valid and set correctly and the `use`
            // method is called. Also `loaded_images` info corresponds
            // to data in image buffer that are bound at the moment
            unsafe {
                program.dispatch_compute(&loaded_images, &mut image_data);
            }

            // SAFETY:
            // This call is basically safe and lead to
            // no error or UB.
            // At this moment the GPU command buffer is
            // filled with render commands, so we have to
            // wait until all the job is done before proceeding to
            // image saving
            unsafe {
                gl::Finish();
            }

            Self::save_images(
                // SAFETY:
                // The buffer is not being used
                // by OpenGL at this moment os it is
                // safe to read from it
                unsafe { output_buffer.data() },
                &loaded_images,
                &config.output_dir,
            );

            _ = glfw::flush_messages(&self.receiver);
        }

        Ok(())
    }

    fn load_images(
        &self,
        images: &mut Vec<Arc<Path>>,
        mut buffer: &mut [u8],
    ) -> Vec<(ImageInfo, usize)> {
        let working_buffer_size = buffer.len();
        let mut loaded_images = vec![];
        let mut offset = 0;
        let mut i = 0;
        // This loop tries to load as many images as possible
        // into working buffer at once. If image size is bigger
        // than size of the entire buffer then error message is printed and image
        // is removed from input list and won't be processed. Otherwise, it is either loaded
        // or if the buffer is too full and can't hold the image at the moment,
        // then it is skipped until next iteration of this function
        while i < images.len() {
            let path = images[i].clone();
            match Self::try_load(path.clone(), buffer, working_buffer_size) {
                Ok(info) => {
                    // If image is loaded successfully then the offset of this
                    // image in buffer is calculated and saved alongside with
                    // the image info
                    buffer = buffer.split_at_mut(info.rgba_size).1;
                    let size = info.rgba_size;
                    loaded_images.push((info, offset));
                    offset += size;
                    _ = images.remove(i);
                }
                Err(err) => match err {
                    LoadError::TooLargeImage => {
                        error!("{path:?} does not fit into working buffer");
                        _ = images.remove(i);
                    }
                    LoadError::DecoderError(e) => {
                        error!("Can't decode image at {path:?}: {e}");
                        _ = images.remove(i);
                    }
                    LoadError::NoSpaceLeft => i += 1,
                },
            }
        }
        loaded_images
    }

    fn try_load(
        path: Arc<Path>,
        buffer: &mut [u8],
        working_buffer_size: usize,
    ) -> result::Result<ImageInfo, LoadError> {
        match get_decoder(&path) {
            Ok(decoder) => {
                let size = match decoder.color_type() {
                    ColorType::Rgb8 => self::rgb_size_to_rgba_size(decoder.total_bytes())
                        .ok_or(LoadError::TooLargeImage)?,
                    ColorType::Rgba8 => decoder.total_bytes(),
                    _ => unreachable!(),
                };
                let size: usize = size.try_into().map_err(|_| LoadError::TooLargeImage)?;
                if size > working_buffer_size {
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
                // Assumes that conversion is valid
                // and rgba-size of image and size of buffer are equal
                let size = decoder.total_bytes() as usize;
                // Creating an uninitialized buffer for image reading
                let mut image_buf = Vec::with_capacity(size);
                // SAFETY:
                // This is safe as `image_buf` is created with capacity of `size`
                #[allow(clippy::uninit_vec)]
                unsafe {
                    image_buf.set_len(size);
                }

                decoder.read_image(&mut image_buf)?;

                // This code copies image to GPU memory effectively
                // converting RGB to RGBA without setting the Alpha
                // component, so it is some random value from memory
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
            let filename = info.path.file_name().unwrap_or(OsStr::new("None"));
            let mut path = output_dir.to_path_buf();
            path.push(filename);
            match info.color_type {
                ColorType::Rgb8 => {
                    let img = RgbImage::from_par_fn(info.width, info.height, |x, y| {
                        let pos = offset + (x + y * info.width) as usize * RGBA_SIZE;
                        Rgb([buffer[pos], buffer[pos + 1], buffer[pos + 2]])
                    });
                    if let Err(e) = img.save(&path) {
                        error!("{e}");
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
                        error!("{e}");
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
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
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

fn get_decoder(path: &Path) -> Result<Decoder> {
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
        _ => return Err(anyhow!("{path:#?} has unsupported image format")),
    };

    match decoder.color_type() {
        ColorType::Rgb8 | ColorType::Rgba8 => Ok(decoder),
        _ => Err(anyhow!("{path:#?} has unsupported color type")),
    }
}

pub type Decoder = Box<dyn ImageDecoder>;

#[derive(Debug)]
pub struct ImageInfo {
    pub path: Arc<Path>,
    pub width: u32,
    pub height: u32,
    pub color_type: ColorType,
    pub rgba_size: usize,
}

#[derive(Debug, Parser, Deserialize)]
pub struct Config {
    pub working_buffer_size: usize,
    pub group_size: u32,
    pub sigma: f32,
    pub output_dir: PathBuf,
}

impl Config {
    const PREF: &str = "GBLUR";

    pub fn new(args: &Args) -> Result<Self> {
        let mut conf = config::Config::builder();
        if let Ok(path) = env::var(Self::PREF) {
            conf = conf.add_source(config::File::with_name(&path));
        }
        let ret = conf
            .add_source(Environment::with_prefix(Self::PREF))
            .add_source(config::File::from_str(
                &toml::to_string(args)?,
                config::FileFormat::Toml,
            ))
            .build()?
            .try_deserialize()?;
        Ok(ret)
    }
}
