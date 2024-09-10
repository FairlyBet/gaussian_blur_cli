use crate::{
    blur_program::BlurProgram,
    buffer::{ImageBuffer, PersistentRead, PersistentWrite, UniformBuffer},
};
use anyhow::{anyhow, Result};
use glfw::{
    fail_on_errors, ClientApiHint, GlfwReceiver, OpenGlProfileHint, PWindow, WindowEvent,
    WindowHint, WindowMode,
};
use image::codecs::{
    bmp::BmpDecoder, jpeg::JpegDecoder, png::PngDecoder, tga::TgaDecoder, tiff::TiffDecoder,
    webp::WebPDecoder,
};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageFormat, Limits};
use moro_local::{Scope, Spawned};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::{f32, fmt::Display, fs::File, future::Future, io::BufReader, result, sync::Arc};
use tokio::task;

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
        let kernel = gaussian_kernel(6 * config.sigma as usize + 1, config.sigma);
        let program = BlurProgram::new(
            self.window.get_context_version(),
            config.group_size,
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

        let mut input_slice = unsafe { input_buffer.data() };
        while !images.is_empty() {
            let mut i = 0;
            while i < images.len() {
                // let path = images[i].;

                match self.try_load("path".to_string(), input_slice) {
                    Ok(info) => {
                        input_slice = input_slice.split_at_mut(info.rgba_size).1;
                        // loaded_images.push(info);
                        // handles.push(future);
                        i += 1;
                    }
                    Err(err) => match err {
                        LoadError::TooLargeImage => {
                            
                        }
                        LoadError::DecoderError(_) => {}
                        _ => {}
                    },
                }
            }

            /*
                       // thread::scope(|scope| {
                       //     // Start loading
                       //     let (advanced, handles) =
                       //         Self::fill_buffer(&infos, load_progress.end, input_buf, scope);
                       //     load_progress.start = render_progess.end;
                       //     load_progress.end = advanced;

                       //     for (img, i) in &loaded_images {
                       //         image_data_buffer.copy_update(img);

                       //         // SAFETY:
                       //         unsafe {
                       //             render_buf.input_buffer.bind_image_texture(
                       //                 BlurProgram::INPUT_BINDING_UNIT,
                       //                 gl::READ_ONLY,
                       //                 gl::RGBA8,
                       //             );
                       //             intermediate_buffer.bind_image_texture(
                       //                 BlurProgram::OUTPUT_BINDING_UNIT,
                       //                 gl::WRITE_ONLY,
                       //                 gl::RGBA8,
                       //             );
                       //             program.set_horizontal();
                       //             gl::DispatchCompute(
                       //                 infos[*i].width.div_ceil(program.group_size().0),
                       //                 infos[*i].height.div_ceil(program.group_size().1),
                       //                 1,
                       //             );

                       //             intermediate_buffer.bind_image_texture(
                       //                 BlurProgram::INPUT_BINDING_UNIT,
                       //                 gl::READ_ONLY,
                       //                 gl::RGBA8,
                       //             );
                       //             render_buf.output_buffer.bind_image_texture(
                       //                 BlurProgram::OUTPUT_BINDING_UNIT,
                       //                 gl::WRITE_ONLY,
                       //                 gl::RGBA8,
                       //             );
                       //             program.set_vertical();
                       //             gl::DispatchCompute(
                       //                 infos[*i].width.div_ceil(program.group_size().0),
                       //                 infos[*i].height.div_ceil(program.group_size().1),
                       //                 1,
                       //             );

                       //             gl::Finish();
                       //         }
                       //     }
                       // });
            */
            _ = glfw::flush_messages(&self.receiver);
        }

        Ok(())
    }

    fn try_load<'a>(
        &self,
        path: String,
        buffer: &'a mut [u8],
    ) -> result::Result<ImageInfo, LoadError> {
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
                    original_color_type: decoder.original_color_type(),
                    color_type: decoder.color_type(),
                    rgba_size: size,
                };

                if let Err(e) = Self::read_image(decoder, buffer) {
                    return Err(LoadError::DecoderError(e));
                }
                Ok((image_info))
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
                // without setting the Alpha component so it is
                // some random value from memory
                buffer
                    .par_chunks_exact_mut(4)
                    .enumerate()
                    .for_each(|(i, pixel)| {
                        let pos = i * 3;
                        pixel[0..3].copy_from_slice(&image_buf[pos..pos + 3]);
                    });
            }
            ColorType::Rgba8 => decoder.read_image(buffer)?,
            _ => unreachable!(),
        }
        Ok(())
    }

    fn save_images() {}
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

fn get_decoder(path: &str) -> Result<Decoder> {
    let format = ImageFormat::from_path(path)?;
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let decoder: Box<dyn ImageDecoder + Send + Sync> = match format {
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
        _ => Err(anyhow!("{path} has unsuppoted color type")),
    }
}

fn rgb_size_to_rgba_size(rbg_size: u64) -> Option<u64> {
    rbg_size.checked_add(rbg_size / 3)
}

pub type Decoder = Box<dyn ImageDecoder + Send + Sync>;

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

impl std::error::Error for LoadError {}

pub struct ImageInfo {
    path: String,
    width: u32,
    height: u32,
    original_color_type: ExtendedColorType,
    color_type: ColorType,
    rgba_size: usize,
}

pub struct Config {
    pub working_buffer_size: usize,
    pub group_size: (u32, u32),
    pub sigma: f32,
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
