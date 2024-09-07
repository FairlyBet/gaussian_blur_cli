use crate::{
    blur_program::BlurProgram,
    buffer::{ImageBuffer, PersistentRead, PersistentWrite, Regular, UniformBuffer},
};
use anyhow::anyhow;
use glfw::{
    fail_on_errors, ClientApiHint, GlfwReceiver, OpenGlProfileHint, PWindow, WindowEvent,
    WindowHint, WindowMode,
};
use image::codecs::{
    bmp::BmpDecoder, jpeg::JpegDecoder, png::PngDecoder, tga::TgaDecoder, tiff::TiffDecoder,
    webp::WebPDecoder,
};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageFormat, Limits, Rgb};
// use rayon::Scope;
use std::{
    f32,
    fs::File,
    io::BufReader,
    thread::{self, Scope, ScopedJoinHandle},
};

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

        let kernel = gaussian_kernel(6 * config.sigma as usize + 1, config.sigma);
        let program = BlurProgram::new(
            self.window.get_context_version(),
            config.group_size,
            &kernel,
        )
        .ok_or(anyhow!("Cannot create shader program"))?;

        let mut image_data_buffer =
            UniformBuffer::<ImageData>::new().ok_or(anyhow!("Cannot create uniform buffer"))?;
        let mut buf1 = WorkingBuffer::new(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"))?;
        let buf2 = WorkingBuffer::new(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"))?;
        let intermediate_buffer = ImageBuffer::new(config.working_buffer_size)
            .ok_or(anyhow!("Cannot create working buffer"))?;

        let mut render_progess = 0..0;
        let mut load_progress = 0..0;
        let loaded_images: Vec<(ImageData, usize)> = vec![];
        let mut i = 0;
        while render_progess.start < infos.len() {
            let render_buf = if i % 2 == 0 { &buf2 } else { &buf1 };
            let (input_slice, output_slice) = unsafe {
                if i % 2 == 0 {
                    (buf1.input_buffer.data(), buf1.output_buffer.data())
                } else {
                    (buf2.input_buffer.data(), buf2.output_buffer.data())
                }
            };
            let input_slice = unsafe { buf1.input_buffer.data() };
            let output_slice = unsafe { buf1.output_buffer.data() };
            thread::scope(|scope| {
                // Start loading
                let (advanced, handles) =
                    Self::fill_buffer(&infos, load_progress.end, input_slice, scope);
                load_progress.start = render_progess.end;
                load_progress.end = advanced;
            });
            for (img, i) in &loaded_images {
                image_data_buffer.copy_update(img);

                // SAFETY:
                unsafe {
                    render_buf.input_buffer.bind_image_texture(
                        BlurProgram::INPUT_BINDING_UNIT,
                        gl::READ_ONLY,
                        gl::RGBA8,
                    );
                    intermediate_buffer.bind_image_texture(
                        BlurProgram::OUTPUT_BINDING_UNIT,
                        gl::WRITE_ONLY,
                        gl::RGBA8,
                    );
                    image_data_buffer.bind_buffer_base(BlurProgram::IMAGE_DATA_BINDING_POINT);
                    program.set_horizontal();
                    gl::DispatchCompute(
                        infos[*i].width.div_ceil(program.group_size().0),
                        infos[*i].height.div_ceil(program.group_size().1),
                        1,
                    );

                    intermediate_buffer.bind_image_texture(
                        BlurProgram::INPUT_BINDING_UNIT,
                        gl::READ_ONLY,
                        gl::RGBA8,
                    );
                    render_buf.output_buffer.bind_image_texture(
                        BlurProgram::OUTPUT_BINDING_UNIT,
                        gl::WRITE_ONLY,
                        gl::RGBA8,
                    );
                    program.set_vertical();
                    gl::DispatchCompute(
                        infos[*i].width.div_ceil(program.group_size().0),
                        infos[*i].height.div_ceil(program.group_size().1),
                        1,
                    );
                    gl::Finish();
                }
            }
            i += 1;
        }

        Ok(())
    }

    fn fill_buffer<'env, 'scope>(
        infos: &'env [ImageInfo],
        start: usize,
        mut buffer: &'env mut [u8],
        scope: &'scope Scope<'scope, 'env>,
    ) -> (usize, Vec<ScopedJoinHandle<'scope, anyhow::Result<()>>>) {
        let mut i = start;
        let mut handles = vec![];

        while i < infos.len() {
            let info = &infos[i];
            if info.rgba_size > buffer.len() {
                break;
            }
            let (occupied, free) = buffer.split_at_mut(infos[i].rgba_size);
            buffer = free;
            let handle = scope.spawn(move || Self::read_image(info, occupied));
            handles.push(handle);
            i += 1;
        }
        return (i, handles);
    }

    fn read_image(info: &ImageInfo<'_>, occupied: &mut [u8]) -> anyhow::Result<()> {
        match info.color_type {
            ColorType::Rgb8 => {
                let image = image::open(info.path)?;
                let image = image.as_rgb8().unwrap();
                let mut i = 0;
                image.as_raw().chunks_exact(3).for_each(|pixel| {
                    occupied[i..i + 3].copy_from_slice(pixel);
                    i += 4;
                });
            }
            ColorType::Rgba8 => {
                let decoder = self::get_decoder(info.path)?;
                decoder.read_image(occupied)?;
            }
            _ => return Err(anyhow!("{} has unsupported color type", info.path)),
        }
        Ok(())
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

fn get_decoder(path: &str) -> anyhow::Result<Decoder> {
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
        _ => return Err(anyhow!("Unsupported image format")),
    };

    match decoder.color_type() {
        ColorType::Rgb8 | ColorType::Rgba8 => Ok(decoder),
        _ => Err(anyhow!("Unsuppoted color type")),
    }
}

fn rgb_size_to_rgba_size(rbg_size: usize) -> Option<usize> {
    rbg_size.checked_add(rbg_size / 3)
}

pub type Decoder = Box<dyn ImageDecoder + Send + Sync>;

pub struct ImageInfo<'a> {
    path: &'a str,
    width: u32,
    height: u32,
    size: u64,
    original_color_type: ExtendedColorType,
    color_type: ColorType,
    rgba_size: usize,
}

impl<'a> ImageInfo<'a> {
    pub fn new(path: &'a str, renderer: &Renderer) -> anyhow::Result<Self> {
        let decoder = self::get_decoder(path)?;
        let err = format!("{path} doesn't fit into working buffer");
        let mut rgba_size: usize = decoder
            .total_bytes()
            .try_into()
            .map_err(|_| anyhow!(err.clone()))?;
        let color_type = decoder.color_type();
        if color_type == ColorType::Rgb8 {
            rgba_size = rgb_size_to_rgba_size(rgba_size).ok_or(anyhow!(err.clone()))?;
        }

        if rgba_size > renderer.max_image_size {
            return Err(anyhow!(err));
        }

        Ok(Self {
            path,
            width: decoder.dimensions().0,
            height: decoder.dimensions().1,
            size: decoder.total_bytes(),
            original_color_type: decoder.original_color_type(),
            rgba_size,
            color_type,
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
