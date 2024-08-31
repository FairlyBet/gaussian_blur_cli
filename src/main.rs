mod blur_render;
mod buffer;

use anyhow::anyhow;
use blur_render::Renderer;
use buffer::{ImageBuffer, ReadableImageBuffer, WritableImageBuffer};
use glfw::{fail_on_errors, ClientApiHint, OpenGlProfileHint, WindowHint, WindowMode};
use image::EncodableLayout as _;

fn main() -> anyhow::Result<()> {
    // let packed: &[u32] = &[1, 2, 3, 4];
    // let (widht, height) = (2_usize, 2_usize);
    // const RGB: usize = 3;

    // for h in 0..height {
    //     for w in 0..widht {
    //         let w = w * RGB;
    //         let h = (h * widht) * RGB;

    //         let r_index = w + h;
    //         let g_index = w + h + 1;
    //         let b_index = w + h + 2;

    //         let r_chunk = packed[r_index / size_of::<u32>()];
    //         let g_chunk = packed[g_index / size_of::<u32>()];
    //         let b_chunk = packed[b_index / size_of::<u32>()];

    //         let r_shift = r_index % size_of::<u32>();
    //         let g_shift = g_index % size_of::<u32>();
    //         let b_shift = b_index % size_of::<u32>();

    //         let r = (r_chunk >> (8 * r_shift)) & 0xFF;
    //         let g = (g_chunk >> (8 * g_shift)) & 0xFF;
    //         let b = (b_chunk >> (8 * b_shift)) & 0xFF;

    //         println!("{r} {g} {b}\n");

    //         let rgb = (r, g, b);
    //     }
    // }

    // Simple way to set up offscreen rendering is just to use invisible window
    let mut glfw = glfw::init(fail_on_errors!())?;
    glfw.window_hint(WindowHint::ClientApi(ClientApiHint::OpenGl));
    glfw.window_hint(WindowHint::OpenGlProfile(OpenGlProfileHint::Core));
    glfw.window_hint(WindowHint::ContextVersion(4, 5));
    glfw.window_hint(WindowHint::Visible(false));

    let (mut w, _) = glfw
        .create_window(1, 1, "", WindowMode::Windowed)
        .ok_or(anyhow!("Can't create window"))?;

    gl::load_with(|symbol| w.get_proc_address(symbol));
    println!("{:?}", Renderer::new());
    
    // unsafe {
    // let size = 100;

    // let mut buffer1 = 0;
    // let flags = gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT;
    // gl::GenBuffers(1, &mut buffer1);
    // gl::BindBuffer(gl::TEXTURE_BUFFER, buffer1);
    // gl::BufferStorage(gl::TEXTURE_BUFFER, size, std::ptr::null(), flags);
    // let ptr = gl::MapBufferRange(gl::TEXTURE_BUFFER, 0, size, flags);
    // println!("{}", ptr as usize);
    // gl::BufferData(
    //     gl::TEXTURE_BUFFER,
    //     buffer_size,
    //     std::ptr::null(),
    //     gl::STATIC_READ,
    // );

    // let mut texture1 = 0;
    // gl::GenTextures(1, &mut texture1);
    // gl::BindTexture(gl::TEXTURE_BUFFER, texture1);
    // gl::TexBuffer(gl::TEXTURE_BUFFER, gl::R8, buffer1);
    // gl::BindImageTexture(0, texture1, 0, gl::FALSE, 0, gl::READ_ONLY, gl::R8);
    // }

    // let mut max_ssbo_size = 0;
    // unsafe {
    //     gl::GetInteger64v(gl::MAX_TEXTURE_BUFFER_SIZE, &mut max_ssbo_size);
    // }
    // println!("{max_ssbo_size}");

    // // let img = image::open("sample.jpg")?;
    // // if let Some(rgb8) = img.as_rgb8() {}

    // std::io::stdin().read_line(&mut String::new())?;

    Ok(())
}

fn gaussian_kernel_1d(size: usize, sigma: f32) -> Vec<f32> {
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
