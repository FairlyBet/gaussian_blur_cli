mod blur_render;
mod buffer;
mod blur_program;

use blur_render::Renderer;

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

    Renderer::new(20.0);

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
