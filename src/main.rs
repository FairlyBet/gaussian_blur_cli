mod blur_program;
mod blur_render;
mod buffer;

fn main() -> anyhow::Result<()> {
    // let images = ["", "", ""].map(|item| ImageInfo::new(item).unwrap());

    let renderer = blur_render::Renderer::new().unwrap();
    println!("{}", renderer.max_image_size());

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
