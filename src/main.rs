mod blur;
mod blur_program;
mod buffer;

fn main() -> anyhow::Result<()> {
    let images = &["", "", ""];
    let config = blur::Config {
        working_buffer_size: 100_000_000,
        group_size: (2, 2),
        sigma: 10.0,
    };
    let renderer = blur::Renderer::new().unwrap();
    println!("{}", renderer.max_image_size());
    renderer.process(images, &config)?;
    Ok(())
}
