mod blur;
mod blur_program;
mod buffer;

fn main() {
    let mut images: Vec<std::sync::Arc<str>> = vec![];
    // images.push("sample.jpg".into());
    // images.push("sample.webp".into());

    for d in std::fs::read_dir(r"C:\Users\Aleksandr\Desktop\input").unwrap() {
        images.push(d.unwrap().path().to_str().unwrap().into());
    }
    let config = blur::Config {
        working_buffer_size: 100_000_000,
        group_size: 2,
        sigma: 4.0,
        output_dir: std::path::PathBuf::try_from(r"C:\Users\Aleksandr\Desktop\output").unwrap(),
    };

    let renderer = blur::Renderer::new().unwrap();
    println!("{}", renderer.max_image_size());
    if let Err(e) = renderer.process(images, &config) {
        eprintln!("{e}");
    }
}
