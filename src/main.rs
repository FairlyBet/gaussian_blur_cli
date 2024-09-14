mod blur;
mod blur_program;
mod buffer;

use blur::{Config, Renderer};
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf};
use tracing::error;
use tracing_subscriber::EnvFilter;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let mut paths = vec![];
    for path in &args.images {
        paths.push(path.as_path().into());
    }
    if let Some(input_dir) = &args.input_dir {
        match fs::read_dir(input_dir) {
            Ok(entries) => {
                for entry in entries {
                    match entry {
                        Ok(entry) => {
                            paths.push(entry.path().as_path().into());
                        }
                        Err(e) => error!("{e}"),
                    }
                }
            }
            Err(e) => error!("{e}"),
        }
    }
    let config = match Config::new(&args) {
        Ok(config) => config,
        Err(e) => {
            error!("{e}");
            return;
        }
    };

    match Renderer::new() {
        Some(renderer) => {
            if args.max_image_resolution {
                println!(
                    "Maximum resolution of an image that can be processed is {} pixels",
                    renderer.max_image_resolution()
                );
            }
            if let Err(e) = renderer.process(paths, &config) {
                error!("{e}");
            }
        }
        _ => error!("Can't create OpenGL context"),
    }
}

#[derive(Debug, Parser, Serialize)]
pub struct Args {
    #[arg(index = 1, value_name = "FILES", help = "A list of images to process")]
    images: Vec<PathBuf>,
    #[arg(
        short,
        help = "Print maximum resolution of an image that can be processed"
    )]
    max_image_resolution: bool,
    #[arg(
        short,
        value_name = "NUM",
        default_value_t = 100_000_000,
        help = "Set size of a working buffer. The actual memory consumption will be x3 of that value"
    )]
    working_buffer_size: usize,
    #[arg(
        short,
        value_name = "NUM",
        default_value_t = 8,
        help = "Set group-size of a compute shader. (Recommended values are 4, 8, 16, 32)"
    )]
    group_size: u32,
    #[arg(
        short,
        value_name = "NUM",
        default_value_t = 5.0,
        help = "Blur intensity"
    )]
    sigma: f32,
    #[arg(
        short,
        value_name = "DIR",
        default_value = "./",
        help = "Output directory for processed images"
    )]
    output_dir: PathBuf,
    #[arg(
        short,
        value_name = "DIR",
        help = "Input directory with images to process"
    )]
    input_dir: Option<PathBuf>,
}
