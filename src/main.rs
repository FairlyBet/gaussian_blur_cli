mod blur;
mod blur_program;
mod buffer;

use blur::{Config, Renderer};
use clap::Parser;
use std::{fs, path::PathBuf};

fn main() {
    let args = Args::parse();
    let mut paths = vec![];

    for path in args.images {
        paths.push(path.as_path().into());
    }
    if let Some(input_dir) = args.input_dir {
        match fs::read_dir(input_dir) {
            Ok(entries) => {
                for entry in entries {
                    match entry {
                        Ok(entry) => {
                            paths.push(entry.path().as_path().into());
                        }
                        Err(e) => eprintln!("{e}"),
                    }
                }
            }
            Err(e) => eprintln!("{e}"),
        }
    }
    let config = Config {
        working_buffer_size: args.working_buffer_size,
        group_size: args.group_size,
        sigma: args.sigma,
        output_dir: args.output_dir,
    };

    match Renderer::new() {
        Some(renderer) => {
            if args.max_image_size {
                println!("{}", renderer.max_image_size());
            }
            if let Err(e) = renderer.process(paths, &config) {
                eprintln!("{e}");
            }
        }
        _ => eprintln!("Can't create OpenGL context"),
    }
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(short)]
    max_image_size: bool,
    #[arg(short, default_value_t = 100_000_000)]
    working_buffer_size: usize,
    #[arg(short, default_value_t = 8)]
    group_size: u32,
    #[arg(short)]
    sigma: f32,
    #[arg(short, default_value = "")]
    output_dir: PathBuf,
    #[arg(short)]
    input_dir: Option<PathBuf>,
    #[arg(long)]
    images: Vec<PathBuf>,
}
