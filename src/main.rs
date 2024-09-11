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
    match fs::read_dir(args.input_dir) {
        Ok(entries) => {
            for enty in entries {
                match enty {
                    Ok(enty) => {
                        paths.push(enty.path().as_path().into());
                    }
                    Err(e) => eprintln!("{e}"),
                }
            }
        }
        Err(e) => eprintln!("{e}"),
    }
    let config = Config {
        working_buffer_size: args.working_buffer_size,
        group_size: args.group_size,
        sigma: args.sigma,
        output_dir: args.output_dir,
    };
    // let config = Config {
    //     working_buffer_size: 10000000,
    //     group_size: 16,
    //     sigma: 11.0,
    //     output_dir: Default::default(),
    // };
    match Renderer::new() {
        Some(renderer) => {
            if args.max_image_size {
                println!("{}", renderer.max_image_size());
            }
            if let Err(e) = renderer.process(paths, &config) {
                println!("{e}");
            }
        }
        _ => println!("Can't create OpenGL context"),
    }
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(short)]
    max_image_size: bool,
    #[arg(short)]
    working_buffer_size: usize,
    #[arg(short)]
    group_size: u32,
    #[arg(short)]
    sigma: f32,
    #[arg(short)]
    output_dir: PathBuf,
    #[arg(short)]
    input_dir: PathBuf,
    #[arg(long)]
    images: Vec<PathBuf>,
}
