This is a CLI tool that applies gaussian blur to images using OpenGL of version 4.4 and compute shaders. 

The tool is designed to process a large amount of images, but it is also fit to blur a single image as well. There are a bunch of parameters that need to be set to use the program: 
1. Working buffer size: this is the size of a buffer that will be used to process images. It is meant to be big enough to hold several images at once. Also, the actual memory consumption will be x3 of that value, which is described below. 
2. Group size: this is the size of a working group of a compute shader. It can be said that it is something like the number of threads that will run the shader program. For instance if group size is 8 then GPU will run groups of 8x8 which is 64 shader invocations at once.
3. Sigma: This is a numeric value that determines blur intense. 

So these are parameters that need to be understood before running the blur program. 

---

The process of applying blur is fairly simple.

The first step is to read as many images as possible into the buffer. If an image's size is larger than the entire buffer, then it can't be processed, so it is removed from the input list and an error is printed. 

The second step is to draw a blur, which is done in two [passes][1]. First pass calculates the horizontal average of an image. Second calculates the vertical average of horizontal averages produced by the previous pass. This is why actually 3 buffers are used. One that contains input images, the second that contains horizontal averages, and the third one that contains the result. So this is why memory consumption will be x3 of the Working buffer size value.

The third step is to read processed images from the GPU and save them to disk. 

These steps repeat until no images are left in the input list. 

---

Implementation uses persistent mapping so that image loading should be fast. Also, it utilizes parallelization on the CPU side, but on most GPUs, input and output operations remain bottlenecks of the program. 

#### Limitations:

1. Typically, the maximum size of a working buffer is either up to the VRAM capacity of a GPU but not bigger than 8 GB because of implementaion's internal peculiarities. However, some GPUs may have severe limitations. For example, Intel HD Graphics 620 has a limitation of 134 MB. Though in the case of such a GPU, using compute shader won't provide much benefit compared to CPU processing.
2. As said above, the required version of OpenGL is 4.4, which is not supported by some old GPUs.
3. Image pixel type: RGB8 or RGBA8.
4. Image formats:
    * PNG
    * JPEG
    * WebP
    * TIFF
    * TGA
    * BMP

---

#### Configuration:

There are several ways to configure the program:
1. You can set environment variable GBLUR and provide path to a config file which may be some markup file such as Json or Toml.
2. You can setup several variables with prefix GBLUR and provide them desired values.
3. You can set values as command line arguments.

```
Config values:
working_buffer_size: NUMBER
group_size: NUMBER
sigma: NUMBER
output_dir: DIRECTORY PATH
```
Example of a config file:
``` Toml
working_buffer_size = 10000000
group_size = 2
sigma = 10
```
Example of env variable:
``` PowerShell
$env:GBLUR_SIGMA = 30;
```

All the command line arguments can be retrieved with `--help` or `-h` command.
There are two of them that show limitations of the current GPU:
`-m` - Prints maximum image size in pixels (both rgb and rgba) which can be processed if buffer has maximum size.
`-b` - Prints the actual maximum buffer size in bytes.

All the values are optional to specify as the program uses default values in case of absence of some.
Default `output_dir` is current location of the program.
Default `sigma` is 5.
Default `group_size` is 8.
Default `buffer_size` is 100 MB.

[1]: https://en.wikipedia.org/wiki/Gaussian_blur#:~:text=In%20practice%2C%20it%20is%20best%20to%20take%20advantage%20of%20the%20Gaussian%20blur%E2%80%99s%20separable%20property%20by%20dividing%20the%20process%20into%20two%20passes.