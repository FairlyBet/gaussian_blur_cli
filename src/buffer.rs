use std::{marker::PhantomData, mem, ptr, slice};

pub struct WritableImageBuffer {
    buffer: u32,
    texture: u32,
    size: usize,
    ptr: *mut u8,
}

impl WritableImageBuffer {
    pub fn new(size: usize) -> Option<Self> {
        // SAFETY:
        // The subsequent code is valid OpenGL code.
        // In case of not loaded OpenGL functions it will result panicing.
        // In case of calling from non-opengl context thread returns `None`
        unsafe {
            let target = gl::TEXTURE_BUFFER;
            let flags = gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT;
            let isize: isize = size.try_into().ok()?;

            let mut buffer = 0;
            gl::GenBuffers(1, &mut buffer);
            if buffer == 0 {
                return None;
            }
            gl::BindBuffer(target, buffer);
            gl::BufferStorage(target, isize, ptr::null(), flags);
            let ptr = gl::MapBufferRange(target, 0, isize, flags);
            if ptr.is_null() {
                return None;
            }

            let mut texture = 0;
            gl::GenTextures(1, &mut texture);
            if texture == 0 {
                return None;
            }
            gl::BindTexture(target, texture);
            gl::TexBuffer(target, gl::R8, buffer);

            Some(Self {
                buffer,
                texture,
                size,
                ptr: ptr.cast(),
            })
        }
    }

    /// SAFETY:
    /// Caller must valid `access` value
    pub unsafe fn bind_image_texture(&self, unit: u32, access: u32) {
        // SAFETY:
        // As creation API guarantees buffer existance
        // it is safe to make this call
        unsafe {
            gl::BindImageTexture(unit, self.texture, 0, gl::FALSE, 0, access, gl::R8);
        }
    }

    /// SAFETY:
    /// Caller must not read from returning buffer
    /// as it is mapped only for writing
    pub unsafe fn data(&mut self) -> &mut [u8] {
        slice::from_raw_parts_mut(self.ptr, self.size)
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for WritableImageBuffer {
    fn drop(&mut self) {
        // SAFETY:
        // Buffer creation API guarantees its existence and mapping state
        // so subsequent code is valid and safe
        unsafe {
            gl::DeleteTextures(1, &self.texture);
            gl::BindBuffer(gl::TEXTURE_BUFFER, self.buffer);
            gl::UnmapBuffer(gl::TEXTURE_BUFFER);
            gl::DeleteBuffers(1, &self.buffer);
        }
    }
}

pub struct ReadableImageBuffer {
    buffer: u32,
    texture: u32,
    size: usize,
    ptr: *const u8,
}

impl ReadableImageBuffer {
    pub fn new(size: usize) -> Option<Self> {
        // SAFETY:
        // The subsequent code is valid OpenGL code, so it is safe
        // In case of not loaded OpenGL functions it will result panicing
        unsafe {
            let target = gl::TEXTURE_BUFFER;
            let flags = gl::MAP_READ_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT;
            let isize: isize = size.try_into().ok()?;

            let mut buffer = 0;
            gl::GenBuffers(1, &mut buffer);
            if buffer == 0 {
                return None;
            }
            gl::BindBuffer(target, buffer);
            gl::BufferStorage(target, isize, ptr::null(), flags);
            let ptr = gl::MapBufferRange(target, 0, isize, flags);
            if ptr.is_null() {
                return None;
            }

            let mut texture = 0;
            gl::GenTextures(1, &mut texture);
            if texture == 0 {
                return None;
            }
            gl::BindTexture(target, texture);
            gl::TexBuffer(target, gl::R8, buffer);

            Some(Self {
                buffer,
                texture,
                size,
                ptr: ptr.cast(),
            })
        }
    }

    /// SAFETY:
    /// Caller must valid `access` value
    pub unsafe fn bind_image_texture(&self, unit: u32, access: u32) {
        // SAFETY:
        // As creation API guarantees buffer existance
        // it is safe to make this call
        unsafe {
            gl::BindImageTexture(unit, self.texture, 0, gl::FALSE, 0, access, gl::R8);
        }
    }

    /// SAFETY:
    /// Caller must ensure that this buffer isn't being
    /// rendered to while reading from it. Else the result is not
    /// predictable
    pub unsafe fn data(&self) -> &[u8] {
        // SAFETY:
        // It is safe as underlying buffer exists as long as `self`
        slice::from_raw_parts(self.ptr, self.size)
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for ReadableImageBuffer {
    fn drop(&mut self) {
        // SAFETY:
        // Buffer creation API guarantees its existence and mapping state
        // so subsequent code is valid and safe
        unsafe {
            gl::DeleteTextures(1, &self.texture);
            gl::BindBuffer(gl::TEXTURE_BUFFER, self.buffer);
            gl::UnmapBuffer(gl::TEXTURE_BUFFER);
            gl::DeleteBuffers(1, &self.buffer);
        }
    }
}

pub struct ImageBuffer {
    buffer: u32,
    texture: u32,
    _marker: PhantomData<*const ()>,
}

impl ImageBuffer {
    pub fn new(size: usize) -> Option<Self> {
        // SAFETY:
        // The subsequent code is valid OpenGL code, so it is safe
        // In case of not loaded OpenGL functions it will result panicing
        unsafe {
            let target = gl::TEXTURE_BUFFER;
            let isize: isize = size.try_into().ok()?;

            let mut buffer = 0;
            gl::GenBuffers(1, &mut buffer);
            if buffer == 0 {
                return None;
            }
            gl::BindBuffer(target, buffer);
            gl::BufferData(
                gl::TEXTURE_BUFFER,
                isize,
                std::ptr::null(),
                gl::DYNAMIC_DRAW,
            );

            let mut texture = 0;
            gl::GenTextures(1, &mut texture);
            if texture == 0 {
                return None;
            }
            gl::BindTexture(target, texture);
            gl::TexBuffer(target, gl::R8, buffer);

            Some(Self {
                buffer,
                texture,
                _marker: PhantomData,
            })
        }
    }

    /// SAFETY:
    /// Caller must provide valid `access` value
    pub unsafe fn bind_image_texture(&self, unit: u32, access: u32) {
        // SAFETY:
        // As creation API guarantees buffer existance
        // it is safe to make this call
        unsafe {
            gl::BindImageTexture(unit, self.texture, 0, gl::FALSE, 0, access, gl::R8);
        }
    }
}

impl Drop for ImageBuffer {
    fn drop(&mut self) {
        // SAFETY:
        // Buffer creation API guarantees its existence
        // so subsequent delete call is valid and safe
        unsafe {
            gl::DeleteTextures(1, &self.texture);
            gl::DeleteBuffers(1, &self.buffer);
        }
    }
}

pub struct UniformBuffer<T> {
    buffer: u32,
    _marker: PhantomData<*const T>,
}

impl<T> UniformBuffer<T> {
    pub fn new() -> Option<Self> {
        unsafe {
            let mut buffer = 0;
            let isize: isize = mem::size_of::<T>().try_into().ok()?;
            gl::GenBuffers(1, &mut buffer);
            if buffer == 0 {
                return None;
            }
            gl::BindBuffer(gl::UNIFORM_BUFFER, buffer);
            gl::BufferData(gl::UNIFORM_BUFFER, isize, ptr::null(), gl::STREAM_DRAW);
            Some(Self {
                buffer,
                _marker: PhantomData,
            })
        }
    }

    pub fn update(&mut self, data: T) {
        unsafe {
            gl::BindBuffer(gl::UNIFORM_BUFFER, self.buffer);
            gl::BufferSubData(
                gl::UNIFORM_BUFFER,
                0,
                mem::size_of::<T>() as isize,
                (&data as *const T).cast(),
            );
        }
    }

    pub fn bind_buffer_base(&self, index: u32) {
        unsafe {
            gl::BindBufferBase(gl::UNIFORM_BUFFER, index, self.buffer);
        }
    }
}

impl<T: Copy> UniformBuffer<T> {
    pub fn copy_update(&mut self, data: &T) {
        unsafe {
            gl::BindBuffer(gl::UNIFORM_BUFFER, self.buffer);
            gl::BufferSubData(
                gl::UNIFORM_BUFFER,
                0,
                mem::size_of::<T>() as isize,
                (data as *const T).cast(),
            );
        }
    }
}

impl<T> Drop for UniformBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.buffer);
        }
    }
}
