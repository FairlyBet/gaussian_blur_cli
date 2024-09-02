use std::{marker::PhantomData, mem, ptr, slice};

pub struct PersistentRead;
pub struct PersistentWrite;
pub struct NonPersistent;

pub struct ImageBuffer<T> {
    buffer: u32,
    texture: u32,
    size: usize,
    ptr: *mut u8,
    _marker: PhantomData<T>,
}

impl<T> ImageBuffer<T> {
    const TARGET: u32 = gl::TEXTURE_BUFFER;

    fn new_persistent(size: usize, flags: u32) -> Option<Self> {
        let isize: isize = size.try_into().ok()?;
        let flags = flags | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT;
        // SAFETY:
        // The subsequent code is valid OpenGL code.
        // In case of not loaded OpenGL functions it will result panicing.
        // In case of calling from non-opengl context thread returns `None`
        unsafe {
            let mut buffer = 0;
            gl::GenBuffers(1, &mut buffer);
            if buffer == 0 {
                return None;
            }
            gl::BindBuffer(Self::TARGET, buffer);
            gl::BufferStorage(Self::TARGET, isize, ptr::null(), flags);
            let ptr = gl::MapBufferRange(Self::TARGET, 0, isize, flags);
            if ptr.is_null() {
                return None;
            }

            let mut texture = 0;
            gl::GenTextures(1, &mut texture);
            if texture == 0 {
                return None;
            }

            Some(Self {
                buffer,
                texture,
                size,
                ptr: ptr.cast(),
                _marker: PhantomData,
            })
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub unsafe fn bind_image_texture(&self, unit: u32, access: u32, format: u32) {
        gl::BindTexture(Self::TARGET, self.texture);
        gl::TexBuffer(Self::TARGET, format, self.buffer);
        gl::BindImageTexture(unit, self.texture, 0, gl::FALSE, 0, access, format);
    }
}

impl ImageBuffer<NonPersistent> {
    pub fn new(size: usize) -> Option<Self> {
        let isize: isize = size.try_into().ok()?;
        // SAFETY:
        // The subsequent code is valid OpenGL code, so it is safe
        // In case of not loaded OpenGL functions it will result panicing
        unsafe {
            let mut buffer = 0;
            gl::GenBuffers(1, &mut buffer);
            if buffer == 0 {
                return None;
            }
            gl::BindBuffer(Self::TARGET, buffer);
            gl::BufferData(gl::TEXTURE_BUFFER, isize, ptr::null(), gl::DYNAMIC_DRAW);

            let mut texture = 0;
            gl::GenTextures(1, &mut texture);
            if texture == 0 {
                return None;
            }

            Some(Self {
                buffer,
                texture,
                size,
                ptr: ptr::null_mut(),
                _marker: PhantomData,
            })
        }
    }
}

impl ImageBuffer<PersistentRead> {
    pub fn new_readable(size: usize) -> Option<Self> {
        Self::new_persistent(size, gl::MAP_READ_BIT)
    }

    /// SAFETY:
    /// Caller must ensure that this buffer isn't being
    /// rendered to while reading from it. Else the result is not
    /// predictable
    pub unsafe fn data(&self) -> &[u8] {
        slice::from_raw_parts(self.ptr, self.size)
    }
}

impl ImageBuffer<PersistentWrite> {
    pub fn new_writable(size: usize) -> Option<Self> {
        Self::new_persistent(size, gl::MAP_WRITE_BIT)
    }

    /// SAFETY:
    /// Caller must not read from returning buffer
    /// as it is mapped only for writing
    pub unsafe fn data(&mut self) -> &mut [u8] {
        slice::from_raw_parts_mut(self.ptr, self.size)
    }
}

impl<T> Drop for ImageBuffer<T> {
    fn drop(&mut self) {
        // SAFETY:
        // Buffer creation API guarantees its existence and mapping state
        // so subsequent code is valid and safe
        unsafe {
            gl::DeleteTextures(1, &self.texture);

            if !self.ptr.is_null() {
                gl::BindBuffer(Self::TARGET, self.buffer);
                gl::UnmapBuffer(Self::TARGET);
            }
            gl::DeleteBuffers(1, &self.buffer);
        }
    }
}

pub struct UniformBuffer<T> {
    buffer: u32,
    _marker: PhantomData<*const T>,
}

impl<T> UniformBuffer<T> {
    const TARGET: u32 = gl::UNIFORM_BUFFER;

    pub fn new() -> Option<Self> {
        let isize: isize = mem::size_of::<T>().try_into().ok()?;
        // SAFETY:
        //
        unsafe {
            let mut buffer = 0;
            gl::GenBuffers(1, &mut buffer);
            if buffer == 0 {
                return None;
            }
            gl::BindBuffer(Self::TARGET, buffer);
            gl::BufferData(Self::TARGET, isize, ptr::null(), gl::STREAM_DRAW);
            
            Some(Self {
                buffer,
                _marker: PhantomData,
            })
        }
    }

    pub fn update(&mut self, data: T) {
        unsafe {
            gl::BindBuffer(Self::TARGET, self.buffer);
            gl::BufferSubData(
                Self::TARGET,
                0,
                mem::size_of::<T>() as isize,
                (&data as *const T).cast(),
            );
        }
    }

    pub fn bind_buffer_base(&self, index: u32) {
        unsafe {
            gl::BindBufferBase(Self::TARGET, index, self.buffer);
        }
    }
}

impl<T: Copy> UniformBuffer<T> {
    pub fn copy_update(&mut self, data: &T) {
        // SAFETY:
        // 
        unsafe {
            gl::BindBuffer(Self::TARGET, self.buffer);
            gl::BufferSubData(
                Self::TARGET,
                0,
                mem::size_of::<T>() as isize,
                (data as *const T).cast(),
            );
        }
    }
}

impl<T> Drop for UniformBuffer<T> {
    fn drop(&mut self) {
        // SAFETY:
        //
        unsafe {
            gl::DeleteBuffers(1, &self.buffer);
        }
    }
}
