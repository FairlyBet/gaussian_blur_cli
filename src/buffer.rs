use std::{marker::PhantomData, mem, ptr, slice};

pub struct PersistentRead;
pub struct PersistentWrite;
pub struct Regular;

pub struct ImageBuffer<T> {
    buffer: u32,
    texture: u32,
    size: usize,
    ptr: *mut u8,
    _marker: PhantomData<T>,
}

impl<T> ImageBuffer<T> {
    const TARGET: u32 = gl::TEXTURE_BUFFER;

    /// Creates and maps persistently a `TEXTURE_BUFFER` object
    /// with `flags | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT` flags set
    /// of size `size`
    fn new_persistent(size: usize, flags: u32) -> Option<Self> {
        let isize = size.try_into().ok()?;
        let flags = flags | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT;
        // SAFETY:
        // The subsequent code is valid OpenGL API calls
        // Returning pointer is checked to not be null
        // In case of calling from OpenGl context-less thread
        // or not being able to create required objects will return `None`
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

    /// ### SAFETY:
    ///
    /// Correct `access` and `format` values must be provided.
    /// Providing incorrect values does not lead to undefined behaviour but
    /// will cause OpenGL errors or incorrect results in shader program which may vary
    /// depending on driver implementation.
    /// Providing `unit` that does not correspond to shader program
    /// will not cause errors and will be just ignored
    pub unsafe fn bind_image_texture(&self, unit: u32, access: u32, format: u32) {
        // SAFETY:
        // The subsequent code is memory-safe as no pointers are involved,
        // is correct OpenGL API calls and provides valid `texture` and `buffer` values
        unsafe {
            gl::BindTexture(Self::TARGET, self.texture);
            gl::TexBuffer(Self::TARGET, format, self.buffer);
            gl::BindImageTexture(unit, self.texture, 0, gl::FALSE, 0, access, format);
        }
    }
}

impl ImageBuffer<Regular> {
    pub fn new(size: usize) -> Option<Self> {
        let isize = size.try_into().ok()?;
        // SAFETY:
        // The subsequent code is valid OpenGL API calls
        // In case of calling from OpenGl context-less thread
        // or not being able to create required objects will return `None`
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

    /// ### SAFETY:
    ///
    /// Reading from returning slice while it is used by GPU will
    /// cause data races and unpredictable results
    pub unsafe fn data(&self) -> &[u8] {
        slice::from_raw_parts(self.ptr, self.size)
    }
}

impl ImageBuffer<PersistentWrite> {
    pub fn new_writable(size: usize) -> Option<Self> {
        Self::new_persistent(size, gl::MAP_WRITE_BIT)
    }

    /// ### SAFETY:
    ///
    /// As buffer is mapped only for writing
    /// the returning slice should not be read from.
    /// Also writing to buffer while it is used by GPU will
    /// cause data races and unpredictable result
    pub unsafe fn data(&mut self) -> &mut [u8] {
        // SAFETY:
        // It is safe because inner buffer exists as long as `self`
        // so pointer is valid and `size` is the size of the buffer
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size) }
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
    _marker: PhantomData<*const T>, // Neither Send nor Sync
}

impl<T> UniformBuffer<T> {
    const TARGET: u32 = gl::UNIFORM_BUFFER;

    pub fn new() -> Option<Self> {
        let isize = mem::size_of::<T>().try_into().ok()?;
        // SAFETY:
        // The subsequent code is valid OpenGL API calls,
        // provides correct values for functions,
        // is memory-safe as providing pointer is converted from
        // a mutable reference which is valid by default
        // In case of calling from OpenGl context-less thread
        // or not being able to create required objects will return `None`
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
        // SAFETY:
        // This is safe because
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
        // SAFETY:
        // This is safe as providing `buffer` value is valid
        // and guaranteed by creation API
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
        // As buffer existence is guaranteed by creation API
        // it is safe to delete it
        unsafe {
            gl::DeleteBuffers(1, &self.buffer);
        }
    }
}
