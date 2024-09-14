use std::{
    marker::PhantomData,
    mem,
    ptr::{self, NonNull},
    slice,
};

pub struct PersistentRead;
pub struct PersistentWrite;
pub struct Regular;

pub struct ImageBuffer<T> {
    buffer: u32,
    texture: u32,
    size: usize,
    ptr: Option<NonNull<u8>>,
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
        // Returning pointer is checked not to be null
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
                ptr: NonNull::new(ptr.cast()),
                _marker: PhantomData,
            })
        }
    }

    /// ### SAFETY:
    ///
    /// Correct `access` and `format` values must be provided.
    /// Providing incorrect values does not lead to undefined behaviour but
    /// will cause OpenGL errors or implementation-specific behaviour
    /// in shader program.
    /// Providing `unit` that does not correspond to shader program
    /// will not cause errors and will be just ignored
    pub unsafe fn bind_image_texture(&self, unit: u32, access: u32, format: u32) {
        // SAFETY:
        // The subsequent code is valid OpenGL calls, provides correct
        // values such as `texture`, `buffer`, `TARGET`
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
            gl::BufferData(Self::TARGET, isize, ptr::null(), gl::DYNAMIC_DRAW);

            let mut texture = 0;
            gl::GenTextures(1, &mut texture);
            if texture == 0 {
                return None;
            }

            Some(Self {
                buffer,
                texture,
                size,
                ptr: None,
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
        // SAFETY:
        // It is safe because inner buffer exists as long as `self`
        // and the pointer is guaranteed to be valid
        unsafe { slice::from_raw_parts(self.ptr.unwrap().as_ptr(), self.size) }
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
        // and the pointer is guaranteed to be valid
        unsafe { slice::from_raw_parts_mut(self.ptr.unwrap().as_ptr(), self.size) }
    }
}

impl<T> Drop for ImageBuffer<T> {
    fn drop(&mut self) {
        // SAFETY:
        // Buffer and texture are guaranteed to exist,
        // so it is safe to delete it
        // Also if buffer was mapped the `ptr` value will be `Some`
        // so in this case it is valid to unmap it
        unsafe {
            gl::DeleteTextures(1, &self.texture);

            if let Some(_) = self.ptr {
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
        // In case of calling from OpenGl context-less thread
        // or not being able to create required object will return `None`
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
        // This is safe because the `buffer` value is
        // guaranteed to be valid buffer object id
        // and the sending data and the buffer has the exact same size
        // Also providing pointer is created from a valid reference to a local parameter
        // Converting size of `T` into `isize` is also safe as it is assured that size of `T`
        // fits into `isize` while creating buffer with `new` function
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
        unsafe {
            gl::BindBufferBase(Self::TARGET, index, self.buffer);
        }
    }
}

impl<T: Copy> UniformBuffer<T> {
    #[allow(unused)]
    pub fn copy_update(&mut self, data: &T) {
        // SAFETY:
        // This is safe because the `buffer` value is
        // guaranteed to be valid buffer object id
        // and the sending data and the buffer has the exact same size
        // Also providing pointer is created from a valid reference given by caller
        // Converting size of `T` into `isize` is also safe as it is assured that size of `T`
        // fits into `isize` while creating buffer with `new` function
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
        // As buffer existence is guaranteed by its creation API
        // it is safe to delete it
        unsafe {
            gl::DeleteBuffers(1, &self.buffer);
        }
    }
}
