use crate::ffi::{self, troy_utils_MemoryPoolHandle};

struct MemoryPoolHandle {
    pool: ffi::troy_utils_MemoryPoolHandle
}

impl MemoryPoolHandle {

    fn new(device_index: usize) -> Self {
        let pool = unsafe {
            let mut pool = troy_utils_MemoryPoolHandle::default();
            ffi::troy_wrapper_create_memory_pool_handle(device_index, &mut pool);
            pool
        };
        Self { pool }
    } 

}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_memory_pool_handle() {
        let pool = MemoryPoolHandle::new(0);
    }

}