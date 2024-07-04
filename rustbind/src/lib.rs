#[cxx::bridge]
mod ffi {
    #[namespace = "troy_rust"]
    unsafe extern "C++" {
        include!("troy-rustbind/interfaces/memory_pool.cuh");
        type MemoryPool;
        fn create_memory_pool(device_index: usize) -> UniquePtr<MemoryPool>;
    }
}


pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn create_pool() {
        let pool = ffi::create();
        assert!(!pool.is_null());
    }
}
