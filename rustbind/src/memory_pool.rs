use super::ffi;

type UpMemoryPool = cxx::UniquePtr<ffi::MemoryPool>;

pub struct MemoryPool {
    p: UpMemoryPool,
}

impl MemoryPool {

    #[inline]
    pub fn new(device_index: usize) -> Self {
        let p = ffi::memory_pool_constructor(device_index);
        Self { p }
    }

    #[inline]
    pub fn device_index(&self) -> usize {
        self.p.device_index()
    }

    #[inline]
    pub fn handle_address(&self) -> usize {
        self.p.handle_address()
    }

    #[inline]
    pub fn global_pool() -> Self {
        let p = ffi::memory_pool_static_global_pool();
        Self { p }
    }

    #[inline]
    pub fn destroy() {
        ffi::memory_pool_static_destroy();
    }

    #[inline]
    pub fn nullptr() -> Self {
        let p = ffi::memory_pool_static_nullptr();
        Self { p }
    }

}

impl std::fmt::Display for MemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.handle_address() == 0 {
            write!(f, "MemoryPool(nullptr)")
        } else {
            write!(f, "MemoryPool(device:{} at 0x{:x})", self.device_index(), self.handle_address())
        }
    }
}

impl std::fmt::Debug for MemoryPool {
    // simply call display
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl Default for MemoryPool {
    // default is global pool
    fn default() -> Self {
        Self::global_pool()
    }
}

impl std::cmp::PartialEq for MemoryPool {
    fn eq(&self, other: &Self) -> bool {
        self.handle_address() == other.handle_address()
    }
}

impl Clone for MemoryPool {
    fn clone(&self) -> Self {
        let p = ffi::memory_pool_constructor_copy(&self.p);
        Self { p }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn addresses() {
        let device_count = crate::basics::device_count();
        if device_count == 0 {
            return;
        }
        // nullptr is zero
        assert!(MemoryPool::nullptr().handle_address() == 0);
        // global pool's address is not zero
        let global_pool = MemoryPool::global_pool();
        assert!(MemoryPool::global_pool().handle_address() != 0);
        // get another global pool, should have same address
        assert!(MemoryPool::global_pool().handle_address() == global_pool.handle_address());
        // all other pool is not zero and not global pool
        for i in 0..device_count {
            let pool = MemoryPool::new(i);
            assert!(pool.handle_address() != 0);
            assert!(pool.handle_address() != global_pool.handle_address());
        }
    }

}