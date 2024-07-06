use super::ffi;

type UpParmsID = cxx::UniquePtr<ffi::ParmsID>;

pub struct ParmsID {
    p: UpParmsID,
}

impl ParmsID {

    #[inline]
    pub fn new() -> Self {
        let p = ffi::parms_id_static_zero();
        Self { p }
    }

    #[inline]
    pub fn to_array(&self) -> [u64; 4] {
        self.p.to_array()
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.p.is_zero()
    }

}

impl Default for ParmsID {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ParmsID {
    fn clone(&self) -> Self {
        let p = ffi::parms_id_constructor_copy(&self.p);
        Self { p }
    }
}

impl std::fmt::Debug for ParmsID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let a = self.to_array();
        // print 4 uint64s as hex
        write!(f, "ParmsID({:016x} {:016x} {:016x} {:016x})", a[0], a[1], a[2], a[3])
    }
}

// display same as debug
impl std::fmt::Display for ParmsID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl std::cmp::PartialEq for ParmsID {
    fn eq(&self, other: &Self) -> bool {
        self.p.equals_to(other.p.as_ref().expect("nullptr"))
    }
}

impl std::hash::Hash for ParmsID {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let a = self.to_array();
        a.hash(state);
    }
}

impl std::cmp::Eq for ParmsID {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parms_id() {
        let p = ParmsID::new();
        assert!(p.is_zero());
        let a = p.to_array();
        assert_eq!(a, [0, 0, 0, 0]);
        let q = p.clone();
        assert_eq!(p, q);
    }

    #[test]
    fn test_can_create_hash_map() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let p = ParmsID::new();
        map.insert(p, 1);
        assert_eq!(map.len(), 1);
    }

}