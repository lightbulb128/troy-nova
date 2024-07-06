use super::ffi;

pub fn device_count() -> usize {
    return ffi::device_count();
}

// SchemeType

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct SchemeType(std::ffi::c_int);

unsafe impl cxx::ExternType for SchemeType {
    type Kind = cxx::kind::Trivial;
    type Id = cxx::type_id!("troy_rust::SchemeType");
}

#[allow(non_upper_case_globals)]
impl SchemeType {
    pub const Nil: Self = Self(0);
    pub const BFV: Self = Self(1);
    pub const CKKS: Self = Self(2);
    pub const BGV: Self = Self(3);
}

impl SchemeType {
    pub fn to_string(&self) -> String {
        return ffi::scheme_type_to_string(*self);
    }
}

impl std::fmt::Display for SchemeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// SecurityLevel


#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct SecurityLevel(std::ffi::c_int);

unsafe impl cxx::ExternType for SecurityLevel {
    type Kind = cxx::kind::Trivial;
    type Id = cxx::type_id!("troy_rust::SecurityLevel");
}

#[allow(non_upper_case_globals)]
impl SecurityLevel {
    pub const Nil: Self = Self(0);
    pub const Classical128: Self = Self(128);
    pub const Classical192: Self = Self(192);
    pub const Classical256: Self = Self(256);
}

impl SecurityLevel {
    pub fn to_string(&self) -> String {
        return ffi::security_level_to_string(*self);
    }
}

impl std::fmt::Display for SecurityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn scheme_type_to_string() {
        assert_eq!(super::SchemeType::Nil.to_string(), "Nil");
        assert_eq!(super::SchemeType::BFV.to_string(), "BFV");
        assert_eq!(super::SchemeType::CKKS.to_string(), "CKKS");
        assert_eq!(super::SchemeType::BGV.to_string(), "BGV");
    }

    #[test]
    fn security_level_to_string() {
        assert_eq!(super::SecurityLevel::Nil.to_string(), "Nil");
        assert_eq!(super::SecurityLevel::Classical128.to_string(), "Classical128");
        assert_eq!(super::SecurityLevel::Classical192.to_string(), "Classical192");
        assert_eq!(super::SecurityLevel::Classical256.to_string(), "Classical256");
    }

}