import numpy as np
import pytroy
from pytroy import CKKSEncoder, BatchEncoder, SchemeType, CoeffModulus, ParmsID, SecurityLevel, EncryptionParameters
from pytroy import HeContext, KeyGenerator, Encryptor, Decryptor, Evaluator, Plaintext
import typing

class GeneralVectorDataType:
    Complexes = 0
    Integers = 1
    Doubles = 2

class GeneralVector:

    def __init__(self, data_type: GeneralVectorDataType, data: np.ndarray):
        self.data_type = data_type
        self.data = data

    def size(self):
        return self.data.size
    
    def is_complexes(self):
        return self.data_type == GeneralVectorDataType.Complexes
    
    def is_integers(self):
        return self.data_type == GeneralVectorDataType.Integers
    
    def is_doubles(self):
        return self.data_type == GeneralVectorDataType.Doubles

    @classmethod
    def random_complexes(cls, size: int, component_max: float):
        reals = np.random.rand(size) * component_max * 2 - component_max
        imags = np.random.rand(size) * component_max * 2 - component_max
        return cls(GeneralVectorDataType.Complexes, reals + 1j * imags)
    
    @classmethod
    def random_integers(cls, size: int, modulus: int):
        data = np.random.randint(0, modulus, size, dtype=np.uint64)
        return cls(GeneralVectorDataType.Integers, data)

    @classmethod
    def random_doubles(cls, size: int, component_max: float):
        data = np.random.rand(size) * component_max * 2 - component_max
        return cls(GeneralVectorDataType.Doubles, data)
    
    def subvector(self, low: int, high: int):
        return GeneralVector(self.data_type, self.data[low:high].copy())
    
    def element(self, index: int):
        return self.subvector(index, index + 1)
    
    def negate(self, modulus: int) -> 'GeneralVector':
        if self.data_type == GeneralVectorDataType.Complexes:
            return GeneralVector(self.data_type, -self.data)
        elif self.data_type == GeneralVectorDataType.Integers:
            return GeneralVector(self.data_type, np.where(self.data == 0, 0, modulus - self.data))
        elif self.data_type == GeneralVectorDataType.Doubles:
            return GeneralVector(self.data_type, -self.data)
        else:
            raise RuntimeError("[GeneralVector.negate] Unknown data type")
        
    def add(self, other: 'GeneralVector', modulus: int) -> 'GeneralVector':
        if self.data_type != other.data_type:
            raise RuntimeError("[GeneralVector.add] Data type mismatch")
        if self.data_type == GeneralVectorDataType.Complexes:
            return GeneralVector(self.data_type, self.data + other.data)
        elif self.data_type == GeneralVectorDataType.Integers:
            return GeneralVector(self.data_type, (self.data + other.data) % modulus)
        elif self.data_type == GeneralVectorDataType.Doubles:
            return GeneralVector(self.data_type, self.data + other.data)
        else:
            raise RuntimeError("[GeneralVector.add] Unknown data type")
        
    def sub(self, other: 'GeneralVector', modulus: int) -> 'GeneralVector':
        return self.add(other.negate(modulus), modulus)
    
    def mul(self, other: 'GeneralVector', modulus: int) -> 'GeneralVector':
        if self.data_type != other.data_type:
            raise RuntimeError("[GeneralVector.mul] Data type mismatch")
        if self.data_type == GeneralVectorDataType.Complexes:
            return GeneralVector(self.data_type, self.data * other.data)
        elif self.data_type == GeneralVectorDataType.Integers:
            data = np.zeros_like(self.data)
            for i in range(self.size()):
                data[i] = int(self.data[i]) * int(other.data[i]) % modulus
            return GeneralVector(self.data_type, data)
        elif self.data_type == GeneralVectorDataType.Doubles:
            return GeneralVector(self.data_type, self.data * other.data)
        else:
            raise RuntimeError("[GeneralVector.mul] Unknown data type")
        
    def square(self, modulus: int) -> 'GeneralVector':
        return self.mul(self, modulus)
    
    def near_equal(self, other: 'GeneralVector', tolerance: float) -> 'GeneralVector':
        if self.data_type != other.data_type:
            raise RuntimeError("[GeneralVector.near_equal] Data type mismatch")
        if self.data_type == GeneralVectorDataType.Complexes:
            return np.allclose(self.data, other.data, atol=tolerance)
        elif self.data_type == GeneralVectorDataType.Integers:
            return np.all(self.data - other.data == 0)
        elif self.data_type == GeneralVectorDataType.Doubles:
            return np.allclose(self.data, other.data, atol=tolerance)
        else:
            raise RuntimeError("[GeneralVector.near_equal] Unknown data type")
        
    def rotate(self, steps: int) -> 'GeneralVector':
        if self.data_type == GeneralVectorDataType.Complexes:
            return GeneralVector(self.data_type, np.roll(self.data, -steps))
        elif self.data_type == GeneralVectorDataType.Integers:
            data = np.zeros_like(self.data)
            # rotate by left half and right half
            half = self.size() // 2
            for i in range(half):
                data[i] = self.data[(i + steps) % half]
                data[i + half] = self.data[(i + half + steps) % half + half]
            return GeneralVector(self.data_type, data)
        elif self.data_type == GeneralVectorDataType.Doubles:
            raise RuntimeError("[GeneralVector.rotate] Rotation not supported for doubles")
        else:
            raise RuntimeError("[GeneralVector.rotate] Unknown data type")
        
    def conjugate(self) -> 'GeneralVector':
        if self.data_type == GeneralVectorDataType.Complexes:
            return GeneralVector(self.data_type, np.conj(self.data))
        elif self.data_type == GeneralVectorDataType.Integers:
            # swap left half and right half
            half = self.size() // 2
            data = np.zeros_like(self.data)
            data[:half] = self.data[half:]
            data[half:] = self.data[:half]
            return GeneralVector(self.data_type, data)
        elif self.data_type == GeneralVectorDataType.Doubles:
            raise RuntimeError("[GeneralVector.conjugate] Conjugate not supported for doubles")
        else:
            raise RuntimeError("[GeneralVector.conjugate] Unknown data type")
        
    def __str__(self) -> str:
        return str(self.data)

class GeneralEncoder:

    def __init__(self, encoder: typing.Union[CKKSEncoder, BatchEncoder]):
        self.encoder = encoder
        self.is_ckks = isinstance(encoder, CKKSEncoder)

    def to_device_inplace(self):
        self.encoder.to_device_inplace()

    def slot_count(self) -> int:
        return self.encoder.slot_count()
    
    def encode_simd(self, vec: GeneralVector, parms_id: typing.Union[ParmsID, None] = None, scale: float = 1<<20) -> Plaintext:
        if vec.is_complexes():
            return self.encoder.encode_complex64_simd_new(vec.data, parms_id, scale)
        elif vec.is_integers():
            return self.encoder.encode_simd_new(vec.data)
        else:
            raise RuntimeError("[GeneralEncoder.encode_simd] Unknown data type")
    
    def encode_polynomial(self, vec: GeneralVector, parms_id: typing.Union[ParmsID, None] = None, scale: float = 1<<20) -> Plaintext:
        if vec.is_doubles():
            return self.encoder.encode_float64_polynomial_new(vec.data, parms_id, scale)
        elif vec.is_integers():
            return self.encoder.encode_polynomial_new(vec.data)
        else:
            raise RuntimeError("[GeneralEncoder.encode_polynomial] Unknown data type")
        
    def decode_simd(self, plain: Plaintext) -> GeneralVector:
        if self.is_ckks:
            return GeneralVector(GeneralVectorDataType.Complexes, self.encoder.decode_complex64_simd_new(plain))
        else:
            return GeneralVector(GeneralVectorDataType.Integers, self.encoder.decode_simd_new(plain))
        
    def decode_polynomial(self, plain: Plaintext) -> GeneralVector:
        if self.is_ckks:
            return GeneralVector(GeneralVectorDataType.Doubles, self.encoder.decode_float64_polynomial_new(plain))
        else:
            return GeneralVector(GeneralVectorDataType.Integers, self.encoder.decode_polynomial_new(plain))
        
    def random_simd(self, size: int, t: int, max: float) -> GeneralVector:
        if self.is_ckks:
            return GeneralVector.random_complexes(size, max)
        else:
            return GeneralVector.random_integers(size, t)
    
    def random_simd_full(self, t: int, max: float) -> GeneralVector:
        return self.random_simd(self.slot_count(), t, max)
    
    def random_polynomial(self, size: int, t: int, max: float) -> GeneralVector:
        if self.is_ckks:
            return GeneralVector.random_doubles(size, max)
        else:
            return GeneralVector.random_integers(size, t)
        
    def random_polynomial_full(self, t: int, max: float) -> GeneralVector:
        if self.is_ckks:
            return self.random_polynomial(self.slot_count() * 2, t, max)
        else:
            return self.random_polynomial(self.slot_count(), t, max)

class GeneralHeContext:

    def __init__(self, 
        device: bool, scheme: SchemeType, 
        n: int, log_t: int, log_qi: typing.List[int], 
        expand_mod_chain: bool, seed: int, 
        input_max: float = 0, scale: float = 0, tolerance: float = 0,
        to_device_after_key_generation: bool = False
    ):
        if scheme == SchemeType.CKKS and (input_max == 0 or scale == 0 or tolerance == 0):
            raise RuntimeError("[GeneralHeContext.__init__] CKKS requires input_max and scale")
        # create enc params
        parms = EncryptionParameters(scheme)
        parms.set_poly_modulus_degree(n)
        # create modulus
        if scheme != SchemeType.CKKS:
            log_qi.append(log_t)
            moduli = CoeffModulus.create(n, log_qi)
            plain_modulus = moduli[len(moduli) - 1]
            coeff_moduli = list(moduli[:-1])
            parms.set_plain_modulus(plain_modulus)
            parms.set_coeff_modulus(coeff_moduli)
        else:
            moduli = CoeffModulus.create(n, log_qi)
            parms.set_coeff_modulus(moduli)
        self.parms = parms
        self.scheme = scheme
        # create gadgets
        context = HeContext(parms, expand_mod_chain, SecurityLevel.Nil, seed)
        if scheme == SchemeType.CKKS:
            encoder = CKKSEncoder(context)
        else:
            encoder = BatchEncoder(context)
        encoder = GeneralEncoder(encoder)
        if device and not to_device_after_key_generation:
            context.to_device_inplace()
            encoder.to_device_inplace()
        key_generator = KeyGenerator(context)
        public_key = key_generator.create_public_key(False)
        encryptor = Encryptor(context)
        encryptor.set_public_key(public_key)
        encryptor.set_secret_key(key_generator.secret_key())
        decryptor = Decryptor(context, key_generator.secret_key())
        evaluator = Evaluator(context)
        t = 0 if scheme == SchemeType.CKKS else parms.plain_modulus().value()
        
        if device and to_device_after_key_generation:
            context.to_device_inplace()
            encoder.to_device_inplace()
            key_generator.secret_key().to_device_inplace()
            encryptor.to_device_inplace()
            decryptor.to_device_inplace()
        
        self.context = context
        self.encoder = encoder
        self.key_generator = key_generator
        self.encryptor = encryptor
        self.decryptor = decryptor
        self.evaluator = evaluator
        self.t = t
        self.input_max = input_max
        self.scale = scale
        self.tolerance = tolerance
        self.is_ckks = scheme == SchemeType.CKKS
        self.is_ntt = scheme == SchemeType.CKKS or scheme == SchemeType.BGV
        self.device = device

    def random_simd(self, size: int) -> GeneralVector:
        return self.encoder.random_simd(size, self.t, self.input_max)
    def random_simd_full(self) -> GeneralVector:
        return self.encoder.random_simd_full(self.t, self.input_max)
    def random_polynomial(self, size: int) -> GeneralVector:
        return self.encoder.random_polynomial(size, self.t, self.input_max)
    def random_polynomial_full(self) -> GeneralVector:
        return self.encoder.random_polynomial_full(self.t, self.input_max)
    def add(self, a: GeneralVector, b: GeneralVector) -> GeneralVector:
        return a.add(b, self.t)
    def sub(self, a: GeneralVector, b: GeneralVector) -> GeneralVector:
        return a.sub(b, self.t)
    def mul(self, a: GeneralVector, b: GeneralVector) -> GeneralVector:
        return a.mul(b, self.t)
    def square(self, a: GeneralVector) -> GeneralVector:
        return a.square(self.t)
    def negate(self, a: GeneralVector) -> GeneralVector:
        return a.negate(self.t)
    def rotate(self, a: GeneralVector, steps: int) -> GeneralVector:
        return a.rotate(steps)
    def conjugate(self, a: GeneralVector) -> GeneralVector:
        return a.conjugate()
    def near_equal(self, a: GeneralVector, b: GeneralVector) -> bool:
        return a.near_equal(b, self.tolerance)
