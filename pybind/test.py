import pytroy
from pytroy import Modulus, CoeffModulus, PlainModulus, EncryptionParameters, SchemeType
from pytroy import BatchEncoder, CKKSEncoder, ParmsID, Plaintext, Ciphertext, HeContext, SecurityLevel
from pytroy import KeyGenerator, Encryptor, Decryptor, Evaluator, RelinKeys, GaloisKeys
import typing
import unittest
import numpy as np
import argparse

class TestBasics(unittest.TestCase):

    def test_hello(self):
        self.assertEqual(pytroy.it_works(), 42)

    def test_modulus(self):
        modulus = pytroy.Modulus(1234)
        self.assertEqual(modulus.value(), 1234)
        
        bit_sizes = [60, 40, 40, 60]
        moduli = CoeffModulus.create(8192, bit_sizes)
        for i in range(len(moduli)):
            self.assertEqual(moduli[i].bit_count(), bit_sizes[i])

    def test_encryption_parameters(self):
        modulus = CoeffModulus.create(8192, [60, 40, 40, 60])
        params = EncryptionParameters(SchemeType.BFV)
        params.set_poly_modulus_degree(8192)
        params.set_coeff_modulus(modulus)
        t = PlainModulus.batching(8192, 20)
        params.set_plain_modulus(t)
        self.assertEqual(params.scheme(), SchemeType.BFV)
        self.assertEqual(params.poly_modulus_degree(), 8192)
        for i in range(len(modulus)):
            self.assertEqual(params.coeff_modulus()[i].value(), modulus[i].value())
        self.assertEqual(params.plain_modulus().value(), t.value())
        
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

class HeTest:

    def __init__(self, ghe: GeneralHeContext, tester: unittest.TestCase):
        self.ghe = ghe
        self.tester = tester

    def test_encode_simd(self):
        ghe = self.ghe
        message = ghe.random_simd_full()
        plain = ghe.encoder.encode_simd(message)
        decoded = ghe.encoder.decode_simd(plain)
        self.tester.assertTrue(message.near_equal(decoded, ghe.tolerance))

    def test_encode_polynomial(self):
        ghe = self.ghe
        message = ghe.random_polynomial_full()
        plain = ghe.encoder.encode_polynomial(message)
        decoded = ghe.encoder.decode_polynomial(plain)
        self.tester.assertTrue(message.near_equal(decoded, ghe.tolerance))

    def test_encrypt(self):
        ghe = self.ghe
        message = ghe.random_simd_full()

        plain = ghe.encoder.encode_simd(message)
        cipher = ghe.encryptor.encrypt_asymmetric_new(plain)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(cipher))
        self.tester.assertTrue(message.near_equal(decoded, ghe.tolerance))

        cipher = ghe.encryptor.encrypt_symmetric_new(plain, False)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(cipher))

    def test_encrypt_serialize(self):
        ghe = self.ghe
        message = ghe.random_simd_full()
        plain = ghe.encoder.encode_simd(message)
        
        cipher = ghe.encryptor.encrypt_asymmetric_new(plain)
        serialized = cipher.save(ghe.context)
        cipher = Ciphertext.load_new(serialized, ghe.context)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(cipher))
        self.tester.assertTrue(message.near_equal(decoded, ghe.tolerance))

        cipher = ghe.encryptor.encrypt_symmetric_new(plain, True)
        serialized = cipher.save(ghe.context)
        cipher = Ciphertext.load_new(serialized, ghe.context)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(cipher))
        self.tester.assertTrue(message.near_equal(decoded, ghe.tolerance))

        terms = [1, 3, 5, 7]
        message = ghe.random_polynomial_full()
        plain = ghe.encoder.encode_polynomial(message)
        cipher = ghe.encryptor.encrypt_asymmetric_new(plain)
        serialized = cipher.save_terms(ghe.context, terms)
        cipher = Ciphertext.load_terms_new(serialized, ghe.context, terms)
        decoded = ghe.encoder.decode_polynomial(ghe.decryptor.decrypt_new(cipher))
        for term in terms:
            self.tester.assertTrue(message.element(term).near_equal(decoded.element(term), ghe.tolerance))
        
        cipher = ghe.encryptor.encrypt_symmetric_new(plain, True)
        serialized = cipher.save_terms(ghe.context, terms)
        cipher = Ciphertext.load_terms_new(serialized, ghe.context, terms)
        decoded = ghe.encoder.decode_polynomial(ghe.decryptor.decrypt_new(cipher))
        for term in terms:
            self.tester.assertTrue(message.element(term).near_equal(decoded.element(term), ghe.tolerance))

    def test_negate(self):
        ghe = self.ghe
        message = ghe.random_simd_full()
        plain = ghe.encoder.encode_simd(message)
        cipher = ghe.encryptor.encrypt_symmetric_new(plain, False)
        negated = ghe.evaluator.negate_new(cipher)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(negated))
        self.tester.assertTrue(ghe.near_equal(ghe.negate(message), decoded))
    
    def test_add_sub(self):
        ghe = self.ghe
        message1 = ghe.random_simd_full()
        message2 = ghe.random_simd_full()
        plain1 = ghe.encoder.encode_simd(message1)
        plain2 = ghe.encoder.encode_simd(message2)
        cipher1 = ghe.encryptor.encrypt_symmetric_new(plain1, False)
        cipher2 = ghe.encryptor.encrypt_symmetric_new(plain2, False)
        added = ghe.evaluator.add_new(cipher1, cipher2)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(added))
        self.tester.assertTrue(ghe.near_equal(ghe.add(message1, message2), decoded))
        subtracted = ghe.evaluator.sub_new(cipher1, cipher2)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(subtracted))
        self.tester.assertTrue(ghe.near_equal(ghe.sub(message1, message2), decoded))
        
    def test_multiply_relinearize(self):
        ghe = self.ghe
        message1 = ghe.random_simd_full()
        message2 = ghe.random_simd_full()
        message3 = ghe.random_simd_full()
        plain1 = ghe.encoder.encode_simd(message1)
        plain2 = ghe.encoder.encode_simd(message2)
        cipher1 = ghe.encryptor.encrypt_symmetric_new(plain1, False)
        cipher2 = ghe.encryptor.encrypt_symmetric_new(plain2, False)
        multiplied = ghe.evaluator.multiply_new(cipher1, cipher2)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(multiplied))
        self.tester.assertTrue(ghe.near_equal(ghe.mul(message1, message2), decoded))

        plain3 = ghe.encoder.encode_simd(message3, None, multiplied.scale())
        cipher3 = ghe.encryptor.encrypt_symmetric_new(plain3, False)
        multiply_added = ghe.evaluator.add_new(multiplied, cipher3)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(multiply_added))
        self.tester.assertTrue(ghe.near_equal(ghe.add(ghe.mul(message1, message2), message3), decoded))

        relin_keys = ghe.key_generator.create_relin_keys(False)
        relinearized = ghe.evaluator.relinearize_new(multiplied, relin_keys)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(relinearized))
        self.tester.assertTrue(ghe.near_equal(ghe.mul(message1, message2), decoded))

        squared = ghe.evaluator.square_new(cipher1)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(squared))
        self.tester.assertTrue(ghe.near_equal(ghe.square(message1), decoded))
    
    def test_keyswitching(self):
        ghe = self.ghe
        keygen_other = KeyGenerator(ghe.context)
        secret_key_other = keygen_other.secret_key()
        encryptor_other = Encryptor(ghe.context)
        encryptor_other.set_secret_key(secret_key_other)
        kswitch_key = ghe.key_generator.create_keyswitching_key(secret_key_other, False)

        message = ghe.random_simd_full()
        plain = ghe.encoder.encode_simd(message)
        cipher = encryptor_other.encrypt_symmetric_new(plain, False)
        switched = ghe.evaluator.apply_keyswitching_new(cipher, kswitch_key)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(switched))
        self.tester.assertTrue(ghe.near_equal(message, decoded))

    def test_mod_switch_to_next(self):
        ghe = self.ghe
        message = ghe.random_simd_full()
        plain = ghe.encoder.encode_simd(message)
        cipher = ghe.encryptor.encrypt_symmetric_new(plain, False)
        ghe.evaluator.mod_switch_to_next_inplace(cipher)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(cipher))
        self.tester.assertTrue(ghe.near_equal(message, decoded))

        if ghe.is_ckks:
            ghe.evaluator.mod_switch_plain_to_next_inplace(plain)
            decoded = ghe.encoder.decode_simd(plain)
            self.tester.assertTrue(ghe.near_equal(message, decoded))

            parms = ghe.parms
            coeff_modulus = parms.coeff_modulus()
            expanded_scale = ghe.scale * coeff_modulus[len(coeff_modulus) - 2].value()
            encoded = ghe.encoder.encode_simd(message, None, expanded_scale)
            encrypted = ghe.encryptor.encrypt_symmetric_new(encoded, False)
            ghe.evaluator.mod_switch_to_next_inplace(encrypted)
            decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(encrypted))
            self.tester.assertTrue(ghe.near_equal(message, decoded))

    def test_add_sub_plain(self):
        ghe = self.ghe
        message1 = ghe.random_simd_full()
        message2 = ghe.random_simd_full()
        plain1 = ghe.encoder.encode_simd(message1)
        plain2 = ghe.encoder.encode_simd(message2)
        cipher1 = ghe.encryptor.encrypt_symmetric_new(plain1, False)
        added = ghe.evaluator.add_plain_new(cipher1, plain2)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(added))
        self.tester.assertTrue(ghe.near_equal(ghe.add(message1, message2), decoded))
        subtracted = ghe.evaluator.sub_plain_new(cipher1, plain2)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(subtracted))
        self.tester.assertTrue(ghe.near_equal(ghe.sub(message1, message2), decoded))

    def test_multiply_plain(self):
        ghe = self.ghe
        message1 = ghe.random_simd_full()
        message2 = ghe.random_simd_full()
        plain1 = ghe.encoder.encode_simd(message1)
        plain2 = ghe.encoder.encode_simd(message2)
        cipher1 = ghe.encryptor.encrypt_symmetric_new(plain1, False)
        multiplied = ghe.evaluator.multiply_plain_new(cipher1, plain2)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(multiplied))
        self.tester.assertTrue(ghe.near_equal(ghe.mul(message1, message2), decoded))

        if not cipher1.is_ntt_form():
            ghe.evaluator.transform_to_ntt_inplace(cipher1)
            ghe.evaluator.transform_plain_to_ntt_inplace(plain2, cipher1.parms_id())
            multiplied = ghe.evaluator.multiply_plain_new(cipher1, plain2)
            ghe.evaluator.transform_from_ntt_inplace(multiplied)
            decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(multiplied))
            self.tester.assertTrue(ghe.near_equal(ghe.mul(message1, message2), decoded))
        else:
            ghe.evaluator.transform_from_ntt_inplace(cipher1)
            ghe.evaluator.transform_to_ntt_inplace(cipher1)
            multiplied = ghe.evaluator.multiply_plain_new(cipher1, plain2)
            decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(multiplied))
            self.tester.assertTrue(ghe.near_equal(ghe.mul(message1, message2), decoded))

    def test_rotate(self):
        ghe = self.ghe
        message = ghe.random_simd_full()
        plain = ghe.encoder.encode_simd(message)
        cipher = ghe.encryptor.encrypt_symmetric_new(plain, False)
        glk = ghe.key_generator.create_galois_keys(False)
        for step in [1, 7]:
            if ghe.is_ckks:
                rotated = ghe.evaluator.rotate_vector_new(cipher, step, glk)
            else:
                rotated = ghe.evaluator.rotate_rows_new(cipher, step, glk)
            decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(rotated))
            self.tester.assertTrue(ghe.near_equal(ghe.rotate(message, step), decoded))
            
    def test_conjugate(self):
        ghe = self.ghe
        message = ghe.random_simd_full()
        plain = ghe.encoder.encode_simd(message)
        cipher = ghe.encryptor.encrypt_symmetric_new(plain, False)
        glk = ghe.key_generator.create_galois_keys(False)
        if ghe.is_ckks:
            rotated = ghe.evaluator.complex_conjugate_new(cipher, glk)
        else:
            rotated = ghe.evaluator.rotate_columns_new(cipher, glk)
        decoded = ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(rotated))
        self.tester.assertTrue(ghe.near_equal(ghe.conjugate(message), decoded))

    def test_extract_lwe(self):
        ghe = self.ghe
        message = ghe.random_polynomial_full()
        plain = ghe.encoder.encode_polynomial(message)
        cipher = ghe.encryptor.encrypt_symmetric_new(plain, False)
        
        for term in [0, 1, 3, 7]:
            extracted = ghe.evaluator.extract_lwe_new(cipher, term)
            assembled = ghe.evaluator.assemble_lwe_new(extracted)
            if ghe.is_ckks:
                ghe.evaluator.transform_to_ntt_inplace(assembled)
            decoded = ghe.encoder.decode_polynomial(ghe.decryptor.decrypt_new(assembled))
            self.tester.assertTrue(ghe.near_equal(message.element(term), decoded.element(0)))

    def test_pack_lwe(self):
        ghe = self.ghe
        if ghe.parms.poly_modulus_degree() != 32:
            return
        ak = self.ghe.key_generator.create_automorphism_keys(False)
        message = ghe.random_polynomial_full()
        plain = ghe.encoder.encode_polynomial(message)
        cipher = ghe.encryptor.encrypt_symmetric_new(plain, False)
        
        # pack 32 lwes
        extracted = []
        for i in range(32):
            extracted.append(ghe.evaluator.extract_lwe_new(cipher, i))
        assembled = ghe.evaluator.pack_lwe_ciphertexts_new(extracted, ak)
        decoded = ghe.encoder.decode_polynomial(ghe.decryptor.decrypt_new(assembled))
        self.tester.assertTrue(ghe.near_equal(message, decoded))

        # pack 7 lwes
        for i in range(32):
            if i % 4 == 0 and i // 4 < 7:
                continue
            message.data[i] = 0;
        extracted = []
        for i in range(7):
            extracted.append(ghe.evaluator.extract_lwe_new(cipher, i * 4))
        assembled = ghe.evaluator.pack_lwe_ciphertexts_new(extracted, ak)
        decoded = ghe.encoder.decode_polynomial(ghe.decryptor.decrypt_new(assembled))
        self.tester.assertTrue(ghe.near_equal(message, decoded))

def create_test_class(ghe: GeneralHeContext):

    class UnnamedClass(unittest.TestCase):

        def setUp(self) -> None:
            self.ghe = ghe
            self.tester = HeTest(self.ghe, self)
            return super().setUp()
        def test_setup_ok(self):
            pass
        def test_encode_simd(self):
            self.tester.test_encode_simd()
        def test_encode_polynomial(self):
            self.tester.test_encode_polynomial()
        def test_encrypt(self):
            self.tester.test_encrypt()
        def test_encrypt_serialize(self):
            self.tester.test_encrypt_serialize()
        def test_negate(self):
            self.tester.test_negate()
        def test_add_sub(self):
            self.tester.test_add_sub()
        def test_multiply_relinearize(self):
            self.tester.test_multiply_relinearize()
        def test_keyswitching(self):
            self.tester.test_keyswitching()
        def test_mod_switch_to_next(self):
            self.tester.test_mod_switch_to_next()
        def test_add_sub_plain(self):
            self.tester.test_add_sub_plain()
        def test_multiply_plain(self):
            self.tester.test_multiply_plain()
        def test_rotate(self):
            self.tester.test_rotate()
        def test_conjugate(self):
            self.tester.test_conjugate()
        def test_extract_lwe(self):
            self.tester.test_extract_lwe()
        def test_pack_lwe(self):
            self.tester.test_pack_lwe()

    return UnnamedClass

TestBFVHost = create_test_class(GeneralHeContext(False, SchemeType.BFV, 32, 20, [60, 40, 40, 60], True, 0x123))
TestCKKSHost = create_test_class(GeneralHeContext(False, SchemeType.CKKS, 32, 0, [60, 40, 40, 60], True, 0x123, 10, 1<<20, 1e-2))
TestBGVHost = create_test_class(GeneralHeContext(False, SchemeType.BGV, 32, 20, [60, 40, 40, 60], True, 0x123))

class HostTestSuite(unittest.TestSuite):

    def __init__(self):
        super().__init__()
        self.addTest(unittest.makeSuite(TestBFVHost))
        self.addTest(unittest.makeSuite(TestCKKSHost))
        self.addTest(unittest.makeSuite(TestBGVHost))

TestBFVDevice = create_test_class(GeneralHeContext(True, SchemeType.BFV, 32, 20, [60, 40, 40, 60], True, 0x123))
TestCKKSDevice = create_test_class(GeneralHeContext(True, SchemeType.CKKS, 32, 0, [60, 40, 40, 60], True, 0x123, 10, 1<<20, 1e-2))
TestBGVDevice = create_test_class(GeneralHeContext(True, SchemeType.BGV, 32, 20, [60, 40, 40, 60], True, 0x123))

class DeviceTestSuite(unittest.TestSuite):
    
    def __init__(self):
        super().__init__()
        self.addTest(unittest.makeSuite(TestBFVDevice))
        self.addTest(unittest.makeSuite(TestCKKSDevice))
        self.addTest(unittest.makeSuite(TestBGVDevice))

def custom_main():
    print("There is nothing here.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", action="store_true")

    args = parser.parse_args()

    pytroy.initialize_kernel(0)

    if args.custom:

        custom_main()

    else:

        # run basic tests
        print("Running basic tests")
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestBasics))
        unittest.TextTestRunner().run(suite)
        print("")

        # run host suite
        print("Running host test suite")
        suite = HostTestSuite()
        unittest.TextTestRunner().run(suite)
        print("")

        # run device suite
        print("Running device test suite")
        suite = DeviceTestSuite()
        unittest.TextTestRunner().run(suite)
        print("")

    pytroy.destroy_memory_pool()
