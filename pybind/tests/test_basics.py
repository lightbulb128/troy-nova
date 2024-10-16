import unittest
import pytroy
from pytroy import CoeffModulus, SchemeType
from pytroy import PlainModulus, EncryptionParameters
from pytroy import BatchEncoder, HeContext
from pytroy import KeyGenerator, Encryptor, MemoryPool

class Basics(unittest.TestCase):

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

    def test_global_pool(self):
        params = EncryptionParameters(SchemeType.BFV)
        params.set_coeff_modulus(CoeffModulus.create(8192, [60, 40, 40, 60]))
        params.set_plain_modulus(PlainModulus.batching(8192, 20))
        params.set_poly_modulus_degree(8192)
        context = HeContext(params)
        encoder = BatchEncoder(context)
        context.to_device_inplace()
        encoder.to_device_inplace()
        keygen = KeyGenerator(context)
        encryptor = Encryptor(context)
        public_key = keygen.create_public_key(False)
        encryptor.set_public_key(public_key)
        global_pool = MemoryPool.global_pool()
        self.assertNotEqual(global_pool, None)
        self.assertEqual(context.pool(), global_pool)
        self.assertEqual(public_key.pool(), global_pool)
        encoded = encoder.encode_simd_new([1, 2, 3, 4])
        self.assertEqual(encoded.pool(), global_pool)
        encrypted = encryptor.encrypt_asymmetric_new(encoded)
        self.assertEqual(encrypted.pool(), global_pool)

    def test_host_pool(self):
        params = EncryptionParameters(SchemeType.BFV)
        params.set_coeff_modulus(CoeffModulus.create(8192, [60, 40, 40, 60]))
        params.set_plain_modulus(PlainModulus.batching(8192, 20))
        params.set_poly_modulus_degree(8192)
        context = HeContext(params)
        encoder = BatchEncoder(context)
        keygen = KeyGenerator(context)
        encryptor = Encryptor(context)
        public_key = keygen.create_public_key(False)
        encryptor.set_public_key(public_key)
        self.assertEqual(context.pool(), None)
        self.assertEqual(public_key.pool(), None)
        encoded = encoder.encode_simd_new([1, 2, 3, 4])
        self.assertEqual(encoded.pool(), None)
        encrypted = encryptor.encrypt_asymmetric_new(encoded)
        self.assertEqual(encrypted.pool(), None)

    def test_custom_pool(self):
        params = EncryptionParameters(SchemeType.BFV)
        params.set_coeff_modulus(CoeffModulus.create(8192, [60, 40, 40, 60]))
        params.set_plain_modulus(PlainModulus.batching(8192, 20))
        params.set_poly_modulus_degree(8192)
        global_pool = MemoryPool.global_pool()
        context_pool = MemoryPool()
        self.assertNotEqual(context_pool, None)
        self.assertNotEqual(context_pool, global_pool)
        context = HeContext(params)
        encoder = BatchEncoder(context)
        context.to_device_inplace(context_pool)
        encoder.to_device_inplace(context_pool)
        keygen = KeyGenerator(context)
        encryptor = Encryptor(context)
        public_key = keygen.create_public_key(False, context_pool)
        encryptor.set_public_key(public_key)
        self.assertEqual(context.pool(), context_pool)
        self.assertEqual(public_key.pool(), context_pool)
        text_pool = MemoryPool()
        self.assertNotEqual(text_pool, None)
        self.assertNotEqual(text_pool, context_pool)
        self.assertNotEqual(text_pool, global_pool)
        encoded = encoder.encode_simd_new([1, 2, 3, 4], text_pool)
        self.assertEqual(encoded.pool(), text_pool)
        encrypted = encryptor.encrypt_asymmetric_new(encoded, text_pool)
        self.assertEqual(encrypted.pool(), text_pool)

def get_suite():
    suite = unittest.makeSuite(Basics)
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(get_suite())