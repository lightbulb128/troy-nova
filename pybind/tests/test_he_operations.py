import pytroy
from pytroy import SchemeType
from pytroy import Ciphertext
from pytroy import KeyGenerator, Encryptor
import typing
import unittest
import numpy as np
import argparse
from utils import GeneralHeContext, GeneralVector

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

        # test multiply plain batched
        batch_size = 16
        message1 = [ghe.random_simd_full() for _ in range(batch_size)]
        message2 = [ghe.random_simd_full() for _ in range(batch_size)]
        plain1 = [ghe.encoder.encode_simd(message1[i]) for i in range(batch_size)]
        plain2 = [ghe.encoder.encode_simd(message2[i]) for i in range(batch_size)]
        cipher1 = [ghe.encryptor.encrypt_symmetric_new(plain1[i], False) for i in range(batch_size)]
        multiplied = ghe.evaluator.multiply_plain_new_batched(cipher1, plain2)
        decoded = [ghe.encoder.decode_simd(ghe.decryptor.decrypt_new(multiplied[i])) for i in range(batch_size)]
        for i in range(batch_size):
            self.tester.assertTrue(ghe.near_equal(ghe.mul(message1[i], message2[i]), decoded[i]))

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
            if ghe.is_ntt:
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

def create_test_class(name: str, ghe: GeneralHeContext):

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

    new_type = type(name, (UnnamedClass,), {})

    return new_type

TestBFVHost  = create_test_class("HostBFVHeOperations",  GeneralHeContext(False, SchemeType.BFV, 32, 20, [60, 40, 40, 60], True, 0x123))
TestCKKSHost = create_test_class("HostCKKSHeOperations", GeneralHeContext(False, SchemeType.CKKS, 32, 0, [60, 40, 40, 60], True, 0x123, 10, 1<<20, 1e-2))
TestBGVHost  = create_test_class("HostBGVHeOperations",  GeneralHeContext(False, SchemeType.BGV, 32, 20, [60, 40, 40, 60], True, 0x123))

class HostTestSuite(unittest.TestSuite):

    def __init__(self):
        super().__init__()
        self.addTest(unittest.makeSuite(TestBFVHost))
        self.addTest(unittest.makeSuite(TestCKKSHost))
        self.addTest(unittest.makeSuite(TestBGVHost))

TestBFVDevice  = create_test_class("DeviceBFVHeOperations",  GeneralHeContext(True, SchemeType.BFV, 32, 20, [60, 40, 40, 60], True, 0x123))
TestCKKSDevice = create_test_class("DeviceCKKSHeOperations", GeneralHeContext(True, SchemeType.CKKS, 32, 0, [60, 40, 40, 60], True, 0x123, 10, 1<<20, 1e-2))
TestBGVDevice  = create_test_class("DeviceBGVHeOperations",  GeneralHeContext(True, SchemeType.BGV, 32, 20, [60, 40, 40, 60], True, 0x123))

class DeviceTestSuite(unittest.TestSuite):
    
    def __init__(self):
        super().__init__()
        self.addTest(unittest.makeSuite(TestBFVDevice))
        self.addTest(unittest.makeSuite(TestCKKSDevice))
        self.addTest(unittest.makeSuite(TestBGVDevice))

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(HostTestSuite())
    suite.addTest(DeviceTestSuite())
    return suite

if __name__ == "__main__":
    pytroy.initialize_kernel(0)
    unittest.TextTestRunner(verbosity=2).run(get_suite())
    pytroy.destroy_memory_pool()
