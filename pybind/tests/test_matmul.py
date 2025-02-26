from utils import GeneralVectorDataType, GeneralEncoder, GeneralVector, GeneralHeContext
import pytroy
from pytroy import SchemeType
import numpy as np
import unittest
import argparse

class HeUint64MatmulTest:

    def __init__(self, ghe: GeneralHeContext, tester: unittest.TestCase):
        self.ghe = ghe
        self.tester = tester

    def test_matmul(self, m: int, r: int, n: int, pack_lwe: bool, mod_switch_to_next: bool):
        ghe = self.ghe
        if ghe.scheme != SchemeType.BFV and ghe.scheme != SchemeType.BGV: return
        t = ghe.t
        
        def mulmod(a, b):
            return (int(a) * int(b)) % t
        def addmod(a, b):
            return (int(a) + int(b)) % t
        
        x = ghe.random_polynomial(m * r)
        w = ghe.random_polynomial(r * n)
        s = ghe.random_polynomial(m * n)
        helper = pytroy.MatmulHelper(m, r, n, ghe.parms.poly_modulus_degree(), pytroy.MatmulObjective.EncryptLeft, pack_lwe)
        
        he = ghe.context
        encoder = ghe.encoder
        encryptor = ghe.encryptor
        decryptor = ghe.decryptor
        evaluator = ghe.evaluator
        automorphism_keys = None
        if pack_lwe:
            automorphism_keys = ghe.key_generator.create_automorphism_keys(False)
        
        x_encrypted = helper.encrypt_inputs(encryptor, encoder.encoder, x.data)
        w_encoded = helper.encode_weights(encoder.encoder, w.data)
        s_encoded = helper.encode_outputs(encoder.encoder, s.data)

        x_serialized = x_encrypted.save(he)
        x_encrypted = pytroy.Cipher2d.load_new(x_serialized, he)

        y_encrypted = helper.matmul(evaluator, x_encrypted, w_encoded)
        if mod_switch_to_next:
            y_encrypted.mod_switch_to_next_inplace(evaluator)
        if pack_lwe:
            y_encrypted = helper.pack_outputs(evaluator, automorphism_keys, y_encrypted)
        y_encrypted.add_plain_inplace(evaluator, s_encoded)

        y_serialized = helper.serialize_outputs(evaluator, y_encrypted)
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized)

        y_decrypted = helper.decrypt_outputs(encoder.encoder, decryptor, y_encrypted)

        x_np_mat = np.reshape(x.data, (m, r))
        w_np_mat = np.reshape(w.data, (r, n))
        s_np_mat = np.reshape(s.data, (m, n))
        y_np_mat = np.matmul(x_np_mat, w_np_mat) + s_np_mat
        y_truth = np.reshape(y_np_mat, (m * n,)) % t

        decrypted = GeneralVector(GeneralVectorDataType.Integers, y_decrypted)
        truth = GeneralVector(GeneralVectorDataType.Integers, y_truth)
        self.tester.assertTrue(ghe.near_equal(decrypted, truth))

def create_test_uint64s_class(name, ghe: GeneralHeContext):

    class UnnamedClass(unittest.TestCase):

        def setUp(self) -> None:
            self.ghe = ghe
            self.tester = HeUint64MatmulTest(self.ghe, self)
            return super().setUp()

    if ghe.scheme == SchemeType.BFV or ghe.scheme == SchemeType.BGV:
        UnnamedClass.test_matmul_small_no_pack = lambda self: self.tester.test_matmul(4, 5, 6, pack_lwe=False, mod_switch_to_next=False)
        UnnamedClass.test_matmul_medium_no_pack = lambda self: self.tester.test_matmul(64, 128, 256, pack_lwe=False, mod_switch_to_next=False)
        UnnamedClass.test_matmul_small_pack = lambda self: self.tester.test_matmul(4, 5, 6, pack_lwe=True, mod_switch_to_next=False)
        UnnamedClass.test_matmul_medium_pack = lambda self: self.tester.test_matmul(64, 128, 256, pack_lwe=True, mod_switch_to_next=False)

    return type(name, (UnnamedClass,), {})

class HeDoubleMatmulTest:

    def __init__(self, ghe: GeneralHeContext, tester: unittest.TestCase):
        self.ghe = ghe
        self.tester = tester

    def test_matmul(self, m: int, r: int, n: int, pack_lwe: bool, mod_switch_to_next: bool):
        ghe = self.ghe
        if ghe.scheme != SchemeType.CKKS: return
        
        x = ghe.random_polynomial(m * r)
        w = ghe.random_polynomial(r * n)
        s = ghe.random_polynomial(m * n)
        helper = pytroy.MatmulHelper(m, r, n, ghe.parms.poly_modulus_degree(), pytroy.MatmulObjective.EncryptLeft, pack_lwe)
        
        he = ghe.context
        encoder = ghe.encoder
        encryptor = ghe.encryptor
        decryptor = ghe.decryptor
        evaluator = ghe.evaluator
        automorphism_keys = None
        if pack_lwe:
            automorphism_keys = ghe.key_generator.create_automorphism_keys(False)
        
        x_encrypted = helper.encrypt_inputs_doubles(encryptor, encoder.encoder, x.data, None, ghe.scale)
        w_encoded = helper.encode_weights_doubles(encoder.encoder, w.data, None, ghe.scale)
        s_encoded = helper.encode_outputs_doubles(encoder.encoder, s.data, None, ghe.scale * ghe.scale)

        x_serialized = x_encrypted.save(he)
        x_encrypted = pytroy.Cipher2d.load_new(x_serialized, he)

        y_encrypted = helper.matmul(evaluator, x_encrypted, w_encoded)
        if mod_switch_to_next:
            y_encrypted.mod_switch_to_next_inplace(evaluator)
        if pack_lwe:
            y_encrypted = helper.pack_outputs(evaluator, automorphism_keys, y_encrypted)
        y_encrypted.add_plain_inplace(evaluator, s_encoded)

        y_serialized = helper.serialize_outputs(evaluator, y_encrypted)
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized)

        y_decrypted = helper.decrypt_outputs_doubles(encoder.encoder, decryptor, y_encrypted)

        x_np_mat = np.reshape(x.data, (m, r))
        w_np_mat = np.reshape(w.data, (r, n))
        s_np_mat = np.reshape(s.data, (m, n))
        y_np_mat = np.matmul(x_np_mat, w_np_mat) + s_np_mat
        y_truth = np.reshape(y_np_mat, (m * n,))

        decrypted = GeneralVector(GeneralVectorDataType.Doubles, y_decrypted)
        truth = GeneralVector(GeneralVectorDataType.Doubles, y_truth)
        self.tester.assertTrue(ghe.near_equal(decrypted, truth))

def create_test_doubles_class(name, ghe: GeneralHeContext):

    class UnnamedClass(unittest.TestCase):

        def setUp(self) -> None:
            self.ghe = ghe
            self.tester = HeDoubleMatmulTest(self.ghe, self)
            return super().setUp()

    UnnamedClass.test_matmul_small_no_pack = lambda self: self.tester.test_matmul(4, 5, 6, pack_lwe=False, mod_switch_to_next=False)
    UnnamedClass.test_matmul_medium_no_pack = lambda self: self.tester.test_matmul(64, 128, 256, pack_lwe=False, mod_switch_to_next=False)
    UnnamedClass.test_matmul_small_pack = lambda self: self.tester.test_matmul(4, 5, 6, pack_lwe=True, mod_switch_to_next=False)
    UnnamedClass.test_matmul_medium_pack = lambda self: self.tester.test_matmul(64, 128, 256, pack_lwe=True, mod_switch_to_next=False)

    return type(name, (UnnamedClass,), {})

class HeRing2kMatmulTest:

    def __init__(self, device: bool, t_bits: int, poly_degree: int, q_bits: "list[int]", tester: unittest.TestCase):
        self.tester = tester
        params = pytroy.EncryptionParameters(pytroy.SchemeType.BFV)
        params.set_poly_modulus_degree(poly_degree)
        params.set_plain_modulus(1 << 20) # this is not important
        params.set_coeff_modulus(pytroy.CoeffModulus.create(poly_degree, q_bits))
        t_mask = (1 << t_bits) - 1

        context = pytroy.HeContext(params, True, pytroy.SecurityLevel.Nil, 0x123)
        if t_bits > 32:
            encoder = pytroy.PolynomialEncoderRing2k64(context, t_bits)
        else:
            encoder = pytroy.PolynomialEncoderRing2k32(context, t_bits)
        
        if device:
            context.to_device_inplace()
            encoder.to_device_inplace()
        
        keygen = pytroy.KeyGenerator(context)
        encryptor = pytroy.Encryptor(context)
        encryptor.set_secret_key(keygen.secret_key())
        decryptor = pytroy.Decryptor(context, keygen.secret_key())
        evaluator = pytroy.Evaluator(context)
        automorphism_keys = keygen.create_automorphism_keys(False)

        self.poly_degree = poly_degree
        self.context = context
        self.encoder = encoder
        self.encryptor = encryptor
        self.decryptor = decryptor
        self.evaluator = evaluator
        self.automorphism_keys = automorphism_keys
        self.t_bits = t_bits
        self.t_mask = t_mask

    def random_polynomial(self, n: int):
        if self.t_bits > 32:
            return np.random.randint(0, np.iinfo(np.uint64).max, n, dtype=np.uint64) & self.t_mask
        else:
            return np.random.randint(0, np.iinfo(np.uint32).max, n, dtype=np.uint32) & self.t_mask

    def test_matmul(self, m: int, r: int, n: int, pack_lwe: bool, mod_switch_to_next: bool):
        x = self.random_polynomial(m * r)
        w = self.random_polynomial(r * n)
        s = self.random_polynomial(m * n)
        helper = pytroy.MatmulHelper(m, r, n, self.poly_degree, pytroy.MatmulObjective.EncryptLeft, pack_lwe)
        
        he = self.context
        encoder = self.encoder
        encryptor = self.encryptor
        decryptor = self.decryptor
        evaluator = self.evaluator
        automorphism_keys = self.automorphism_keys
        
        if self.t_bits > 32:
            x_encrypted = helper.encrypt_inputs_ring2k64(encryptor, encoder, x.data, None)
            w_encoded = helper.encode_weights_ring2k64(encoder, w.data, None)
            s_encoded = helper.encode_outputs_ring2k64(encoder, s.data, None)
        else:
            x_encrypted = helper.encrypt_inputs_ring2k32(encryptor, encoder, x.data, None)
            w_encoded = helper.encode_weights_ring2k32(encoder, w.data, None)
            s_encoded = helper.encode_outputs_ring2k32(encoder, s.data, None)

        x_serialized = x_encrypted.save(he)
        x_encrypted = pytroy.Cipher2d.load_new(x_serialized, he)

        y_encrypted = helper.matmul(evaluator, x_encrypted, w_encoded)
        if mod_switch_to_next:
            y_encrypted.mod_switch_to_next_inplace(evaluator)
        if pack_lwe:
            y_encrypted = helper.pack_outputs(evaluator, automorphism_keys, y_encrypted)
        y_encrypted.add_plain_inplace(evaluator, s_encoded)

        y_serialized = helper.serialize_outputs(evaluator, y_encrypted)
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized)

        if self.t_bits > 32:
            y_decrypted = helper.decrypt_outputs_ring2k64(encoder, decryptor, y_encrypted)
        else:
            y_decrypted = helper.decrypt_outputs_ring2k32(encoder, decryptor, y_encrypted)

        x_np_mat = np.reshape(x.data, (m, r))
        w_np_mat = np.reshape(w.data, (r, n))
        s_np_mat = np.reshape(s.data, (m, n))
        y_np_mat = np.matmul(x_np_mat, w_np_mat) + s_np_mat
        y_truth = np.reshape(y_np_mat, (m * n,)) & self.t_mask

        self.tester.assertTrue(np.allclose(y_decrypted, y_truth))

def create_test_ring2k_class(name, device: bool, t_bits: int, poly_degree: int, q_bits: "list[int]"):

    class UnnamedClass(unittest.TestCase):

        def setUp(self) -> None:
            self.tester = HeRing2kMatmulTest(device, t_bits, poly_degree, q_bits, self)
            return super().setUp()

    UnnamedClass.test_matmul_small_no_pack = lambda self: self.tester.test_matmul(4, 5, 6, pack_lwe=False, mod_switch_to_next=False)
    UnnamedClass.test_matmul_medium_no_pack = lambda self: self.tester.test_matmul(40, 50, 60, pack_lwe=False, mod_switch_to_next=False)
    UnnamedClass.test_matmul_small_pack = lambda self: self.tester.test_matmul(4, 5, 6, pack_lwe=True, mod_switch_to_next=False)
    UnnamedClass.test_matmul_medium_pack = lambda self: self.tester.test_matmul(40, 50, 60, pack_lwe=True, mod_switch_to_next=False)
    
    return type(name, (UnnamedClass,), {})

class CompleteTestSuite(unittest.TestSuite):

    def __init__(self, device):
        super().__init__()
        devstr = "Device" if device else "Host"
        
        test_case = create_test_uint64s_class(devstr + "BFVMatmul", GeneralHeContext(device, SchemeType.BFV, 8192, 20, [60, 40, 40, 60], True, 0x123))
        self.addTest(unittest.makeSuite(test_case))
        test_case = create_test_uint64s_class(devstr + "BGVMatmul", GeneralHeContext(device, SchemeType.BGV, 8192, 20, [60, 40, 40, 60], True, 0x123))
        self.addTest(unittest.makeSuite(test_case))
        
        test_case = create_test_doubles_class(devstr + "CKKSMatmul", GeneralHeContext(device, SchemeType.CKKS, 8192, 20, [60, 40, 40, 60], True, 0x123, 5, 1<<20, 1e-2))
        self.addTest(unittest.makeSuite(test_case))
        
        test_case = create_test_ring2k_class(devstr + "Ring32Matmul", device, 32, 8192, [60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))
        test_case = create_test_ring2k_class(devstr + "Ring20Matmul", device, 20, 8192, [60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))
        test_case = create_test_ring2k_class(devstr + "Ring17Matmul", device, 17, 8192, [60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))

        test_case = create_test_ring2k_class(devstr + "Ring64Matmul", device, 64, 8192, [60, 60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))
        test_case = create_test_ring2k_class(devstr + "Ring50Matmul", device, 50, 8192, [60, 60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))
        test_case = create_test_ring2k_class(devstr + "Ring33Matmul", device, 33, 8192, [60, 60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(CompleteTestSuite(False))
    suite.addTest(CompleteTestSuite(True))
    return suite

if __name__ == "__main__":
    pytroy.initialize_kernel(0)
    unittest.TextTestRunner(verbosity=2).run(get_suite())
    pytroy.destroy_memory_pool()