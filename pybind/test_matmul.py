from test import GeneralVectorDataType, GeneralEncoder, GeneralVector, GeneralHeContext
import pytroy
from pytroy import SchemeType
import numpy as np
import unittest
import argparse

class HeMatmulTest:

    def __init__(self, ghe: GeneralHeContext, tester: unittest.TestCase):
        self.ghe = ghe
        self.tester = tester

    def test_matmul(self, m: int, r: int, n: int, pack_lwe: bool, mod_switch_to_next: bool):
        ghe = self.ghe
        if ghe.scheme != SchemeType.BFV: return
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
        
        x_encoded = helper.encode_inputs(encoder.encoder, x.data)
        w_encoded = helper.encode_weights(encoder.encoder, w.data)
        s_encoded = helper.encode_outputs(encoder.encoder, s.data)

        x_encrypted = x_encoded.encrypt_symmetric(encryptor)
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


def create_test_class(ghe: GeneralHeContext):

    class UnnamedClass(unittest.TestCase):

        def setUp(self) -> None:
            self.ghe = ghe
            self.tester = HeMatmulTest(self.ghe, self)
            return super().setUp()

    if ghe.scheme == SchemeType.BFV:
        UnnamedClass.test_matmul_small_no_pack = lambda self: self.tester.test_matmul(4, 5, 6, pack_lwe=False, mod_switch_to_next=False)
        UnnamedClass.test_matmul_medium_no_pack = lambda self: self.tester.test_matmul(64, 128, 256, pack_lwe=False, mod_switch_to_next=False)
        UnnamedClass.test_matmul_small_pack = lambda self: self.tester.test_matmul(4, 5, 6, pack_lwe=True, mod_switch_to_next=False)
        UnnamedClass.test_matmul_medium_pack = lambda self: self.tester.test_matmul(64, 128, 256, pack_lwe=True, mod_switch_to_next=False)
        if ghe.device:
            UnnamedClass.test_matmul_large_no_pack = lambda self: self.tester.test_matmul(400, 500, 600, pack_lwe=False, mod_switch_to_next=False)
            UnnamedClass.test_matmul_large_pack = lambda self: self.tester.test_matmul(400, 500, 600, pack_lwe=True, mod_switch_to_next=False)

    return UnnamedClass

class HostTestSuite(unittest.TestSuite):

    def __init__(self):
        super().__init__()
        test_case = create_test_class(GeneralHeContext(False, SchemeType.BFV, 8192, 20, [60, 40, 40, 60], True, 0x123))
        self.addTest(unittest.makeSuite(test_case))

class DeviceTestSuite(unittest.TestSuite):

    def __init__(self):
        super().__init__()
        test_case = create_test_class(GeneralHeContext(True, SchemeType.BFV, 8192, 20, [60, 40, 40, 60], True, 0x123))
        self.addTest(unittest.makeSuite(test_case))


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