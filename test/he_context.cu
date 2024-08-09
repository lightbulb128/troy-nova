#include <gtest/gtest.h>
#include <vector>
#include "test.h"
#include "../src/he_context.h"

using namespace troy;
using troy::utils::Array;
using std::vector;

namespace he_context {

    Array<Modulus> to_moduli(vector<uint64_t> m) {
        Array<Modulus> moduli(m.size(), false);
        for (size_t i = 0; i < m.size(); i++) {
            moduli[i] = Modulus(m[i]);
        }
        return moduli;
    }

    bool test_qualifiers(
        HeContextPointer context,
        EncryptionParameterErrorType result,
        bool parameters_set,
        bool fft,
        bool ntt,
        bool batching,
        bool fast_plain_lift,
        bool descending_chain,
        SecurityLevel sec_level,
        bool keyswitching,
        bool key_descend
    ) {
        const EncryptionParameterQualifiers& qualifiers = context->first_context_data().value()->qualifiers();
        bool success = true;
        success &= qualifiers.parameter_error == result;
        if (!success) { std::cout << "parameter_error " << (int)(qualifiers.parameter_error) << std::endl; return success; }
        success &= qualifiers.parameters_set() == parameters_set;
        if (!success) { std::cout << "parameters_set" << std::endl; return success; }
        success &= qualifiers.using_fft == fft;
        if (!success) { std::cout << "using_fft" << std::endl; return success; }
        success &= qualifiers.using_ntt == ntt;
        if (!success) { std::cout << "using_ntt" << std::endl; return success; }
        success &= qualifiers.using_batching == batching;
        if (!success) { std::cout << "using_batching" << std::endl; return success; }
        success &= qualifiers.using_fast_plain_lift == fast_plain_lift;
        if (!success) { std::cout << "using_fast_plain_lift" << std::endl; return success; }
        if (!key_descend) {
            success &= qualifiers.using_descending_modulus_chain == descending_chain;
            if (!success) { std::cout << "using_descending_modulus_chain" << std::endl; return success; }
        } else {
            const EncryptionParameterQualifiers& key_qualifiers = context->key_context_data().value()->qualifiers();
            success &= key_qualifiers.using_descending_modulus_chain == descending_chain;
            if (!success) { std::cout << "using_descending_modulus_chain" << std::endl; return success; }
        }
        success &= qualifiers.security_level == sec_level;
        if (!success) { std::cout << "security_level" << std::endl; return success; }
        success &= context->using_keyswitching() == keyswitching;
        if (!success) { std::cout << "using_keyswitching" << std::endl; return success; }
        return success;
    }

    TEST(HeContextTest, BFVConstruct) {
        SchemeType scheme = SchemeType::BFV;
        EncryptionParameters parms(scheme);

        auto context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::InvalidCoeffModulusSize,
            false, false, false, false, false, false, SecurityLevel::Nil, false, false));

        Array<Modulus> moduli;
        parms.set_poly_modulus_degree(4);
        moduli = to_moduli({2, 30});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(2));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::FailedCreatingRNSBase,
            false, true, false, false, false, false, SecurityLevel::Nil, false, false));

        parms.set_poly_modulus_degree(4);
        moduli = to_moduli({17, 41});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(34));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::InvalidPlainModulusCoprimality,
            false, true, true, false, false, false, SecurityLevel::Nil, false, false
        ));
        
        parms.set_poly_modulus_degree(4);
        moduli = to_moduli({17});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(41));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::InvalidPlainModulusTooLarge,
            false, true, true, false, false, false, SecurityLevel::Nil, false, false
        ));

        parms.set_poly_modulus_degree(4);
        moduli = to_moduli({3});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(2));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::InvalidCoeffModulusNoNTT,
            false, true, false, false, false, false, SecurityLevel::Nil, false, false
        ));

        parms.set_poly_modulus_degree(4);
        moduli = to_moduli({17, 41});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(18));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_EQ(context->first_context_data().value()->total_coeff_modulus()[0], 697);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::Success,
            true, true, true, false, false, false, SecurityLevel::Nil, false, false
        ));

        parms.set_poly_modulus_degree(4);
        moduli = to_moduli({17, 41});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(16));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_EQ(context->first_context_data().value()->total_coeff_modulus()[0], 17);
        ASSERT_EQ(context->key_context_data().value()->total_coeff_modulus()[0], 697);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::Success,
            true, true, true, false, true, false, SecurityLevel::Nil, true, true
        ));

        parms.set_poly_modulus_degree(4);
        moduli = to_moduli({17, 41});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(49));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_EQ(context->first_context_data().value()->total_coeff_modulus()[0], 697);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::Success,
            true, true, true, false, false, false, SecurityLevel::Nil, false, false
        ));
        
        parms.set_poly_modulus_degree(4);
        moduli = to_moduli({17, 41});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(73));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_EQ(context->first_context_data().value()->total_coeff_modulus()[0], 697);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::Success,
            true, true, true, true, false, false, SecurityLevel::Nil, false, false
        ));
        
        parms.set_poly_modulus_degree(4);
        moduli = to_moduli({137, 193});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(73));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_EQ(context->first_context_data().value()->total_coeff_modulus()[0], 137);
        ASSERT_EQ(context->key_context_data().value()->total_coeff_modulus()[0], 26441);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::Success,
            true, true, true, true, true, false, SecurityLevel::Nil, true, true
        ));
        
        parms.set_poly_modulus_degree(4);
        moduli = to_moduli({137, 193});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(73));
        context = HeContext::create(parms, false, SecurityLevel::Classical128);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::InvalidParametersInsecure,
            false, true, false, false, false, false, SecurityLevel::Nil, false, false
        ));

        parms.set_poly_modulus_degree(2048);
        moduli = CoeffModulus::bfv_default(4096, SecurityLevel::Classical128);
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(73));
        context = HeContext::create(parms, false, SecurityLevel::Classical128);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::InvalidParametersInsecure,
            false, true, false, false, false, false, SecurityLevel::Nil, false, false
        ));

        parms.set_poly_modulus_degree(4096);
        moduli = to_moduli({0xffffee001, 0xffffc4001});
        parms.set_coeff_modulus(moduli.const_reference());
        parms.set_plain_modulus(Modulus(73));
        context = HeContext::create(parms, false, SecurityLevel::Classical128);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::Success,
            true, true, true, false, true, true, SecurityLevel::Classical128, true, false
        ));

        parms.set_poly_modulus_degree(2048);
        parms.set_coeff_modulus(to_moduli({0x1ffffe0001, 0xffffee001, 0xffffc4001}).const_reference());
        parms.set_plain_modulus(Modulus(73));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::Success,
            true, true, true, false, true, true, SecurityLevel::Nil, true, true
        ));
        
        parms.set_poly_modulus_degree(2048);
        parms.set_coeff_modulus(CoeffModulus::create(2048, {40}).const_reference());
        parms.set_plain_modulus(Modulus(65537));
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_TRUE(test_qualifiers(context, EncryptionParameterErrorType::Success,
            true, true, true, true, true, true, SecurityLevel::Nil, false, false
        ));
    }

    TEST(HeContextTest, ModulusChainExpansion) {

        // BFV

        SchemeType scheme = SchemeType::BFV;
        EncryptionParameters parms(scheme);

        parms.set_poly_modulus_degree(4);
        parms.set_coeff_modulus(to_moduli({41, 137, 193, 65537}).const_reference());
        parms.set_plain_modulus(Modulus(73));
        auto context = HeContext::create(parms, true, SecurityLevel::Nil);
        auto context_data = context->key_context_data().value();
        ASSERT_EQ(context_data->chain_index(), 2);
        ASSERT_EQ(context_data->total_coeff_modulus()[0], 71047416497);
        auto prev_context_data = context_data;
        context_data = prev_context_data->next_context_data().value();
        ASSERT_EQ(context_data->chain_index(), 1);
        ASSERT_EQ(context_data->total_coeff_modulus()[0], 1084081);
        ASSERT_EQ(
            context_data->prev_context_data().value().lock()->parms_id(),
            prev_context_data->parms_id()
        );
        prev_context_data = context_data;
        context_data = prev_context_data->next_context_data().value();
        ASSERT_EQ(context_data->chain_index(), 0);
        ASSERT_EQ(context_data->total_coeff_modulus()[0], 5617);
        ASSERT_EQ(
            context_data->prev_context_data().value().lock()->parms_id(),
            prev_context_data->parms_id()
        );
        ASSERT_FALSE(context_data->next_context_data().has_value());
        ASSERT_EQ(context_data->parms_id(), context->last_parms_id());

        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_EQ(context->key_context_data().value()->chain_index(), 1);
        ASSERT_EQ(context->first_context_data().value()->chain_index(), 0);
        ASSERT_EQ(context->key_context_data().value()->total_coeff_modulus()[0], 71047416497);
        ASSERT_EQ(context->first_context_data().value()->total_coeff_modulus()[0], 1084081);

        // BGV

        scheme = SchemeType::BGV;
        parms = EncryptionParameters(scheme);

        parms.set_poly_modulus_degree(4);
        parms.set_coeff_modulus(to_moduli({41, 137, 193, 65537}).const_reference());
        parms.set_plain_modulus(Modulus(73));
        context = HeContext::create(parms, true, SecurityLevel::Nil);
        context_data = context->key_context_data().value();
        ASSERT_EQ(context_data->chain_index(), 2);
        ASSERT_EQ(context_data->total_coeff_modulus()[0], 71047416497);
        prev_context_data = context_data;
        context_data = prev_context_data->next_context_data().value();
        ASSERT_EQ(context_data->chain_index(), 1);
        ASSERT_EQ(context_data->total_coeff_modulus()[0], 1084081);
        ASSERT_EQ(
            context_data->prev_context_data().value().lock()->parms_id(),
            prev_context_data->parms_id()
        );
        prev_context_data = context_data;
        context_data = prev_context_data->next_context_data().value();
        ASSERT_EQ(context_data->chain_index(), 0);
        ASSERT_EQ(context_data->total_coeff_modulus()[0], 5617);
        ASSERT_EQ(
            context_data->prev_context_data().value().lock()->parms_id(),
            prev_context_data->parms_id()
        );
        ASSERT_FALSE(context_data->next_context_data().has_value());
        ASSERT_EQ(context_data->parms_id(), context->last_parms_id());

        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_EQ(context->key_context_data().value()->chain_index(), 1);
        ASSERT_EQ(context->first_context_data().value()->chain_index(), 0);
        ASSERT_EQ(context->key_context_data().value()->total_coeff_modulus()[0], 71047416497);
        ASSERT_EQ(context->first_context_data().value()->total_coeff_modulus()[0], 1084081);

        // CKKS

        scheme = SchemeType::CKKS;
        parms = EncryptionParameters(scheme);

        parms.set_poly_modulus_degree(4);
        parms.set_coeff_modulus(to_moduli({41, 137, 193, 65537}).const_reference());
        context = HeContext::create(parms, true, SecurityLevel::Nil);
        context_data = context->key_context_data().value();
        ASSERT_EQ(context_data->chain_index(), 3);
        ASSERT_EQ(context_data->total_coeff_modulus()[0], 71047416497);
        prev_context_data = context_data;
        context_data = prev_context_data->next_context_data().value();
        ASSERT_EQ(context_data->chain_index(), 2);
        ASSERT_EQ(context_data->total_coeff_modulus()[0], 1084081);
        ASSERT_EQ(
            context_data->prev_context_data().value().lock()->parms_id(),
            prev_context_data->parms_id()
        );
        prev_context_data = context_data;
        context_data = prev_context_data->next_context_data().value();
        ASSERT_EQ(context_data->chain_index(), 1);
        ASSERT_EQ(context_data->total_coeff_modulus()[0], 5617);
        ASSERT_EQ(
            context_data->prev_context_data().value().lock()->parms_id(),
            prev_context_data->parms_id()
        );
        prev_context_data = context_data;
        context_data = prev_context_data->next_context_data().value();
        ASSERT_EQ(context_data->chain_index(), 0);
        ASSERT_EQ(context_data->total_coeff_modulus()[0], 41);
        ASSERT_EQ(
            context_data->prev_context_data().value().lock()->parms_id(),
            prev_context_data->parms_id()
        );
        ASSERT_FALSE(context_data->next_context_data().has_value());
        ASSERT_EQ(context_data->parms_id(), context->last_parms_id());

        context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_EQ(context->key_context_data().value()->chain_index(), 1);
        ASSERT_EQ(context->first_context_data().value()->chain_index(), 0);
        ASSERT_EQ(context->key_context_data().value()->total_coeff_modulus()[0], 71047416497);
        ASSERT_EQ(context->first_context_data().value()->total_coeff_modulus()[0], 1084081);
    }

    TEST(HeContextTest, HeContextToDevice) {
        SKIP_WHEN_NO_CUDA_DEVICE;

        // BFV

        SchemeType scheme = SchemeType::BFV;
        EncryptionParameters parms(scheme);

        parms.set_poly_modulus_degree(4);
        parms.set_coeff_modulus(to_moduli({41, 137, 193, 65537}).const_reference());
        parms.set_plain_modulus(Modulus(73));
        auto context = HeContext::create(parms, true, SecurityLevel::Nil);
        context->to_device_inplace();
        
        // BGV

        scheme = SchemeType::BGV;
        parms = EncryptionParameters(scheme);

        parms.set_poly_modulus_degree(4);
        parms.set_coeff_modulus(to_moduli({41, 137, 193, 65537}).const_reference());
        parms.set_plain_modulus(Modulus(73));
        context->to_device_inplace();

        // CKKS

        scheme = SchemeType::CKKS;
        parms = EncryptionParameters(scheme);

        parms.set_poly_modulus_degree(4);
        parms.set_coeff_modulus(to_moduli({41, 137, 193, 65537}).const_reference());
        context = HeContext::create(parms, true, SecurityLevel::Nil);
        context->to_device_inplace();

        utils::MemoryPool::Destroy();
    }

}