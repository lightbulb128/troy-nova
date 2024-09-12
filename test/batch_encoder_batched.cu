#include <gtest/gtest.h>
#include "test.h"
#include "test_adv.h"

namespace batch_encoder_batched {

    using namespace troy;
    using std::vector;
    using troy::utils::ConstSlice;
    using troy::utils::Slice;
    using std::optional;
    using std::complex;
    using tool::GeneralEncoder;
    using tool::GeneralHeContext;

    std::vector<utils::Array<uint64_t>> gv_to_array(const std::vector<tool::GeneralVector>& v, bool device) {
        std::vector<utils::Array<uint64_t>> res;
        for (const auto& vec : v) {
            vector<uint64_t> r = vec.integers();
            res.push_back(utils::Array<uint64_t>::from_vector(std::move(r)));
            if (device) {
                res.back().to_device_inplace();
            }
        }
        return res;
    }

    void test_unbatch_uint_vector(const GeneralHeContext& context) {
        constexpr size_t batch_size = 16;
        const auto& encoder = context.encoder().batch();

        size_t slot_count = encoder.slot_count();
        for (size_t used: {slot_count, (size_t)(slot_count * 0.3)}) {
            auto message = context.batch_random_simd(batch_size, used);
            auto message_array = gv_to_array(message, context.context()->on_device());
            auto message_batched =  batch_utils::rcollect_const_reference(message_array);
            auto plains = encoder.encode_slice_new_batched(message_batched);
            auto plain_ptrs = batch_utils::collect_const_pointer(plains);
            auto decoded = encoder.decode_slice_new_batched(plain_ptrs);
            auto zeros = utils::Array<uint64_t>(slot_count - used, context.context()->on_device());
            for (size_t i = 0; i < batch_size; i++) {
                ASSERT_TRUE(same_vector(message_array[i].const_reference(), decoded[i].const_slice(0, message_array[i].size())));
                ASSERT_TRUE(same_vector(zeros.const_reference(), decoded[i].const_slice(message_array[i].size(), slot_count)));
            }
        }
    }

    TEST(BatchEncoderBatchTest, HostUnbatchUintVector) {
        test_unbatch_uint_vector(GeneralHeContext(false, SchemeType::BFV, 64, 30, {40, 40, 40}, true, 123));
    }

    TEST(BatchEncoderBatchTest, DeviceUnbatchUintVector) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_unbatch_uint_vector(GeneralHeContext(true, SchemeType::BFV, 64, 30, {40, 40, 40}, true, 123));
        utils::MemoryPool::Destroy();
    }

}