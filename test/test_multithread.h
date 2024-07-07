#include "test_adv.h"

namespace tool {

    class MultithreadHeContext {

        private:
            std::vector<std::shared_ptr<GeneralHeContext>> contexts_;
            std::vector<MemoryPoolHandle> pools_;
            size_t threads;
            GeneralHeContextParameters args;
            SchemeType scheme;
            bool device;
            bool multiple_pools;
            bool multiple_devices;
            size_t device_count;

        public:

            inline MultithreadHeContext() {}

            inline size_t get_context_count() const {
                return contexts_.size();
            }

            inline uint64_t simd_t() const {
                return contexts_[0]->simd_t();
            }

            inline uint128_t ring_t_mask() const {
                return contexts_[0]->ring_t_mask();
            }

            inline const std::vector<std::shared_ptr<GeneralHeContext>>& contexts() const {
                return contexts_;
            }

            inline const std::vector<MemoryPoolHandle>& pools() const {
                return pools_;
            }

            MultithreadHeContext(size_t threads, bool multiple_pools, bool multiple_devices, const GeneralHeContextParameters& args);

            inline size_t get_pool_index(size_t thread_id) const {
                if (device && (multiple_devices || multiple_pools)) {
                    return thread_id;
                } else {
                    return 0;
                }
            }

            inline MemoryPoolHandle get_pool(size_t thread_id) const {
                return pools_[get_pool_index(thread_id)];
            }

            inline MemoryPoolHandle pool_at(size_t index) const {
                return pools_[index];
            }

            inline size_t get_context_index(size_t thread_id) const {
                if (multiple_devices) {
                    return thread_id % contexts_.size();
                } else {
                    return 0;
                }
            }

            inline const GeneralHeContext& get_context(size_t thread_id) const {
                return *contexts_[get_context_index(thread_id)];
            }

            inline const GeneralHeContext& context_at(size_t index) const {
                return *contexts_[index];
            }

            inline size_t get_divided(size_t total, size_t thread_id) const {
                size_t divided = total / threads;
                if (thread_id < total % threads) {
                    return divided + 1;
                } else {
                    return divided;
                }
            }

    };

}