#include "test_multithread.h"

namespace tool {

    MultithreadHeContext::MultithreadHeContext(size_t threads, bool multiple_pools, bool multiple_devices, const GeneralHeContextParameters& args):
        threads(threads), args(args), scheme(args.scheme), device(args.device), multiple_pools(multiple_pools), multiple_devices(multiple_devices)
    {
        contexts_.clear();
        pools_.clear();
        if (!device) {
            // only one pool, which is nullptr
            pools_ = { nullptr };
            // only one context
            GeneralHeContextParameters args_clone = args; args_clone.pool = nullptr;
            auto context = std::make_shared<GeneralHeContext>(args_clone);
            contexts_.push_back(context);
            this->device_count = 0;
        } else {
            if (multiple_devices) {
                int device_count = 0;
                cudaError_t success = cudaGetDeviceCount(&device_count);
                if (success != cudaSuccess) {
                    throw std::runtime_error("cudaGetDeviceCount failed");
                }

                // pools count equal to threads count, each's device index is modulo device count
                for (size_t i = 0; i < threads; i++) {
                    pools_.push_back(MemoryPool::create(i % device_count));
                }

                // contexts count equal to device count
                for (size_t i = 0; i < std::min(static_cast<size_t>(device_count), threads); i++) {
                    GeneralHeContextParameters args_clone = args; args_clone.pool = pools_[i];
                    auto context = std::make_shared<GeneralHeContext>(args_clone);
                    contexts_.push_back(context);
                }

                this->device_count = device_count;
            } else if (multiple_pools) {
                // pools count equal to threads count, each's device index is 0
                for (size_t i = 0; i < threads; i++) {
                    pools_.push_back(MemoryPool::create(0));
                }
                // only one context
                GeneralHeContextParameters args_clone = args; args_clone.pool = pools_[0];
                auto context = std::make_shared<GeneralHeContext>(args_clone);
                contexts_.push_back(context);
                this->device_count = 1;
            } else {
                // one pool which is globalpool
                pools_ = { MemoryPool::GlobalPool() };
                // only one context
                GeneralHeContextParameters args_clone = args; args_clone.pool = pools_[0];
                auto context = std::make_shared<GeneralHeContext>(args_clone);
                contexts_.push_back(context);
                this->device_count = 1;
            }

            if (contexts_.size() > 1) {
                // check all contexts have the same secret key
                SecretKey secret_key = contexts_[0]->key_generator().secret_key().to_host();
                for (size_t i = 1; i < contexts_.size(); i++) {
                    SecretKey secret_key_i = contexts_[i]->key_generator().secret_key().to_host();
                    bool same = secret_key.data().size() == secret_key_i.data().size();
                    for (size_t j = 0; j < secret_key.data().size(); j++) {
                        if (secret_key.data()[j] != secret_key_i.data()[j]) {
                            same = false;
                            break;
                        }
                    }
                    if (!same) {
                        throw std::runtime_error("contexts have different secret keys");
                    }
                }
            }
        }
    }

}