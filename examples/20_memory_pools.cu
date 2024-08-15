#include "examples.h"
#include <thread>

using namespace std;
using namespace troy;

void single_device_multiple_pools(size_t thread_count) {

    std::cout << "Example for single device with multiple memory pools." << std::endl;
    
    // Setup encryption parameters.
    EncryptionParameters params(SchemeType::BFV);
    params.set_poly_modulus_degree(8192);
    params.set_coeff_modulus(CoeffModulus::create(8192, { 40, 40, 40 }));
    params.set_plain_modulus(PlainModulus::batching(8192, 20));

    // With a single device, we can just use the global memory pool to create the context
    // and utilities.
    HeContextPointer context = HeContext::create(params, true, SecurityLevel::Classical128);
    BatchEncoder encoder(context);

    // One could specify the memory pool here, but we simply use the global pool.
    context->to_device_inplace();
    encoder.to_device_inplace();

    // Create keygen, encryptor and decryptor
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.create_public_key(false);
    Encryptor encryptor(context); encryptor.set_public_key(public_key);
    Decryptor decryptor(context, keygen.secret_key());

    // Create a memory pool for the thread.
    std::vector<MemoryPoolHandle> pools;
    for (size_t i = 0; i < thread_count; i++) {
        pools.push_back(MemoryPool::create()); // default device is 0.
    }

    // Thread body
    auto thread_lambda = [&encryptor, &decryptor, &encoder, &pools](size_t thread_index) {
        
        MemoryPoolHandle pool = pools[thread_index];
        // Check we are not using the global pool.
        custom_assert(pool != MemoryPool::GlobalPool()); 

        vector<uint64_t> message = { 1, 2, 3, 4 };

        Plaintext plain = encoder.encode_new(message, pool);
        // Check the create memory pool is used.
        custom_assert(plain.pool() == pool);

        Ciphertext encrypted = encryptor.encrypt_asymmetric_new(plain, nullptr, pool);
        custom_assert(encrypted.pool() == pool);

        Plaintext decrypted = decryptor.decrypt_new(encrypted, pool);
        custom_assert(decrypted.pool() == pool);

        vector<uint64_t> decrypted_message = encoder.decode_new(decrypted, pool);
        message.resize(decrypted_message.size());
        custom_assert(message == decrypted_message);
        
    };

    // Start threads
    vector<thread> threads;
    for (size_t i = 0; i < thread_count; i++) {
        threads.push_back(thread(thread_lambda, i));
    }

    // Join threads
    for (size_t i = 0; i < thread_count; i++) {
        threads[i].join();
    }

    std::cout << "Example completed." << std::endl;

}

void multiple_device(size_t device_count, size_t thread_count) {

    std::cout << "Example for multiple devices with multiple memory pools." << std::endl;
    std::cout << "Device count = " << device_count << ", thread count = " << thread_count << "." << std::endl;
    
    // Setup encryption parameters.
    EncryptionParameters params(SchemeType::BFV);
    params.set_poly_modulus_degree(8192);
    params.set_coeff_modulus(CoeffModulus::create(8192, { 40, 40, 40 }));
    params.set_plain_modulus(PlainModulus::batching(8192, 20));

    // Since we are using multiple devices, we must create the context
    // for each device. We will create `thread_count` pools first, with
    // each pools `i` using the device `i % device_count`. Then we can
    // use the first `device_count` pools for the context and utilities.

    std::vector<MemoryPoolHandle> pools;
    for (size_t i = 0; i < thread_count; i++) {
        pools.push_back(MemoryPool::create(i % device_count));
    }

    // We use shared_ptrs to store those objects, avoiding any memory leaks.
    std::vector<HeContextPointer> contexts;
    std::vector<std::shared_ptr<BatchEncoder>> encoders;
    for (size_t i = 0; i < std::min(device_count, thread_count); i++) {
        HeContextPointer context = HeContext::create(params, true, SecurityLevel::Classical128);
        std::shared_ptr<BatchEncoder> encoder = std::make_shared<BatchEncoder>(context);
        MemoryPoolHandle pool = pools[i];
        context->to_device_inplace(pool);
        encoder->to_device_inplace(pool);
        contexts.push_back(context);
        encoders.push_back(encoder);
    }

    // Next we create the encryptors and decryptors.
    std::vector<std::shared_ptr<Encryptor>> encryptors;
    std::vector<std::shared_ptr<Decryptor>> decryptors;
    for (size_t i = 0; i < std::min(device_count, thread_count); i++) {
        MemoryPoolHandle pool = pools[i];
        KeyGenerator keygen(contexts[i], pool);
        PublicKey public_key = keygen.create_public_key(false, pool);
        std::shared_ptr<Encryptor> encryptor = std::make_shared<Encryptor>(contexts[i]);
        encryptor->set_public_key(public_key);
        std::shared_ptr<Decryptor> decryptor = std::make_shared<Decryptor>(contexts[i], keygen.secret_key(), pool);
        encryptors.push_back(encryptor);
        decryptors.push_back(decryptor);
    }

    // Thread body
    auto thread_lambda = [&pools](size_t thread_index, const BatchEncoder& encoder, const Encryptor& encryptor, const Decryptor& decryptor) {
        
        MemoryPoolHandle pool = pools[thread_index];
        vector<uint64_t> message = { 1, 2, 3, 4 };

        // The thread's pool might not be the same as the context's pool,
        // but that is ok because they are both on the same device.
        Plaintext plain = encoder.encode_new(message, pool);
        custom_assert(plain.pool() == pool);
        Ciphertext encrypted = encryptor.encrypt_asymmetric_new(plain, nullptr, pool);
        custom_assert(encrypted.pool() == pool);
        Plaintext decrypted = decryptor.decrypt_new(encrypted, pool);
        custom_assert(decrypted.pool() == pool);

        vector<uint64_t> decrypted_message = encoder.decode_new(decrypted, pool);
        message.resize(decrypted_message.size());
        custom_assert(message == decrypted_message);
    };

    // Start threads
    vector<thread> threads;
    for (size_t i = 0; i < thread_count; i++) {
        size_t device_index = i % device_count;
        threads.push_back(thread(
            thread_lambda, i, 
            std::ref(*encoders[device_index]),
            std::ref(*encryptors[device_index]),
            std::ref(*decryptors[device_index])
        ));
    }

    // Join threads
    for (size_t i = 0; i < thread_count; i++) {
        threads[i].join();
    }

    std::cout << "Example completed." << std::endl;
}

void example_memory_pools() {

    print_example_banner("Memory pools");

    size_t device_count = troy::utils::device_count();
    if (device_count == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return; 
    } else if (device_count == 1) {
        single_device_multiple_pools(4);
    } else {
        // We are testing with threads more than the number of devices.
        multiple_device(device_count, device_count * 2);
    }
}