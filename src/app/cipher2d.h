#pragma once
#include "../encryptor.h"
#include "../decryptor.h"
#include "../evaluator.h"

namespace troy { namespace linear {

    class Cipher2d;
    
    class Plain2d {

    private:
        std::vector<std::vector<Plaintext>> inner;

        Cipher2d encrypt_internal(const Encryptor& encryptor, bool symmetric, MemoryPoolHandle pool) const;

    public:

        inline size_t size() const {return inner.size();}

        inline const std::vector<std::vector<Plaintext>>& data() const {
            return inner;
        }
        inline std::vector<std::vector<Plaintext>>& data() {
            return inner;
        }

        inline const std::vector<Plaintext>& operator[](size_t i) const {
            return inner[i];
        }
        inline std::vector<Plaintext>& operator[](size_t i) {
            return inner[i];
        }

        inline Plain2d clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plain2d result;
            for (const std::vector<Plaintext>& row : inner) {
                std::vector<Plaintext>& new_row = result.new_row();
                for (const Plaintext& plain : row) {
                    new_row.push_back(plain.clone(pool));
                }
            }
            return result;
        }

        inline size_t rows() const {
            return inner.size();
        }
        inline size_t columns() const {
            if (inner.size() == 0) return 0;
            size_t size = inner[0].size();
            for (size_t i = 1; i < inner.size(); i++) {
                if (inner[i].size() != size) {
                    throw std::runtime_error("[Plain2d::columns] Not all rows have same size");
                }
            }
            return size;
        }

        inline Plain2d() {}

        inline void resize(size_t rows, size_t columns) {
            inner.resize(rows);
            for (size_t i = 0; i < rows; i++) {
                inner[i].resize(columns);
            }
        }

        inline std::vector<Plaintext>& new_row() {
            inner.push_back(std::vector<Plaintext>());
            return inner.back();
        }

        Cipher2d encrypt_asymmetric(const Encryptor& encryptor, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        Cipher2d encrypt_symmetric(const Encryptor& encryptor, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

    };

    class Cipher2d {

    private:
        std::vector<std::vector<Ciphertext>> inner;


        Cipher2d translate(const Evaluator& evaluator, const Cipher2d& other, bool subtract, MemoryPoolHandle pool) const;
        void translate_inplace(const Evaluator& evaluator, const Cipher2d& other, bool subtract, MemoryPoolHandle pool);
        Cipher2d translate_plain(const Evaluator& evaluator, const Plain2d& other, bool subtract, MemoryPoolHandle pool) const;
        void translate_plain_inplace(const Evaluator& evaluator, const Plain2d& other, bool subtract, MemoryPoolHandle pool);

    public:

        inline size_t size() const {return inner.size();}

        inline const std::vector<std::vector<Ciphertext>>& data() const {
            return inner;
        }
        inline std::vector<std::vector<Ciphertext>>& data() {
            return inner;
        }

        inline const std::vector<Ciphertext>& operator[](size_t i) const {
            return inner[i];
        }
        inline std::vector<Ciphertext>& operator[](size_t i) {
            return inner[i];
        }

        inline Cipher2d clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Cipher2d result;
            for (const std::vector<Ciphertext>& row : inner) {
                std::vector<Ciphertext>& new_row = result.new_row();
                for (const Ciphertext& plain : row) {
                    new_row.push_back(plain.clone(pool));
                }
            }
            return result;
        }

        inline size_t rows() const {
            return inner.size();
        }
        inline size_t columns() const {
            if (inner.size() == 0) return 0;
            size_t size = inner[0].size();
            for (size_t i = 1; i < inner.size(); i++) {
                if (inner[i].size() != size) {
                    throw std::runtime_error("[Cipher2d::columns] Not all rows have same size");
                }
            }
            return size;
        }

        inline Cipher2d() {}

        inline void resize(size_t rows, size_t columns) {
            inner.resize(rows);
            for (size_t i = 0; i < rows; i++) {
                inner[i].resize(columns);
            }
        }

        inline std::vector<Ciphertext>& new_row() {
            inner.push_back(std::vector<Ciphertext>());
            return inner.back();
        }

        inline void expand_seed(HeContextPointer context) {
            for (size_t i = 0; i < inner.size(); i++) {
                for (size_t j = 0; j < inner[i].size(); j++) {
                    if (inner[i][j].contains_seed()) inner[i][j].expand_seed(context);
                }
            }
        }

        size_t save(std::ostream& stream, HeContextPointer context, CompressionMode mode = CompressionMode::Nil) const;
        void load(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool());
        inline static Cipher2d load_new(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            Cipher2d result;
            result.load(stream, context, pool);
            return result;
        }
        size_t serialized_size_upperbound(HeContextPointer context, CompressionMode mode = CompressionMode::Nil) const;

        inline void mod_switch_to_next_inplace(const Evaluator& evaluator, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            for (std::vector<Ciphertext>& row : this->data()) {
                for (Ciphertext& cipher : row) {
                    evaluator.mod_switch_to_next_inplace(cipher, pool);
                }
            }
        }

        inline Cipher2d mod_switch_to_next(const Evaluator& evaluator, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Cipher2d result;
            Ciphertext buffer;
            for (const std::vector<Ciphertext>& original_row : this->data()) {
                std::vector<Ciphertext>& row = result.new_row();
                for (const Ciphertext& cipher : original_row) {
                    evaluator.mod_switch_to_next(cipher, buffer, pool);
                    row.push_back(std::move(buffer));
                }
            }
            return result;
        }

        inline void relinearize_inplace(const Evaluator& evaluator, const RelinKeys& relin_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            for (std::vector<Ciphertext>& row : this->data()) {
                for (Ciphertext& cipher : row) {
                    evaluator.relinearize_inplace(cipher, relin_keys, pool);
                }
            }
        }

        inline Cipher2d relinearize(const Evaluator& evaluator, const RelinKeys& relin_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Cipher2d result;
            Ciphertext buffer;
            for (const std::vector<Ciphertext>& original_row : this->data()) {
                std::vector<Ciphertext>& row = result.new_row();
                for (const Ciphertext& cipher : original_row) {
                    evaluator.relinearize(cipher, relin_keys, buffer, pool);
                    row.push_back(std::move(buffer));
                }
            }
            return result;
        }

        inline Cipher2d add(const Evaluator& evaluator, const Cipher2d& other, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            return translate(evaluator, other, false, pool);
        }
        inline void add_inplace(const Evaluator& evaluator, const Cipher2d& other, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            translate_inplace(evaluator, other, false, pool);
        }
        inline Cipher2d add_plain(const Evaluator& evaluator, const Plain2d& other, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            return translate_plain(evaluator, other, false, pool);
        }
        inline void add_plain_inplace(const Evaluator& evaluator, const Plain2d& other, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            translate_plain_inplace(evaluator, other, false, pool);
        }
        inline Cipher2d sub(const Evaluator& evaluator, const Cipher2d& other, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            return translate(evaluator, other, true, pool);
        }
        inline void sub_inplace(const Evaluator& evaluator, const Cipher2d& other, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            translate_inplace(evaluator, other, true, pool);
        }
        inline Cipher2d sub_plain(const Evaluator& evaluator, const Plain2d& other, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            return translate_plain(evaluator, other, true, pool);
        }
        inline void sub_plain_inplace(const Evaluator& evaluator, const Plain2d& other, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            translate_plain_inplace(evaluator, other, true, pool);
        }

        inline Plain2d decrypt(const Decryptor& decryptor, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plain2d result;
            for (const std::vector<Ciphertext>& row : inner) {
                std::vector<Plaintext>& new_row = result.new_row();
                for (const Ciphertext& cipher : row) {
                    new_row.push_back(decryptor.decrypt_new(cipher, pool));
                }
            }
            return result;
        }

    };

}}