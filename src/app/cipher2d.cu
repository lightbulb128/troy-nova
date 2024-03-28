#include "cipher2d.cuh"

namespace troy { namespace linear {

    Cipher2d Plain2d::encrypt_internal(const Encryptor& encryptor, bool symmetric) const {
        Cipher2d cipher;
        Ciphertext buffer;
        for (size_t i = 0; i < this->rows(); ++i) {
            std::vector<Ciphertext>& row = cipher.new_row();
            for (size_t j = 0; j < this->data()[i].size(); ++j) {
                if (!symmetric) {
                    encryptor.encrypt_asymmetric((*this)[i][j], buffer);
                } else {
                    encryptor.encrypt_symmetric((*this)[i][j], true, buffer);
                }
                row.push_back(std::move(buffer));
            }
        }
        return cipher;
    }

    Cipher2d Plain2d::encrypt_asymmetric(const Encryptor& encryptor) const {
        return encrypt_internal(encryptor, false);
    }
    Cipher2d Plain2d::encrypt_symmetric(const Encryptor& encryptor) const {
        return encrypt_internal(encryptor, true);
    }

    void Cipher2d::save(std::ostream& stream, HeContextPointer context) const {
        troy::serialize::save_object(stream, (*this).rows());
        for (size_t i = 0; i < this->rows(); i++) {
            size_t row_size = (*this)[i].size();
            troy::serialize::save_object(stream, row_size);
            for (size_t j = 0; j < row_size; j++) {
                (*this)[i][j].save(stream, context);
            }
        }
    }

    void Cipher2d::load(std::istream& stream, HeContextPointer context) {
        size_t rows;
        this->data().clear();
        troy::serialize::load_object(stream, rows);
        for (size_t i = 0; i < rows; i++) {
            size_t row_size;
            troy::serialize::load_object(stream, row_size);
            std::vector<Ciphertext>& row = this->new_row();
            for (size_t j = 0; j < row_size; j++) {
                Ciphertext cipher;
                cipher.load(stream, context);
                row.push_back(std::move(cipher));
            }
        }
    }
    
    size_t Cipher2d::serialized_size(HeContextPointer context) const {
        throw std::runtime_error("[Cipher2d::serialized_size] Not implemented.");
        size_t bytes = 0;
        bytes += sizeof(size_t); // rows
        for (size_t i = 0; i < this->rows(); i++) {
            bytes += sizeof(size_t); // row size
            for (size_t j = 0; j < (*this)[i].size(); j++) {
                bytes += (*this)[i][j].serialized_size(context);
            }
        }
        return bytes;
    }


    Cipher2d Cipher2d::translate(const Evaluator& evaluator, const Cipher2d& other, bool subtract) const {
        if (this->size() != other.size()) {
            throw std::runtime_error("[Cipher2d::translate] Row size mismatch.");
        }
        Cipher2d result;
        Ciphertext buffer;
        for (size_t i = 0; i < this->rows(); i++) {
            if ((*this)[i].size() != other[i].size()) {
                throw std::runtime_error("[Cipher2d::translate] Column size mismatch.");
            }
            std::vector<Ciphertext>& row = result.new_row();
            for (size_t j = 0; j < (*this)[i].size(); j++) {
                if (!subtract) {
                    evaluator.add((*this)[i][j], other[i][j], buffer);
                } else {
                    evaluator.sub((*this)[i][j], other[i][j], buffer);
                }
                row.push_back(std::move(buffer));
            }
        }
        return result;
    }

    void Cipher2d::translate_inplace(const Evaluator& evaluator, const Cipher2d& other, bool subtract) {
        if (this->size() != other.size()) {
            throw std::runtime_error("[Cipher2d::translate_inplace] Row size mismatch.");
        }
        for (size_t i = 0; i < this->rows(); i++) {
            if ((*this)[i].size() != other[i].size()) {
                throw std::runtime_error("[Cipher2d::translate_inplace] Column size mismatch.");
            }
            for (size_t j = 0; j < (*this)[i].size(); j++) {
                if (!subtract) {
                    evaluator.add_inplace((*this)[i][j], other[i][j]);
                } else {
                    evaluator.sub_inplace((*this)[i][j], other[i][j]);
                }
            }
        }
    }

    Cipher2d Cipher2d::translate_plain(const Evaluator& evaluator, const Plain2d& other, bool subtract) const {
        if (this->size() != other.size()) {
            throw std::runtime_error("[Cipher2d::translate_plain] Row size mismatch.");
        }
        Cipher2d result;
        Ciphertext buffer;
        for (size_t i = 0; i < this->rows(); i++) {
            if ((*this)[i].size() != other[i].size()) {
                throw std::runtime_error("[Cipher2d::translate_plain] Column size mismatch.");
            }
            std::vector<Ciphertext>& row = result.new_row();
            for (size_t j = 0; j < (*this)[i].size(); j++) {
                if (!subtract) {
                    evaluator.add_plain((*this)[i][j], other[i][j], buffer);
                } else {
                    evaluator.sub_plain((*this)[i][j], other[i][j], buffer);
                }
                row.push_back(std::move(buffer));
            }
        }
        return result;
    }

    void Cipher2d::translate_plain_inplace(const Evaluator& evaluator, const Plain2d& other, bool subtract) {
        if (this->size() != other.size()) {
            throw std::runtime_error("[Cipher2d::translate_plain_inplace] Row size mismatch.");
        }
        for (size_t i = 0; i < this->rows(); i++) {
            if ((*this)[i].size() != other[i].size()) {
                throw std::runtime_error("[Cipher2d::translate_plain_inplace] Column size mismatch.");
            }
            for (size_t j = 0; j < (*this)[i].size(); j++) {
                if (!subtract) {
                    evaluator.add_plain_inplace((*this)[i][j], other[i][j]);
                } else {
                    evaluator.sub_plain_inplace((*this)[i][j], other[i][j]);
                }
            }
        }
    }

}}