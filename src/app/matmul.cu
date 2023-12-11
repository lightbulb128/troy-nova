#include "matmul.cuh"

namespace troy { namespace linear {

    void MatmulHelper::determine_block() {
        size_t b_best = 0, i_best = 0, o_best = 0;
        size_t c_best = 2147483647;
        if (!pack_lwe) {
            for (size_t b = batch_size; b >= 1; b--) {
                size_t bc = ceil_div(batch_size, b);
                if (b >= slot_count) continue;
                if (bc * 2 > c_best) continue;
                for (size_t i = 1; i < slot_count / b; i++) {
                    size_t o = slot_count / b / i;
                    if (o > output_dims) o = output_dims;
                    if (i > input_dims) continue;
                    if (o < 1) continue;
                    size_t c = 0;
                    if (objective == MatmulObjective::EncryptLeft) {
                        c = bc * (ceil_div(input_dims, i) + ceil_div(output_dims, o));
                    } else if (objective == MatmulObjective::EncryptRight) {
                        c = (bc + ceil_div(input_dims, i)) * ceil_div(output_dims, o);
                    } else if (objective == MatmulObjective::Crossed) {
                        c = bc * input_dims + (bc + ceil_div(input_dims, i)) * ceil_div(output_dims, o);
                    } else {
                        throw std::runtime_error("[MatmulHelper::determine_block] Invalid objective");
                    }
                    if (c >= c_best) continue;
                    b_best = b; i_best = i; o_best = o; c_best = c;
                }
            }
        } else {
            double sqrtn = std::pow(slot_count, 0.33);
            size_t i = 1; while (i * 2 < sqrtn) {i *= 2;}
            if (i > input_dims) {
                i = 1; while (i < input_dims) i *= 2;
            }
            
            for (size_t b = 1; b <= batch_size; b++) {
                size_t bc = ceil_div(batch_size, b);
                if (b > slot_count) {continue;}
                size_t o = slot_count / b / i;
                if (o > output_dims) {o = output_dims;}
                if (o < 1) {continue;}
                size_t ic = ceil_div(input_dims, i);
                size_t oc = ceil_div(output_dims, o);
                size_t c = 0;
                if (objective == MatmulObjective::EncryptLeft) {
                    c = bc * (ceil_div(input_dims, i) + ceil_div(output_dims, o));
                } else if (objective == MatmulObjective::EncryptRight) {
                    c = (bc + ceil_div(input_dims, i)) * ceil_div(output_dims, o);
                } else if (objective == MatmulObjective::Crossed) {
                    c = bc * input_dims + (bc + ceil_div(input_dims, i)) * ceil_div(output_dims, o);
                } else {
                    throw std::runtime_error("[MatmulHelper::determine_block] Invalid objective");
                }
                c = ceil_div(c, i);
                if (c >= c_best) {continue;}
                b_best = b; i_best = i; o_best = o; c_best = c;
            }

        }
        batch_block = b_best;
        input_block = i_best;
        output_block = o_best;
        // printf("block (%zu, %zu, %zu) -> (%zu, %zu, %zu)\n", batch_size, input_dims, output_dims, batch_block, input_block, output_block);
    }
    
    Plaintext MatmulHelper::encode_weight_small(
        const BatchEncoder& encoder,
        const uint64_t* weights,
        size_t li, size_t ui, size_t lj, size_t uj
    ) const {
        std::vector<uint64_t> vec(input_block * output_block, 0);
        for (size_t j = lj; j < uj; j++) {
            for (size_t i = li; i < ui; i++) {
                size_t r = (j-lj) * input_block + input_block - (i-li) - 1;
                assert(r < slot_count);
                vec[r] = weights[i * output_dims + j];
            }
        }
        Plaintext ret;
        encoder.encode_polynomial(vec, ret);
        return ret;
    }


    Plain2d MatmulHelper::encode_weights(
        const BatchEncoder& encoder,
        const uint64_t* weights
    ) const {
        size_t height = input_dims, width = output_dims;
        size_t h = input_block, w = output_block;
        Plain2d encoded_weights;
        encoded_weights.data().clear();
        encoded_weights.data().reserve(ceil_div(height, h));
        for (size_t li = 0; li < height; li += h) {
            size_t ui = (li + h > height) ? height : (li + h);
            std::vector<Plaintext> encoded_row; encoded_row.reserve(ceil_div(width, w));
            for (size_t lj = 0; lj < width; lj += w) {
                size_t uj = (lj + w > width) ? width : (lj + w);
                encoded_row.push_back(
                    encode_weight_small(encoder, weights, li, ui, lj, uj)
                );
            }
            encoded_weights.data().push_back(std::move(encoded_row));
        }
        return encoded_weights;
    }

    Plain2d MatmulHelper::encode_inputs(
        const BatchEncoder& encoder,
        const uint64_t* inputs
    ) const {
        size_t vecsize = input_block;
        Plain2d ret;
        ret.data().reserve(batch_size);
        for (size_t li = 0; li < batch_size; li += batch_block) {
            size_t ui = (li + batch_block > batch_size) ? batch_size : li + batch_block;
            std::vector<Plaintext> encoded_row;
            encoded_row.reserve(ceil_div(input_dims, vecsize));
            for (size_t lj = 0; lj < input_dims; lj += vecsize) {
                size_t uj = (lj + vecsize > input_dims) ? input_dims : lj + vecsize;
                std::vector<uint64_t> vec(slot_count, 0);
                for (size_t i = li; i < ui; i++)
                    for (size_t j = lj; j < uj; j++)
                        vec[(i - li) * input_block * output_block + (j - lj)] = inputs[i * input_dims + j];
                Plaintext encoded;
                encoder.encode_polynomial(vec, encoded);
                encoded_row.push_back(std::move(encoded));
            }
            ret.data().push_back(std::move(encoded_row));
        }
        return ret;
    }

    Cipher2d MatmulHelper::encrypt_inputs(
        const Encryptor& encryptor,
        const BatchEncoder& encoder, 
        const uint64_t* inputs
    ) const {
        Plain2d plain = encode_inputs(encoder, inputs);
        return plain.encrypt_symmetric(encryptor);
    }

    Cipher2d MatmulHelper::matmul(const Evaluator& evaluator, const Cipher2d& a, const Plain2d& w) const {
        Cipher2d ret; ret.data().reserve(ceil_div(batch_size, batch_block));
        size_t outputVectorCount = ceil_div(output_dims, output_block);
        if (a.data().size() != ceil_div(batch_size, batch_block)) {
            throw std::invalid_argument("[MatmulHelper::matmul] Input batch_size incorrect.");
        }
        if (w.data().size() != ceil_div(input_dims, input_block)) {
            throw std::invalid_argument("[MatmulHelper::matmul] Weight input dimension incorrect.");
        }
        for (size_t b = 0; b < ceil_div(batch_size, batch_block); b++) {
            std::vector<Ciphertext> outVecs(outputVectorCount);
            for (size_t i = 0; i < w.data().size(); i++) {
                for (size_t j = 0; j < w[i].size(); j++) {
                    Ciphertext prod;
                    evaluator.multiply_plain(a[b][i], w[i][j], prod);
                    if (i==0) outVecs[j] = std::move(prod);
                    else {
                        evaluator.add_inplace(outVecs[j], prod);
                    }
                }
            }
            ret.data().push_back(std::move(outVecs));
        }
        return ret;
    }

    Cipher2d MatmulHelper::matmul_cipher(const Evaluator& evaluator, const Cipher2d& a, const Cipher2d& w) const {
        Cipher2d ret; ret.data().reserve(ceil_div(batch_size, batch_block));
        size_t outputVectorCount = ceil_div(output_dims, output_block);
        if (a.data().size() != ceil_div(batch_size, batch_block)) {
            throw std::invalid_argument("[MatmulHelper::matmul_cipher] Input batch_size incorrect.");
        }
        if (w.data().size() != ceil_div(input_dims, input_block)) {
            throw std::invalid_argument("[MatmulHelper::matmul_cipher] Weight input dimension incorrect.");
        }
        for (size_t b = 0; b < ceil_div(batch_size, batch_block); b++) {
            std::vector<Ciphertext> outVecs(outputVectorCount);
            for (size_t i = 0; i < w.data().size(); i++) {
                for (size_t j = 0; j < w[i].size(); j++) {
                    Ciphertext prod;
                    evaluator.multiply(a[b][i], w[i][j], prod);
                    if (i==0) outVecs[j] = std::move(prod);
                    else {
                        evaluator.add_inplace(outVecs[j], prod);
                    }
                }
            }
            ret.data().push_back(std::move(outVecs));
        }
        return ret;
    }

    Cipher2d MatmulHelper::matmul_reverse(const Evaluator& evaluator, const Plain2d& a, const Cipher2d& w) const {
        Cipher2d ret; ret.data().reserve(ceil_div(batch_size, batch_block));
        size_t outputVectorCount = ceil_div(output_dims, output_block);
        if (a.data().size() != ceil_div(batch_size, batch_block)) {
            throw std::invalid_argument("[MatmulHelper::matmul_reverse] Input batch_size incorrect.");
        }
        if (w.data().size() != ceil_div(input_dims, input_block)) {
            throw std::invalid_argument("[MatmulHelper::matmul_reverse] Weight input dimension incorrect.");
        }
        for (size_t b = 0; b < ceil_div(batch_size, batch_block); b++) {
            std::vector<Ciphertext> outVecs(outputVectorCount);
            for (size_t i = 0; i < w.data().size(); i++) {
                for (size_t j = 0; j < w[i].size(); j++) {
                    Ciphertext prod;
                    evaluator.multiply_plain(w[i][j], a[b][i], prod);
                    if (i==0) outVecs[j] = std::move(prod);
                    else {
                        evaluator.add_inplace(outVecs[j], prod);
                    }
                }
            }
            ret.data().push_back(std::move(outVecs));
        }
        return ret;
    }

    Plain2d MatmulHelper::encode_outputs(
        const BatchEncoder& encoder, 
        const uint64_t* outputs
    ) const {
        size_t vecsize = output_block;
        Plaintext pt;
        if (!this->pack_lwe) {
            Plain2d ret; ret.data().reserve(batch_size);
            for (size_t li = 0; li < batch_size; li += batch_block) {
                size_t ui = (li + batch_block > batch_size) ? batch_size : (li + batch_block);
                std::vector<Plaintext> encoded_row;
                encoded_row.reserve(ceil_div(output_dims, vecsize));
                for (size_t lj = 0; lj < output_dims; lj += vecsize) {
                    size_t uj = (lj + vecsize > output_dims) ? output_dims : (lj + vecsize);
                    std::vector<uint64_t> buffer(slot_count, 0);
                    for (size_t i = li; i < ui; i++)
                        for (size_t j = lj; j < uj; j++) 
                            buffer[(i - li) * input_block * output_block + (j - lj) * input_block + input_block - 1] = outputs[i * output_dims + j];
                    encoder.encode_polynomial(buffer, pt);
                    encoded_row.push_back(std::move(pt));
                }
                ret.data().push_back(std::move(encoded_row));
            }
            return ret;
        } else {
            Plain2d plain2d; plain2d.data().reserve(batch_size);
            plain2d.data().push_back(std::vector<Plaintext>());
            size_t batch_blockCount = ceil_div(this->batch_size, this->batch_block);
            size_t output_blockCount = ceil_div(this->output_dims, this->output_block);
            auto ret = std::vector<std::vector<uint64_t>>(ceil_div(batch_blockCount * output_blockCount, this->input_block), std::vector<uint64_t>(this->slot_count, 0)); 
            size_t li = 0; size_t di = 0; while (li < this->batch_size) {
                size_t ui = std::min(this->batch_size, li + this->batch_block);
                size_t lj = 0; size_t dj = 0; while (lj < this->output_dims) {
                    size_t uj = std::min(this->output_dims, lj + vecsize);
                    size_t cipherId = di * ceil_div(this->output_dims, this->output_block) + dj;
                    size_t packedId = cipherId / this->input_block;
                    size_t packedOffset = cipherId % this->input_block;
                    for (size_t i = li; i < ui; i++) {
                        for (size_t j = lj; j < uj; j++) {
                            ret[packedId][(i - li) * this->input_block * this->output_block + (j - lj) * this->input_block + packedOffset] 
                                = outputs[i * this->output_dims + j];
                        }
                    }
                    dj += 1;
                    lj += vecsize; 
                }
                di += 1;
                li += this->batch_block;
            }
            plain2d.data()[0].reserve(ret.size());
            for (size_t i = 0; i < ret.size(); i++) {
                encoder.encode_polynomial(ret[i], pt);
                plain2d.data()[0].push_back(std::move(pt));
            }
            return plain2d;
        }
    }

    std::vector<uint64_t> MatmulHelper::decrypt_outputs(
        const BatchEncoder& encoder,
        const Decryptor& decryptor,
        const Cipher2d& outputs
    ) const {
        std::vector<uint64_t> dec(batch_size * output_dims);
        size_t vecsize = output_block;
        Plaintext pt;
        if (!this->pack_lwe) {
            std::vector<uint64_t> buffer(slot_count);
            size_t di = 0;
            for (size_t li = 0; li < batch_size; li += batch_block) {
                size_t ui = (li + batch_block > batch_size) ? batch_size : (li + batch_block);
                size_t dj = 0;
                for (size_t lj = 0; lj < output_dims; lj += vecsize) {
                    size_t uj = (lj + vecsize > output_dims) ? output_dims : (lj + vecsize);
                    decryptor.decrypt(outputs[di][dj], pt);
                    encoder.decode_polynomial(pt, buffer);
                    for (size_t i = li; i < ui; i++)
                        for (size_t j = lj; j < uj; j++) 
                            dec[i * output_dims + j] = buffer[(i - li) * input_block * output_block + (j - lj) * input_block + input_block - 1];
                    dj += 1;
                }
                di += 1;
            }
        } else {
            std::vector<std::vector<uint64_t>> buffer(outputs[0].size(), std::vector<uint64_t>(slot_count, 0));
            for (size_t i = 0; i < outputs.data()[0].size(); i++) {
                decryptor.decrypt(outputs[0][i], pt);
                encoder.decode_polynomial(pt, buffer[i]);
            }
            size_t li = 0; size_t di = 0; while (li < this->batch_size) {
                size_t ui = std::min(this->batch_size, li + this->batch_block);
                size_t lj = 0; size_t dj = 0; while (lj < this->output_dims) {
                    size_t uj = std::min(this->output_dims, lj + vecsize);
                    size_t cipherId = di * ceil_div(this->output_dims, this->output_block) + dj;
                    size_t packedId = cipherId / this->input_block;
                    size_t packedOffset = cipherId % this->input_block;
                    for (size_t i = li; i < ui; i++) {
                        for (size_t j = lj; j < uj; j++) {
                            dec[i * output_dims + j] = buffer[packedId][(i - li) * input_block * output_block + (j - lj) * input_block + packedOffset];
                        }
                    }
                    dj += 1;
                    lj += vecsize; 
                }
                di += 1;
                li += this->batch_block;
            }
        }
        return dec;
    }

    Cipher2d MatmulHelper::packOutputs(const Evaluator& evaluator, const GaloisKeys& autoKey, const Cipher2d& cipher) const {
        if (!this->pack_lwe) {
            throw std::invalid_argument("[MatmulHelper::packOutputs] PackLwe not enabled");
        }
        if (cipher.data().size() == 0 || cipher.data()[0].size() == 0) {
            Cipher2d ret; ret.data().push_back(std::vector<Ciphertext>());
            return ret;
        }
        size_t packSlots = this->input_block;
        size_t totalCount = cipher.data().size() * cipher.data()[0].size();
        std::vector<Ciphertext> output; output.reserve(ceil_div(totalCount, packSlots));
        Ciphertext current; bool currentSet = false;
        size_t currentSlot = 0;
        
        size_t field_trace_logn = 0;
        size_t field_trace_n = 1;
        while (field_trace_n != slot_count / packSlots) {
            field_trace_logn += 1;
            field_trace_n *= 2;
        }

        Ciphertext buffer = cipher.data()[0][0];
        Ciphertext shifted = buffer;
        for (size_t i = 0; i < cipher.data().size(); i++) {
            for (size_t j = 0; j < cipher.data()[0].size(); j++) {
                size_t shift = packSlots - 1;
                const Ciphertext& ciphertext = cipher.data()[i][j];
                if (shift != 0) {
                    evaluator.negacyclic_shift(ciphertext, 2 * slot_count - shift, buffer);
                } else {
                    buffer = ciphertext;
                }
                evaluator.divide_by_poly_modulus_degree_inplace(buffer, slot_count / packSlots);
                evaluator.field_trace_inplace(buffer, autoKey, field_trace_logn);
                shift = currentSlot;
                if (shift != 0) {
                    evaluator.negacyclic_shift(buffer, shift, shifted);
                } else {
                    shifted = buffer;
                }
                if (currentSet == false) {
                    current = shifted;
                    currentSet = true;
                } else {
                    evaluator.add_inplace(current, shifted);
                }
                currentSlot += 1;
                if (currentSlot == packSlots) {
                    currentSlot = 0; currentSet = false;
                    output.push_back(std::move(current));
                }
            }
        }
        if (currentSet) {
            output.push_back(std::move(current));
        }
        Cipher2d ret; ret.data().push_back(output);
        return ret;
    }

    void MatmulHelper::serialize_encoded_weights(const Plain2d& w, std::ostream& stream) const {
        size_t rows = w.data().size();
        size_t cols = w[0].size();
        if (rows == 0) throw std::invalid_argument("[MatmulHelper::serialize_encoded_weights] No rows in weight matrix.");
        if (cols == 0) throw std::invalid_argument("[MatmulHelper::serialize_encoded_weights] No columns in weight matrix.");
        for (size_t i=0; i<rows; i++) {
            if (w[i].size() != cols) throw std::invalid_argument("[MatmulHelper::serialize_encoded_weights] Weight matrix is not rectangular.");
        }
        serialize::save_object(stream, rows);
        serialize::save_object(stream, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                w[i][j].save(stream);
            }
        }
    }

    Plain2d MatmulHelper::deserialize_encoded_weights(std::istream& stream) const {
        size_t rows, cols;
        serialize::load_object(stream, rows);
        serialize::load_object(stream, cols);
        Plain2d ret; ret.data().reserve(rows);
        for (size_t i = 0; i < rows; i++) {
            std::vector<Plaintext> row; row.reserve(cols);
            for (size_t j = 0; j < cols; j++) {
                Plaintext pt;
                pt.load(stream);
                row.push_back(std::move(pt));
            }
            ret.data().push_back(std::move(row));
        }
        return ret;
    }

    void MatmulHelper::serialize_outputs(const Evaluator &evaluator, const Cipher2d& x, std::ostream& stream) const {
        HeContextPointer context = evaluator.context();
        if (!this->pack_lwe) {
            size_t vecsize = output_block;
            Plaintext pt;
            size_t di = 0;
            for (size_t li = 0; li < batch_size; li += batch_block) {
                size_t ui = (li + batch_block > batch_size) ? batch_size : (li + batch_block);
                size_t dj = 0;
                for (size_t lj = 0; lj < output_dims; lj += vecsize) {
                    size_t uj = (lj + vecsize > output_dims) ? output_dims : (lj + vecsize);
                    std::vector<size_t> required((ui - li) * (uj - lj)); size_t rid = 0;
                    for (size_t i = li; i < ui; i++)
                        for (size_t j = lj; j < uj; j++) 
                            required[rid++] = (i - li) * input_block * output_block + (j - lj) * input_block + input_block - 1;
                    x[di][dj].save_terms(stream, context, required);
                    dj += 1;
                }
                di += 1;
            }
        } else {
            size_t count = ceil_div(batch_size, batch_block) * ceil_div(output_dims, output_block);
            count = ceil_div(count, input_block);
            if (count != x.data()[0].size()) {
                throw std::invalid_argument("[MatmulHelper::serialize_outputs] Output ciphertext count incorrect");
            }
            for (size_t i = 0; i < x.data()[0].size(); i++) {
                x[0][i].save(stream, context);
            }
        }
    }

    Cipher2d MatmulHelper::deserialize_outputs(const Evaluator &evaluator, std::istream& stream) const {
        HeContextPointer context = evaluator.context();
        if (!this->pack_lwe) {
            size_t vecsize = output_block;
            Plaintext pt;
            Cipher2d ret; ret.data().reserve(ceil_div(batch_size, batch_block));
            for (size_t li = 0; li < batch_size; li += batch_block) {
                size_t ui = (li + batch_block > batch_size) ? batch_size : (li + batch_block);
                std::vector<Ciphertext> row; row.reserve(ceil_div(output_dims, vecsize));
                for (size_t lj = 0; lj < output_dims; lj += vecsize) {
                    size_t uj = (lj + vecsize > output_dims) ? output_dims : (lj + vecsize);
                    std::vector<size_t> required((ui - li) * (uj - lj)); size_t rid = 0;
                    for (size_t i = li; i < ui; i++)
                        for (size_t j = lj; j < uj; j++) 
                            required[rid++] = (i - li) * input_block * output_block + (j - lj) * input_block + input_block - 1;
                    Ciphertext c;
                    c.load_terms(stream, context, required);
                    row.push_back(std::move(c));
                }
                ret.data().push_back(std::move(row));
            }
            return ret;
        } else {
            size_t count = ceil_div(batch_size, batch_block) * ceil_div(output_dims, output_block);
            count = ceil_div(count, input_block);
            Cipher2d ret; ret.data().push_back(std::vector<Ciphertext>());
            ret[0].reserve(count);
            for (size_t i = 0; i < count; i++) {
                Ciphertext c; c.load(stream, context);
                ret[0].push_back(std::move(c));
            }
            return ret;
        }
    }

}}