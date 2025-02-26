#include "conv2d.h"

namespace troy { namespace linear {
    
    std::ostream& operator<<(std::ostream& os, const Conv2dHelper& helper) {
        os << "Conv2dHelper("
            << "batch_size=" << helper.batch_size << ", "
            << "input_channels=" << helper.input_channels << ", "
            << "output_channels=" << helper.output_channels << ", "
            << "image_height=" << helper.image_height << ", "
            << "image_width=" << helper.image_width << ", "
            << "kernel_height=" << helper.kernel_height << ", "
            << "kernel_width=" << helper.kernel_width << ", "
            << "slot_count=" << helper.slot_count << ", "
            << "objective=" << helper.objective
            << ")";
        return os;
    }

    using uint128_t = __uint128_t;


    #define D_IMPL_ALL                                               \
        D_IMPL(BatchEncoderAdapter, uint64_t)                        \
        D_IMPL(CKKSEncoderAdapter, double)                           \
        D_IMPL(PolynomialEncoderRing2kAdapter<uint32_t>, uint32_t)   \
        D_IMPL(PolynomialEncoderRing2kAdapter<uint64_t>, uint64_t)   \
        D_IMPL(PolynomialEncoderRing2kAdapter<uint128_t>, uint128_t)


    void Conv2dHelper::determine_block() {
        size_t best = 2147483647;
        // find b, h, w, ci, co, such that minimizes (ceil(B/b)*ceil((H-kh+1)/(h-kh+1))*ceil((W-kh+1)/(h-kh+1))*(ceil(Ci/ci)+ceil(Co/co)))
        size_t bestB = 0, bestH = 0, bestW = 0, bestCi = 0, bestCo = 0;
        for (size_t b = batch_size; b >= 1; b--) {
            size_t upper = slot_count / b;
            for (size_t h = std::min(image_height, upper); h >= kernel_height; h--) {
                size_t upper = slot_count / b / h;
                for (size_t w = std::min(image_width, upper); w >= kernel_width; w--) {
                    size_t upper = slot_count / b / h / w;
                    for (size_t co = std::min(output_channels, upper); co >= 1; co--) {
                        size_t ci = slot_count / b / h / w / co;
                        ci = std::min(ci, input_channels);
                        if (ci == 0) continue;
                        size_t input_cipher_size = (
                            ceil_div(batch_size, b) * 
                            ceil_div(image_height - kernel_height + 1, h - kernel_height + 1) * 
                            ceil_div(image_width - kernel_width + 1, w - kernel_width + 1) * 
                            ceil_div(input_channels, ci)
                        );
                        size_t output_cipher_size = (
                            ceil_div(batch_size, b) * 
                            ceil_div(image_height - kernel_height + 1, h - kernel_height + 1) * 
                            ceil_div(image_width - kernel_width + 1, w - kernel_width + 1) * 
                            ceil_div(output_channels, co)
                        );
                        size_t weight_cipher_size = (
                            ceil_div(input_channels, ci) * 
                            ceil_div(output_channels, co)
                        );
                        size_t current = 0;
                        if (objective == MatmulObjective::EncryptLeft) {
                            current = input_cipher_size + output_cipher_size;
                        } else if (objective == MatmulObjective::EncryptRight) {
                            current = weight_cipher_size + output_cipher_size;
                        } else if (objective == MatmulObjective::Crossed) {
                            current = output_cipher_size + input_cipher_size + weight_cipher_size;
                        } else {
                            throw std::runtime_error("Conv2dHelper: invalid objective");
                        }
                        if (current < best) {
                            best = current;
                            bestB = b;
                            bestH = h;
                            bestW = w;
                            bestCi = ci;
                            bestCo = co;
                        }
                    }
                }
            }
        }
        batch_block = bestB;
        image_height_block = bestH;
        image_width_block = bestW;
        input_channel_block = bestCi;
        output_channel_block = bestCo;
        // printf("Conv2dHelper: batch_block = %zu, image_height_block = %zu, image_width_block = %zu, input_channel_block = %zu, output_channel_block = %zu\n", batch_block, image_height_block, image_width_block, input_channel_block, output_channel_block);
    }

    template <typename E, typename T>
    void Conv2dHelper::encode_weights(
        const E& encoder, const Encryptor* encryptor, const T* weights, 
        bool for_cipher, Plain2d* out_plain, Cipher2d* out_cipher
    ) const {
        size_t block_size = image_height_block * image_width_block;
        if (out_plain) {
            out_plain->data().clear();
            out_plain->data().reserve(ceil_div(output_channels, output_channel_block));
        } else {
            if (!out_cipher || !encryptor) throw std::runtime_error("[Conv2dHelper::encode_weights]: out_cipher is null");
            out_cipher->data().clear();
            out_cipher->data().reserve(ceil_div(output_channels, output_channel_block));
        }
        Plain2d temp_plain; 
        temp_plain.data().clear();
        temp_plain.data().reserve(ceil_div(output_channels, output_channel_block));
        for (size_t loc = 0; loc < output_channels; loc += output_channel_block) {
            size_t uoc = std::min(loc + output_channel_block, output_channels);
            std::vector<Plaintext> current_channel_plain;
            current_channel_plain.reserve(ceil_div(input_channels, input_channel_block));
            for (size_t lic = 0; lic < input_channels; lic += input_channel_block) {
                size_t uic = std::min(lic + input_channel_block, input_channels);
                std::vector<T> spread(input_channel_block * output_channel_block * image_height_block * image_width_block, 0);
                for (size_t oc = loc; oc < uoc; oc++) {
                    for (size_t ic = lic; ic < uic; ic++) {
                        for (size_t ki = 0; ki < kernel_height; ki++) {
                            for (size_t kj = 0; kj < kernel_width; kj++) {
                                // spread[channel_slots - 1 - (j - lic), :k_h, :k_w] = np.flip(weight[oc, j])
                                size_t spreadIndex = (oc - loc) * input_channel_block * block_size + (input_channel_block - 1 - (ic - lic)) * block_size + ki * image_width_block + kj;
                                size_t weightIndex = ((oc * input_channels) + ic) * (kernel_height * kernel_width) + (kernel_height - ki - 1) * kernel_width + (kernel_width - kj - 1);
                                spread[spreadIndex] = weights[weightIndex];
                            }
                        }
                    }
                }
                Plaintext pt = for_cipher
                    ? encoder.encode_for_cipher(spread, pool)
                    : encoder.encode_for_plain(spread, pool);
                current_channel_plain.push_back(std::move(pt));
            }
            temp_plain.data().push_back(std::move(current_channel_plain));
        }
        if (!temp_plain.data()[0][0].is_ntt_form()) {
            Plain2d temp; ensure_ntt_form(batched_mul, encoder.context(), pool, temp_plain, temp, !for_cipher);
            temp_plain = std::move(temp);
        }
        if (out_plain) {
            *out_plain = std::move(temp_plain);
        } else {
            std::vector<const Plaintext*> temp_ptrs;
            std::vector<Ciphertext*> out_ptrs; 
            for (size_t i = 0; i < temp_plain.size(); i++) {
                out_cipher->data().push_back(std::vector<Ciphertext>(temp_plain[i].size()));
                for (size_t j = 0; j < temp_plain[i].size(); j++) {
                    temp_ptrs.push_back(&temp_plain[i][j]);
                    out_ptrs.push_back(&(*out_cipher)[i][j]);
                }
            }
            encryptor->encrypt_symmetric_batched(temp_ptrs, true, out_ptrs, nullptr, pool);
        }
    }

    #define D_IMPL(adapter, dtype) \
        template void Conv2dHelper::encode_weights<adapter, dtype>( \
            const adapter& encoder, const Encryptor* encryptor, const dtype* weights, \
            bool for_cipher, Plain2d* out_plain, Cipher2d* out_cipher \
        ) const;
    D_IMPL_ALL
    #undef D_IMPL

    size_t Conv2dHelper::get_total_batch_size() const {
        size_t kh = kernel_height - 1, kw = kernel_width - 1;
        size_t sh = ceil_div(image_height - kh, image_height_block - kh);
        size_t sw = ceil_div(image_width - kw, image_width_block - kw);
        return ceil_div(batch_size, batch_block) * sh * sw;
    }

    template <typename E, typename T>
    void Conv2dHelper::encode_inputs(
        const E& encoder, const Encryptor* encryptor, const T* inputs,
        bool for_cipher, Plain2d* out_plain, Cipher2d* out_cipher
    ) const {
        size_t kh = kernel_height - 1, kw = kernel_width - 1;
        size_t sh = ceil_div(image_height - kh, image_height_block - kh);
        size_t sw = ceil_div(image_width - kw, image_width_block - kw);
        size_t image_size = image_height * image_width;
        size_t block_size = image_height_block * image_width_block;
        size_t total_batch_size = ceil_div(batch_size, batch_block) * sh * sw;
        if (out_plain) {
            out_plain->data().clear();
            out_plain->data().reserve(total_batch_size);
        } else {
            if (!out_cipher || !encryptor) throw std::runtime_error("[Conv2dHelper::encode_inputs]: out_cipher is null");
            out_cipher->data().clear();
            out_cipher->data().reserve(total_batch_size);
        }
        Plain2d temp_plain;
        temp_plain.data().clear();
        temp_plain.data().reserve(total_batch_size);
        for (size_t lb = 0; lb < batch_size; lb += batch_block) {
            size_t ub = std::min(lb + batch_block, batch_size);
            for (size_t ih = 0; ih < sh; ih++) {
                for (size_t iw = 0; iw < sw; iw++) {
                    size_t si = ih * (image_height_block - kh);
                    size_t sj = iw * (image_width_block - kw);
                    size_t ui = std::min(si + image_height_block, image_height);
                    size_t uj = std::min(sj + image_width_block, image_width);
                    std::vector<Plaintext> group_plain;
                    group_plain.reserve(ceil_div(input_channels, input_channel_block));
                    for (size_t lci = 0; lci < input_channels; lci += input_channel_block) {
                        size_t uci = std::min(lci + input_channel_block, input_channels);
                        std::vector<T> vec(slot_count, 0);
                        for (size_t b = 0; b < ub-lb; b++) {
                            for (size_t tci = 0; tci < uci-lci; tci++) {
                                for (size_t ti = si; ti < ui; ti++) {
                                    for (size_t tj = sj; tj < uj; tj++) {
                                        size_t inputIndex = (lb + b) * input_channels * image_size + (lci + tci) * image_size + ti * image_width + tj;
                                        size_t vecIndex = b * input_channel_block * output_channel_block * block_size 
                                            + tci * block_size + (ti - si) * image_width_block + (tj - sj);
                                        // printf("inputIndex: %lu, vecIndex: %lu, b=%lu, tci=%lu,ti-si=%lu, tj-sj=%ld\n", inputIndex, vecIndex, b, tci, ti-si, tj-sj);
                                        vec[vecIndex] = inputs[inputIndex];
                                        // printf("ok inputIndex: %lu, vecIndex: %lu\n", inputIndex, vecIndex);
                                    }
                                }
                            }
                        }
                        // printf("encode lb=%lu, ub=%lu, ih=%lu, iw=%lu, lci=%lu, uci=%lu, vecsize=%lu\n", lb, ub, ih, iw, lci, uci, vec.size());
                        Plaintext pt = for_cipher
                            ? encoder.encode_for_cipher(vec, pool)
                            : encoder.encode_for_plain(vec, pool);
                        group_plain.push_back(std::move(pt));
                    }
                    temp_plain.data().push_back(std::move(group_plain));
                }
            }
        }
        if (!temp_plain.data()[0][0].is_ntt_form()) {
            Plain2d temp; ensure_ntt_form(batched_mul, encoder.context(), pool, temp_plain, temp, !for_cipher);
            temp_plain = std::move(temp);
        }
        if (out_plain) {
            *out_plain = std::move(temp_plain);
        } else {
            std::vector<const Plaintext*> temp_ptrs;
            std::vector<Ciphertext*> out_ptrs; 
            for (size_t i = 0; i < temp_plain.size(); i++) {
                out_cipher->data().push_back(std::vector<Ciphertext>(temp_plain[i].size()));
                for (size_t j = 0; j < temp_plain[i].size(); j++) {
                    temp_ptrs.push_back(&temp_plain[i][j]);
                    out_ptrs.push_back(&(*out_cipher)[i][j]);
                }
            }
            encryptor->encrypt_symmetric_batched(temp_ptrs, true, out_ptrs, nullptr, pool);
        }
    }

    #define D_IMPL(adapter, dtype) \
        template void Conv2dHelper::encode_inputs<adapter, dtype>( \
            const adapter& encoder, const Encryptor* encryptor, const dtype* inputs, \
            bool for_cipher, Plain2d* out_plain, Cipher2d* out_cipher \
        ) const;
    D_IMPL_ALL
    #undef D_IMPL

    template <typename E, typename T>
    Plain2d Conv2dHelper::encode_outputs(const E& encoder, const T* outputs) const {
        size_t interval = image_width_block * image_height_block;
        std::vector<T> mask(slot_count, 0);
        auto total_batch_size = get_total_batch_size();
        size_t yh = image_height_block - kernel_height + 1;
        size_t yw = image_width_block  - kernel_width  + 1;
        size_t oyh = image_height - kernel_height + 1;
        size_t oyw = image_width - kernel_width + 1;
        Plain2d ret; ret.data().reserve(total_batch_size);
        size_t kh = kernel_height - 1, kw = kernel_width - 1;
        size_t sh = ceil_div(image_height - kh, image_height_block - kh);
        size_t sw = ceil_div(image_width - kw, image_width_block - kw);
        assert(total_batch_size == ceil_div(batch_size, batch_block) * sh * sw);
        Plaintext encoded;
        for (size_t eb = 0; eb < total_batch_size; eb++) {
            size_t ob = eb / (sh * sw);
            size_t si = (eb % (sh * sw)) / sw;
            size_t sj = eb % sw;
            size_t lb = ob * batch_block, ub = std::min(lb + batch_block, batch_size);
            std::vector<Plaintext> group; group.reserve(ceil_div(output_channels, output_channel_block));
            for (size_t lc = 0; lc < output_channels; lc += output_channel_block) {
                size_t uc = std::min(lc + output_channel_block, output_channels);
                for (size_t b = lb; b < ub; b++) {
                    for (size_t c = lc; c < uc; c++) {
                        for (size_t i = 0; i < yh; i++) {
                            for (size_t j = 0; j < yw; j++) {
                                size_t mask_index = ((b - lb) * input_channel_block * output_channel_block + (c - lc) * input_channel_block + input_channel_block - 1) * interval + (image_height_block - yh + i) * image_width_block + (image_width_block - yw + j);
                                size_t original_index = b * output_channels * oyh * oyw + c * oyh * oyw + (si * yh + i) * oyw + (sj * yw + j);
                                if (si * yh + i < oyh && sj * yw + j < oyw) mask[mask_index] = outputs[original_index];
                            }
                        }
                    }
                }
                Plaintext encoded = encoder.encode_for_cipher(mask, pool);
                group.push_back(std::move(encoded));
            }
            ret.data().push_back(std::move(group));
        }
        return ret;
    }

    #define D_IMPL(adapter, dtype) \
        template Plain2d Conv2dHelper::encode_outputs<adapter, dtype>( \
            const adapter& encoder, const dtype* outputs \
        ) const;
    D_IMPL_ALL
    #undef D_IMPL


    template <typename E, typename T>
    std::vector<T> Conv2dHelper::decrypt_outputs(const E& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const {
        size_t interval = image_width_block * image_height_block;
        auto total_batch_size = get_total_batch_size();
        size_t yh = image_height_block - kernel_height + 1;
        size_t yw = image_width_block  - kernel_width  + 1;
        size_t oyh = image_height - kernel_height + 1;
        size_t oyw = image_width - kernel_width + 1;
        std::vector<T> ret(batch_size * output_channels * oyh * oyw, 0);
        size_t kh = kernel_height - 1, kw = kernel_width - 1;
        size_t sh = ceil_div(image_height - kh, image_height_block - kh);
        size_t sw = ceil_div(image_width - kw, image_width_block - kw);
        assert(total_batch_size == ceil_div(batch_size, batch_block) * sh * sw);
        Plaintext encoded;
        for (size_t eb = 0; eb < total_batch_size; eb++) {
            size_t ob = eb / (sh * sw);
            size_t si = (eb % (sh * sw)) / sw;
            size_t sj = eb % sw;
            size_t lb = ob * batch_block, ub = std::min(lb + batch_block, batch_size);
            for (size_t lc = 0; lc < output_channels; lc += output_channel_block) {
                size_t uc = std::min(lc + output_channel_block, output_channels);
                // printf("Decrypting block [%lu][%lu]\n", eb, lc / output_channel_block);
                const Ciphertext& target = outputs[eb][lc / output_channel_block];
                std::vector<T> buffer = encoder.decrypt_outputs(decryptor, target, pool);
                for (size_t b = lb; b < ub; b++) {
                    for (size_t c = lc; c < uc; c++) {
                        for (size_t i = 0; i < yh; i++) {
                            for (size_t j = 0; j < yw; j++) {
                                size_t mask_index = ((b - lb) * input_channel_block * output_channel_block + (c - lc) * input_channel_block + input_channel_block - 1) * interval + (image_height_block - yh + i) * image_width_block + (image_width_block - yw + j);
                                size_t original_index = b * output_channels * oyh * oyw + c * oyh * oyw + (si * yh + i) * oyw + (sj * yw + j);
                                // printf("Original[%lu][%lu][%lu][%lu] <- idx[%lu]\n", b, c, si * yh + i, sj * yw + j, mask_index);
                                if (si * yh + i < oyh && sj * yw + j < oyw) {
                                    ret[original_index] = buffer[mask_index];
                                }
                            }
                        }
                    }
                }
            }
        }
        return ret;
    }

    #define D_IMPL(adapter, dtype) \
        template std::vector<dtype> Conv2dHelper::decrypt_outputs<adapter, dtype>( \
            const adapter& encoder, const Decryptor& decryptor, const Cipher2d& outputs \
        ) const;
    D_IMPL_ALL
    #undef D_IMPL

    Cipher2d Conv2dHelper::conv2d(const Evaluator& evaluator, const Cipher2d& a, const Plain2d& encoded_weights) const {
        size_t total_batch_size = get_total_batch_size();
        Cipher2d ret; ret.data().reserve(total_batch_size);
        size_t group_len = ceil_div(output_channels, output_channel_block);
        size_t input_group_len = ceil_div(input_channels, input_channel_block);
        
        if (!batched_mul) {
            for (size_t b = 0; b < total_batch_size; b++) {
                std::vector<Ciphertext> group; group.reserve(group_len);
                for (size_t oc = 0; oc < group_len; oc++) {
                    Ciphertext cipher;
                    for (size_t i = 0; i < input_group_len; i++) {
                        Ciphertext prod;
                        evaluator.multiply_plain(a[b][i], encoded_weights[oc][i], prod, pool);
                        if (i==0) cipher = std::move(prod);
                        else evaluator.add_inplace(cipher, prod, pool);
                    }
                    group.push_back(std::move(cipher));
                }
                ret.data().push_back(std::move(group));
            }
        } else {
            using std::vector;
            vector<const Ciphertext*> a_ptrs; a_ptrs.reserve(total_batch_size * group_len * input_group_len);
            vector<const Plaintext*> w_ptrs; w_ptrs.reserve(total_batch_size * group_len * input_group_len);
            vector<Ciphertext*> r_ptrs; r_ptrs.reserve(total_batch_size * group_len * input_group_len);
            for (size_t b = 0; b < total_batch_size; b++) {
                ret.data().push_back(vector<Ciphertext>(group_len));
            }
            for (size_t b = 0; b < total_batch_size; b++) {
                for (size_t oc = 0; oc < group_len; oc++) {
                    for (size_t i = 0; i < input_group_len; i++) {
                        a_ptrs.push_back(&a[b][i]);
                        w_ptrs.push_back(&encoded_weights[oc][i]);
                        r_ptrs.push_back(&ret[b][oc]);
                    }
                }
            }
            evaluator.multiply_plain_accumulate(a_ptrs, w_ptrs, r_ptrs, true, pool);
        }
        HeContextPointer context = evaluator.context();
        if (context->first_context_data().value()->parms().scheme() == SchemeType::BFV) {
            ensure_no_ntt_form(batched_mul, context, pool, ret);
        }
        return ret;
    }

    Cipher2d Conv2dHelper::conv2d_cipher(const Evaluator& evaluator, const Cipher2d& a, const Cipher2d& encoded_weights) const {
        size_t total_batch_size = get_total_batch_size();
        Cipher2d ret; ret.data().reserve(total_batch_size);
        for (size_t b = 0; b < total_batch_size; b++) {
            size_t groupLen = ceil_div(output_channels, output_channel_block);
            std::vector<Ciphertext> group; group.reserve(groupLen);
            for (size_t oc = 0; oc < groupLen; oc++) {
                Ciphertext cipher;
                for (size_t i = 0; i < a[b].size(); i++) {
                    Ciphertext prod;
                    evaluator.multiply(a[b][i], encoded_weights[oc][i], prod, pool);
                    if (i==0) cipher = std::move(prod);
                    else evaluator.add_inplace(cipher, prod, pool);
                }
                group.push_back(std::move(cipher));
            }
            ret.data().push_back(std::move(group));
        }
        return ret;
    }

    Cipher2d Conv2dHelper::conv2d_reverse(const Evaluator& evaluator, const Plain2d& a, const Cipher2d& encoded_weights) const {
        size_t total_batch_size = get_total_batch_size();
        Cipher2d ret; ret.data().reserve(total_batch_size);
        size_t group_len = ceil_div(output_channels, output_channel_block);
        size_t input_group_len = ceil_div(input_channels, input_channel_block);

        if (!batched_mul) {
            for (size_t b = 0; b < total_batch_size; b++) {
                std::vector<Ciphertext> group; group.reserve(group_len);
                for (size_t oc = 0; oc < group_len; oc++) {
                    Ciphertext cipher;
                    for (size_t i = 0; i < input_group_len; i++) {
                        Ciphertext prod;
                        evaluator.multiply_plain(encoded_weights[oc][i], a[b][i], prod, pool);
                        if (i==0) cipher = std::move(prod);
                        else evaluator.add_inplace(cipher, prod, pool);
                    }
                    group.push_back(std::move(cipher));
                }
                ret.data().push_back(std::move(group));
            }
        } else {
            using std::vector;
            vector<const Plaintext*> a_ptrs; a_ptrs.reserve(total_batch_size * group_len * input_group_len);
            vector<const Ciphertext*> w_ptrs; w_ptrs.reserve(total_batch_size * group_len * input_group_len);
            vector<Ciphertext*> r_ptrs; r_ptrs.reserve(total_batch_size * group_len * input_group_len);
            for (size_t b = 0; b < total_batch_size; b++) {
                ret.data().push_back(vector<Ciphertext>(group_len));
            }
            for (size_t b = 0; b < total_batch_size; b++) {
                for (size_t oc = 0; oc < group_len; oc++) {
                    for (size_t i = 0; i < input_group_len; i++) {
                        a_ptrs.push_back(&a[b][i]);
                        w_ptrs.push_back(&encoded_weights[oc][i]);
                        r_ptrs.push_back(&ret[b][oc]);
                    }
                }
            }
            evaluator.multiply_plain_accumulate(w_ptrs, a_ptrs, r_ptrs, true, pool);
        }
        HeContextPointer context = evaluator.context();
        if (context->first_context_data().value()->parms().scheme() == SchemeType::BFV) {
            ensure_no_ntt_form(batched_mul, context, pool, ret);
        }
        return ret;
    }

    void Conv2dHelper::serialize_outputs(const Evaluator &evaluator, const Cipher2d& x, std::ostream& stream, CompressionMode mode) const {
        auto total_batch_size = get_total_batch_size();
        size_t interval = image_width_block * image_height_block;
        
        size_t yh = image_height_block - kernel_height + 1;
        size_t yw = image_width_block  - kernel_width  + 1;

        std::vector<size_t> required;
        required.reserve(yh * yw * batch_block * output_channel_block);

        for (size_t b = 0; b < batch_block; b++) {
            for (size_t c = 0; c < output_channel_block; c++) {
                for (size_t i = 0; i < yh; i++) {
                    for (size_t j = 0; j < yw; j++) {
                        size_t mask_index = (b * input_channel_block * output_channel_block + c * input_channel_block + input_channel_block - 1) * interval + (image_height_block - yh + i) * image_width_block + (image_width_block - yw + j);
                        required.push_back(mask_index);
                    }
                }
            }
        }

        for (size_t b = 0; b < total_batch_size; b++) {
            for (size_t oc = 0; oc < ceil_div(output_channels, output_channel_block); oc++) 
                x[b][oc].save_terms(stream, evaluator.context(), required, pool, mode);
        }
    }

    Cipher2d Conv2dHelper::deserialize_outputs(const Evaluator &evaluator, std::istream& stream) const {
        auto total_batch_size = get_total_batch_size();
        size_t interval = image_width_block * image_height_block;
        
        size_t yh = image_height_block - kernel_height + 1;
        size_t yw = image_width_block  - kernel_width  + 1;

        std::vector<size_t> required;
        required.reserve(yh * yw * batch_block * output_channel_block);

        for (size_t b = 0; b < batch_block; b++) {
            for (size_t c = 0; c < output_channel_block; c++) {
                for (size_t i = 0; i < yh; i++) {
                    for (size_t j = 0; j < yw; j++) {
                        size_t mask_index = (b * input_channel_block * output_channel_block + c * input_channel_block + input_channel_block - 1) * interval + (image_height_block - yh + i) * image_width_block + (image_width_block - yw + j);
                        required.push_back(mask_index);
                    }
                }
            }
        }

        Cipher2d ret; ret.data().reserve(total_batch_size);
        for (size_t b = 0; b < total_batch_size; b++) {
            std::vector<Ciphertext> row(output_channels);
            for (size_t oc = 0; oc < ceil_div(output_channels, output_channel_block); oc++) 
                row[oc].load_terms(stream, evaluator.context(), required, pool);
            ret.data().push_back(std::move(row));
        }
        return ret;
    }

    #undef D_IMPL_ALL
    

    // ------------------ adapters -----------------------


    Plain2d Conv2dHelper::encode_weights_uint64s(const BatchEncoder& encoder, const uint64_t* weights) const {
        BatchEncoderAdapter adapter(encoder); Plain2d ret;
        encode_weights(adapter, nullptr, weights, false, &ret, nullptr);
        return ret;
    }
    Plain2d Conv2dHelper::encode_weights_doubles(const CKKSEncoder& encoder, const double* weights, std::optional<ParmsID> parms_id, double scale) const {
        CKKSEncoderAdapter adapter(encoder, parms_id, scale); Plain2d ret;
        encode_weights(adapter, nullptr, weights, false, &ret, nullptr);
        return ret;
    }
    template <typename T>
    Plain2d Conv2dHelper::encode_weights_ring2k(const PolynomialEncoderRing2k<T>& encoder, const T* weights, std::optional<ParmsID> parms_id) const {
        PolynomialEncoderRing2kAdapter<T> adapter(encoder, parms_id); Plain2d ret;
        encode_weights(adapter, nullptr, weights, false, &ret, nullptr);
        return ret;
    }
    template Plain2d Conv2dHelper::encode_weights_ring2k<uint32_t>(
        const PolynomialEncoderRing2k<uint32_t>& encoder, const uint32_t* weights, std::optional<ParmsID> parms_id
    ) const;
    template Plain2d Conv2dHelper::encode_weights_ring2k<uint64_t>(
        const PolynomialEncoderRing2k<uint64_t>& encoder, const uint64_t* weights, std::optional<ParmsID> parms_id
    ) const;
    template Plain2d Conv2dHelper::encode_weights_ring2k<uint128_t>(
        const PolynomialEncoderRing2k<uint128_t>& encoder, const uint128_t* weights, std::optional<ParmsID> parms_id
    ) const;

    Plain2d Conv2dHelper::encode_inputs_uint64s(const BatchEncoder& encoder, const uint64_t* inputs) const {
        BatchEncoderAdapter adapter(encoder); Plain2d ret;
        encode_inputs(adapter, nullptr, inputs, false, &ret, nullptr);
        return ret;
    }
    Plain2d Conv2dHelper::encode_inputs_doubles(const CKKSEncoder& encoder, const double* inputs, std::optional<ParmsID> parms_id, double scale) const {
        CKKSEncoderAdapter adapter(encoder, parms_id, scale); Plain2d ret;
        encode_inputs(adapter, nullptr, inputs, false, &ret, nullptr);
        return ret;
    }
    template <typename T>
    Plain2d Conv2dHelper::encode_inputs_ring2k(const PolynomialEncoderRing2k<T>& encoder, const T* inputs, std::optional<ParmsID> parms_id) const {
        PolynomialEncoderRing2kAdapter<T> adapter(encoder, parms_id); Plain2d ret;
        encode_inputs(adapter, nullptr, inputs, false, &ret, nullptr);
        return ret;
    }
    template Plain2d Conv2dHelper::encode_inputs_ring2k<uint32_t>(
        const PolynomialEncoderRing2k<uint32_t>& encoder, const uint32_t* inputs, std::optional<ParmsID> parms_id
    ) const;
    template Plain2d Conv2dHelper::encode_inputs_ring2k<uint64_t>(
        const PolynomialEncoderRing2k<uint64_t>& encoder, const uint64_t* inputs, std::optional<ParmsID> parms_id
    ) const;
    template Plain2d Conv2dHelper::encode_inputs_ring2k<uint128_t>(
        const PolynomialEncoderRing2k<uint128_t>& encoder, const uint128_t* inputs, std::optional<ParmsID> parms_id
    ) const;

    Cipher2d Conv2dHelper::encrypt_inputs_uint64s(const Encryptor& encryptor, const BatchEncoder& encoder, const uint64_t* inputs) const {
        BatchEncoderAdapter adapter(encoder); Cipher2d ret;
        encode_inputs(adapter, &encryptor, inputs, true, nullptr, &ret);
        return ret;
    }
    Cipher2d Conv2dHelper::encrypt_inputs_doubles(const Encryptor& encryptor, const CKKSEncoder& encoder, const double* inputs, std::optional<ParmsID> parms_id, double scale) const {
        CKKSEncoderAdapter adapter(encoder, parms_id, scale); Cipher2d ret;
        encode_inputs(adapter, &encryptor, inputs, true, nullptr, &ret);
        return ret;
    }
    template <typename T>
    Cipher2d Conv2dHelper::encrypt_inputs_ring2k(const Encryptor& encryptor, const PolynomialEncoderRing2k<T>& encoder, const T* inputs, std::optional<ParmsID> parms_id) const {
        PolynomialEncoderRing2kAdapter<T> adapter(encoder, parms_id); Cipher2d ret;
        encode_inputs(adapter, &encryptor, inputs, true, nullptr, &ret);
        return ret;
    }
    template Cipher2d Conv2dHelper::encrypt_inputs_ring2k<uint32_t>(
        const Encryptor& encryptor, const PolynomialEncoderRing2k<uint32_t>& encoder, const uint32_t* inputs, std::optional<ParmsID> parms_id
    ) const;
    template Cipher2d Conv2dHelper::encrypt_inputs_ring2k<uint64_t>(
        const Encryptor& encryptor, const PolynomialEncoderRing2k<uint64_t>& encoder, const uint64_t* inputs, std::optional<ParmsID> parms_id
    ) const;
    template Cipher2d Conv2dHelper::encrypt_inputs_ring2k<uint128_t>(
        const Encryptor& encryptor, const PolynomialEncoderRing2k<uint128_t>& encoder, const uint128_t* inputs, std::optional<ParmsID> parms_id
    ) const;


    Cipher2d Conv2dHelper::encrypt_weights_uint64s(const Encryptor& encryptor, const BatchEncoder& encoder, const uint64_t* weights) const {
        BatchEncoderAdapter adapter(encoder); Cipher2d ret;
        encode_weights(adapter, &encryptor, weights, true, nullptr, &ret);
        return ret;
    }
    Cipher2d Conv2dHelper::encrypt_weights_doubles(const Encryptor& encryptor, const CKKSEncoder& encoder, const double* weights, std::optional<ParmsID> parms_id, double scale) const {
        CKKSEncoderAdapter adapter(encoder, parms_id, scale); Cipher2d ret;
        encode_weights(adapter, &encryptor, weights, true, nullptr, &ret);
        return ret;
    }
    template <typename T>
    Cipher2d Conv2dHelper::encrypt_weights_ring2k(const Encryptor& encryptor, const PolynomialEncoderRing2k<T>& encoder, const T* inputs, std::optional<ParmsID> parms_id) const {
        PolynomialEncoderRing2kAdapter<T> adapter(encoder, parms_id); Cipher2d ret;
        encode_weights(adapter, &encryptor, inputs, true, nullptr, &ret);
        return ret;
    }
    template Cipher2d Conv2dHelper::encrypt_weights_ring2k<uint32_t>(
        const Encryptor& encryptor, const PolynomialEncoderRing2k<uint32_t>& encoder, const uint32_t* weights, std::optional<ParmsID> parms_id
    ) const;
    template Cipher2d Conv2dHelper::encrypt_weights_ring2k<uint64_t>(
        const Encryptor& encryptor, const PolynomialEncoderRing2k<uint64_t>& encoder, const uint64_t* weights, std::optional<ParmsID> parms_id
    ) const;
    template Cipher2d Conv2dHelper::encrypt_weights_ring2k<uint128_t>(
        const Encryptor& encryptor, const PolynomialEncoderRing2k<uint128_t>& encoder, const uint128_t* weights, std::optional<ParmsID> parms_id
    ) const;

    Plain2d Conv2dHelper::encode_outputs_uint64s(const BatchEncoder& encoder, const uint64_t* outputs) const {
        BatchEncoderAdapter adapter(encoder);
        return encode_outputs(adapter, outputs);
    }
    Plain2d Conv2dHelper::encode_outputs_doubles(const CKKSEncoder& encoder, const double* outputs, std::optional<ParmsID> parms_id, double scale) const {
        CKKSEncoderAdapter adapter(encoder, parms_id, scale);
        return encode_outputs(adapter, outputs);
    }
    template <typename T>
    Plain2d Conv2dHelper::encode_outputs_ring2k(const PolynomialEncoderRing2k<T>& encoder, const T* outputs, std::optional<ParmsID> parms_id) const {
        PolynomialEncoderRing2kAdapter<T> adapter(encoder, parms_id);
        return encode_outputs(adapter, outputs);
    }
    template Plain2d Conv2dHelper::encode_outputs_ring2k<uint32_t>(
        const PolynomialEncoderRing2k<uint32_t>& encoder, const uint32_t* outputs, std::optional<ParmsID> parms_id
    ) const;
    template Plain2d Conv2dHelper::encode_outputs_ring2k<uint64_t>(
        const PolynomialEncoderRing2k<uint64_t>& encoder, const uint64_t* outputs, std::optional<ParmsID> parms_id
    ) const;
    template Plain2d Conv2dHelper::encode_outputs_ring2k<uint128_t>(
        const PolynomialEncoderRing2k<uint128_t>& encoder, const uint128_t* outputs, std::optional<ParmsID> parms_id
    ) const;

    std::vector<uint64_t> Conv2dHelper::decrypt_outputs_uint64s(const BatchEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const {
        BatchEncoderAdapter adapter(encoder);
        return decrypt_outputs<BatchEncoderAdapter, uint64_t>(adapter, decryptor, outputs);
    }
    std::vector<double> Conv2dHelper::decrypt_outputs_doubles(const CKKSEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const {
        CKKSEncoderAdapter adapter(encoder, std::nullopt, 0);
        return decrypt_outputs<CKKSEncoderAdapter, double>(adapter, decryptor, outputs);
    }
    template <typename T>
    std::vector<T> Conv2dHelper::decrypt_outputs_ring2k(const PolynomialEncoderRing2k<T>& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const {
        PolynomialEncoderRing2kAdapter<T> adapter(encoder, std::nullopt);
        return decrypt_outputs<PolynomialEncoderRing2kAdapter<T>, T>(adapter, decryptor, outputs);
    }
    template std::vector<uint32_t> Conv2dHelper::decrypt_outputs_ring2k<uint32_t>(
        const PolynomialEncoderRing2k<uint32_t>& encoder, const Decryptor& decryptor, const Cipher2d& outputs
    ) const;
    template std::vector<uint64_t> Conv2dHelper::decrypt_outputs_ring2k<uint64_t>(
        const PolynomialEncoderRing2k<uint64_t>& encoder, const Decryptor& decryptor, const Cipher2d& outputs
    ) const;
    template std::vector<uint128_t> Conv2dHelper::decrypt_outputs_ring2k<uint128_t>(
        const PolynomialEncoderRing2k<uint128_t>& encoder, const Decryptor& decryptor, const Cipher2d& outputs
    ) const;

}}