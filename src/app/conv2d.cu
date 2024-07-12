#include "conv2d.h"

namespace troy { namespace linear {

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

    Plain2d Conv2dHelper::encode_weights(
        const BatchEncoder& encoder, 
        const uint64_t* weights
    ) const {
        size_t block_size = image_height_block * image_width_block;
        Plain2d encoded_weights;
        encoded_weights.data().clear();
        encoded_weights.data().reserve(ceil_div(output_channels, output_channel_block));
        for (size_t loc = 0; loc < output_channels; loc += output_channel_block) {
            size_t uoc = std::min(loc + output_channel_block, output_channels);
            std::vector<Plaintext> current_channel;
            current_channel.reserve(ceil_div(input_channels, input_channel_block));
            for (size_t lic = 0; lic < input_channels; lic += input_channel_block) {
                size_t uic = std::min(lic + input_channel_block, input_channels);
                std::vector<uint64_t> spread(input_channel_block * output_channel_block * image_height_block * image_width_block, 0);
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
                Plaintext pt; encoder.encode_polynomial(spread, pt, pool);
                current_channel.push_back(std::move(pt));
            }
            encoded_weights.data().push_back(std::move(current_channel));
        }
        return encoded_weights;
    }

    size_t Conv2dHelper::get_total_batch_size() const {
        size_t kh = kernel_height - 1, kw = kernel_width - 1;
        size_t sh = ceil_div(image_height - kh, image_height_block - kh);
        size_t sw = ceil_div(image_width - kw, image_width_block - kw);
        return ceil_div(batch_size, batch_block) * sh * sw;
    }

    Plain2d Conv2dHelper::encode_inputs(
        const BatchEncoder& encoder, 
        const uint64_t* inputs
    ) const {
        size_t kh = kernel_height - 1, kw = kernel_width - 1;
        size_t sh = ceil_div(image_height - kh, image_height_block - kh);
        size_t sw = ceil_div(image_width - kw, image_width_block - kw);
        size_t imageSize = image_height * image_width;
        size_t block_size = image_height_block * image_width_block;
        size_t totalBatch_size = ceil_div(batch_size, batch_block) * sh * sw;
        Plain2d ret; ret.data().reserve(totalBatch_size);
        for (size_t lb = 0; lb < batch_size; lb += batch_block) {
            size_t ub = std::min(lb + batch_block, batch_size);
            for (size_t ih = 0; ih < sh; ih++) {
                for (size_t iw = 0; iw < sw; iw++) {
                    size_t si = ih * (image_height_block - kh);
                    size_t sj = iw * (image_width_block - kw);
                    size_t ui = std::min(si + image_height_block, image_height);
                    size_t uj = std::min(sj + image_width_block, image_width);
                    std::vector<Plaintext> group; group.reserve(ceil_div(input_channels, input_channel_block));
                    for (size_t lci = 0; lci < input_channels; lci += input_channel_block) {
                        size_t uci = std::min(lci + input_channel_block, input_channels);
                        std::vector<uint64_t> vec(slot_count, 0);
                        for (size_t b = 0; b < ub-lb; b++) {
                            for (size_t tci = 0; tci < uci-lci; tci++) {
                                for (size_t ti = si; ti < ui; ti++) {
                                    for (size_t tj = sj; tj < uj; tj++) {
                                        size_t inputIndex = (lb + b) * input_channels * imageSize + (lci + tci) * imageSize + ti * image_width + tj;
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
                        Plaintext pt; encoder.encode_polynomial(vec, pt, pool);
                        // printf("encode ok\n");
                        group.push_back(std::move(pt));
                    }
                    ret.data().push_back(std::move(group));
                }
            }
        }
        return ret;
    }

    Cipher2d Conv2dHelper::encrypt_inputs(
        const Encryptor& encryptor,
        const BatchEncoder& encoder, 
        const uint64_t* inputs
    ) const {
        Plain2d plain = encode_inputs(encoder, inputs);
        return plain.encrypt_symmetric(encryptor, pool);
    }

    Cipher2d Conv2dHelper::conv2d(const Evaluator& evaluator, const Cipher2d& a, const Plain2d& encoded_weights) const {

        // Timer tim; auto t1 = tim.registerTimer("muladds");
        // size_t muladds = 0;

        size_t totalBatch_size = get_total_batch_size();
        Cipher2d ret; ret.data().reserve(totalBatch_size);
        for (size_t b = 0; b < totalBatch_size; b++) {
            size_t groupLen = ceil_div(output_channels, output_channel_block);
            std::vector<Ciphertext> group; group.reserve(groupLen);
            for (size_t oc = 0; oc < groupLen; oc++) {
                Ciphertext cipher;
                for (size_t i = 0; i < a[b].size(); i++) {
                    Ciphertext prod;
                    // tim.tick(t1);
                    evaluator.multiply_plain(a[b][i], encoded_weights[oc][i], prod, pool);
                    // muladds ++;
                    // tim.tock(t1);
                    if (i==0) cipher = std::move(prod);
                    else evaluator.add_inplace(cipher, prod, pool);
                }
                group.push_back(std::move(cipher));
            }
            ret.data().push_back(std::move(group));
        }
        // printTimer(tim.gather(muladds));
        return ret;
    }

    Cipher2d Conv2dHelper::conv2d_cipher(const Evaluator& evaluator, const Cipher2d& a, const Cipher2d& encoded_weights) const {
        size_t totalBatch_size = get_total_batch_size();
        Cipher2d ret; ret.data().reserve(totalBatch_size);
        for (size_t b = 0; b < totalBatch_size; b++) {
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
        size_t totalBatch_size = get_total_batch_size();
        Cipher2d ret; ret.data().reserve(totalBatch_size);
        for (size_t b = 0; b < totalBatch_size; b++) {
            size_t groupLen = ceil_div(output_channels, output_channel_block);
            std::vector<Ciphertext> group; group.reserve(groupLen);
            for (size_t oc = 0; oc < groupLen; oc++) {
                Ciphertext cipher;
                for (size_t i = 0; i < a[b].size(); i++) {
                    Ciphertext prod;
                    evaluator.multiply_plain(encoded_weights[oc][i], a[b][i], prod, pool);
                    if (i==0) cipher = std::move(prod);
                    else evaluator.add_inplace(cipher, prod, pool);
                }
                group.push_back(std::move(cipher));
            }
            ret.data().push_back(std::move(group));
        }
        return ret;
    }

    Plain2d Conv2dHelper::encode_outputs(
        const BatchEncoder& encoder,
        const uint64_t* outputs
    ) const {
        size_t interval = image_width_block * image_height_block;
        std::vector<uint64_t> mask(slot_count, 0);
        auto totalBatch_size = get_total_batch_size();
        size_t yh = image_height_block - kernel_height + 1;
        size_t yw = image_width_block  - kernel_width  + 1;
        size_t oyh = image_height - kernel_height + 1;
        size_t oyw = image_width - kernel_width + 1;
        Plain2d ret; ret.data().reserve(totalBatch_size);
        size_t kh = kernel_height - 1, kw = kernel_width - 1;
        size_t sh = ceil_div(image_height - kh, image_height_block - kh);
        size_t sw = ceil_div(image_width - kw, image_width_block - kw);
        assert(totalBatch_size == ceil_div(batch_size, batch_block) * sh * sw);
        Plaintext encoded;
        std::vector<uint64_t> buffer;
        for (size_t eb = 0; eb < totalBatch_size; eb++) {
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
                                size_t originalIndex = b * output_channels * oyh * oyw + c * oyh * oyw + (si * yh + i) * oyw + (sj * yw + j);
                                if (si * yh + i < oyh && sj * yw + j < oyw)  mask[mask_index] = outputs[originalIndex];
                            }
                        }
                    }
                }
                Plaintext encoded; encoder.encode_polynomial(mask, encoded, pool);
                group.push_back(std::move(encoded));
            }
            ret.data().push_back(std::move(group));
        }
        return ret;
    }

    std::vector<uint64_t> Conv2dHelper::decrypt_outputs(
        const BatchEncoder& encoder,
        const Decryptor& decryptor,
        const Cipher2d& outputs
    ) const {
        size_t interval = image_width_block * image_height_block;
        auto totalBatch_size = get_total_batch_size();
        size_t yh = image_height_block - kernel_height + 1;
        size_t yw = image_width_block  - kernel_width  + 1;
        size_t oyh = image_height - kernel_height + 1;
        size_t oyw = image_width - kernel_width + 1;
        std::vector<uint64_t> ret(batch_size * output_channels * oyh * oyw, 0);
        size_t kh = kernel_height - 1, kw = kernel_width - 1;
        size_t sh = ceil_div(image_height - kh, image_height_block - kh);
        size_t sw = ceil_div(image_width - kw, image_width_block - kw);
        assert(totalBatch_size == ceil_div(batch_size, batch_block) * sh * sw);
        Plaintext encoded;
        std::vector<uint64_t> buffer;
        for (size_t eb = 0; eb < totalBatch_size; eb++) {
            size_t ob = eb / (sh * sw);
            size_t si = (eb % (sh * sw)) / sw;
            size_t sj = eb % sw;
            size_t lb = ob * batch_block, ub = std::min(lb + batch_block, batch_size);
            for (size_t lc = 0; lc < output_channels; lc += output_channel_block) {
                size_t uc = std::min(lc + output_channel_block, output_channels);
                // printf("Decrypting block [%lu][%lu]\n", eb, lc / output_channel_block);
                decryptor.decrypt(outputs[eb][lc / output_channel_block], encoded, pool);
                encoder.decode_polynomial(encoded, buffer);
                for (size_t b = lb; b < ub; b++) {
                    for (size_t c = lc; c < uc; c++) {
                        for (size_t i = 0; i < yh; i++) {
                            for (size_t j = 0; j < yw; j++) {
                                size_t mask_index = ((b - lb) * input_channel_block * output_channel_block + (c - lc) * input_channel_block + input_channel_block - 1) * interval + (image_height_block - yh + i) * image_width_block + (image_width_block - yw + j);
                                size_t originalIndex = b * output_channels * oyh * oyw + c * oyh * oyw + (si * yh + i) * oyw + (sj * yw + j);
                                // printf("Original[%lu][%lu][%lu][%lu] <- idx[%lu]\n", b, c, si * yh + i, sj * yw + j, mask_index);
                                if (si * yh + i < oyh && sj * yw + j < oyw) {
                                    ret[originalIndex] = buffer[mask_index];
                                }
                            }
                        }
                    }
                }
            }
        }
        return ret;
    }

    void Conv2dHelper::serialize_outputs(const Evaluator &evaluator, const Cipher2d& x, std::ostream& stream) const {
        auto totalBatch_size = get_total_batch_size();
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

        for (size_t b = 0; b < totalBatch_size; b++) {
            for (size_t oc = 0; oc < ceil_div(output_channels, output_channel_block); oc++) 
                x[b][oc].save_terms(stream, evaluator.context(), required);
        }
    }

    Cipher2d Conv2dHelper::deserialize_outputs(const Evaluator &evaluator, std::istream& stream) const {
        auto totalBatch_size = get_total_batch_size();
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

        Cipher2d ret; ret.data().reserve(totalBatch_size);
        for (size_t b = 0; b < totalBatch_size; b++) {
            std::vector<Ciphertext> row(output_channels);
            for (size_t oc = 0; oc < ceil_div(output_channels, output_channel_block); oc++) 
                row[oc].load_terms(stream, evaluator.context(), required, pool);
            ret.data().push_back(std::move(row));
        }
        return ret;
    }
    
}}