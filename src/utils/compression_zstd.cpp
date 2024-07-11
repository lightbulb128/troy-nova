#include "compression.h"

#ifdef TROY_ZSTD

#include <zstd.h>

namespace troy::utils::compression::zstd {


    bool available() {
        return true;
    }

    void check_zstd(size_t error_code, const char* message) {
        if (ZSTD_isError(error_code)) {
            throw std::runtime_error(std::string(message) + message + " - " + ZSTD_getErrorName(error_code));
        }
    }
    
    size_t compressed_size_upperbound(size_t input_size) {
        return ZSTD_compressBound(input_size);
    }

    size_t compress(const void* input, size_t input_size, std::ostream& output) {
        if (input_size > ZSTD_MAX_INPUT_SIZE) {
            throw std::invalid_argument("[utils::compression::zstd::compress] Input size is too large");
        }
        size_t buffer_size = std::min(input_size, ZSTD_CStreamOutSize());
        uint8_t* buffer = new uint8_t[buffer_size];
        size_t offset = 0;
        size_t total_written = 0;

        // create context
        ZSTD_CCtx* cctx = ZSTD_createCCtx();
        if (cctx == nullptr) {
            throw std::runtime_error("[utils::compression::zstd::compress] ZSTD_createCCtx() failed");
        }

        // set parameters
        size_t error = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, ZSTD_CLEVEL_DEFAULT);
        check_zstd(error, "[utils::compression::zstd::compress] ZSTD_CCtx_setParameter() failed");

        size_t input_chunk_size = ZSTD_CStreamInSize();
        bool finished = false;
        while (offset < input_size) {
            bool last_chunk = offset + input_chunk_size >= input_size;
            ZSTD_EndDirective mode = last_chunk ? ZSTD_e_end : ZSTD_e_continue;
            size_t input_process_chunk_size = last_chunk ? input_size - offset : input_chunk_size;
            ZSTD_inBuffer input_buffer = { reinterpret_cast<const uint8_t*>(input) + offset, input_process_chunk_size, 0 };
            do {
                ZSTD_outBuffer output_buffer = { buffer, buffer_size, 0 };
                size_t remaining = ZSTD_compressStream2(cctx, &output_buffer, &input_buffer, mode);
                check_zstd(remaining, "[utils::compression::zstd::compress] ZSTD_compressStream() failed");
                output.write(reinterpret_cast<const char*>(buffer), output_buffer.pos);
                total_written += output_buffer.pos;
                finished = last_chunk ? (remaining == 0) : (input_buffer.pos == input_buffer.size);
            } while (!finished);
            offset += input_process_chunk_size;
        }

        delete[] buffer;

        // free context
        ZSTD_freeCCtx(cctx);

        return total_written;
    }

    size_t decompress(const void* input, size_t input_size, std::ostream& output) {
        size_t buffer_size = ZSTD_DStreamOutSize();
        uint8_t* buffer = new uint8_t[buffer_size];
        size_t offset = 0;
        size_t total_written = 0;

        // create context
        ZSTD_DCtx* dctx = ZSTD_createDCtx();
        if (dctx == nullptr) {
            throw std::runtime_error("[utils::compression::zstd::decompress] ZSTD_createDCtx() failed");
        }

        size_t input_chunk_size = ZSTD_DStreamInSize();
        bool finished = false;
        while (offset < input_size) {
            size_t input_process_chunk_size = std::min(input_chunk_size, input_size - offset);
            ZSTD_inBuffer input_buffer = { reinterpret_cast<const uint8_t*>(input) + offset, input_process_chunk_size, 0 };
            do {
                ZSTD_outBuffer output_buffer = { buffer, buffer_size, 0 };
                size_t remaining = ZSTD_decompressStream(dctx, &output_buffer, &input_buffer);
                check_zstd(remaining, "[utils::compression::zstd::decompress] ZSTD_decompressStream() failed");
                output.write(reinterpret_cast<const char*>(buffer), output_buffer.pos);
                total_written += output_buffer.pos;
                finished = input_buffer.pos == input_buffer.size;
            } while (!finished);
            offset += input_process_chunk_size;
        }

        delete[] buffer;

        // free context
        ZSTD_freeDCtx(dctx);

        return total_written;
    }
    
}

#else

namespace troy::utils::compression::zstd {

    void throw_it() {
        throw std::runtime_error("[troy::utils::compression::zstd] Zstd is not enabled");
    }

    size_t compressed_size_upperbound(size_t input_size) {
        throw_it();
    }
    size_t compress(const void* input, size_t input_size, std::ostream& output) {
        throw_it();
    }
    size_t decompress(const void* input, size_t input_size, std::ostream& output) {
        throw_it();
    }

    bool available() {
        return false;
    }

}

#endif