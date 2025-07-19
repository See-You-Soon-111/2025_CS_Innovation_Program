#include<stdio.h>
#include<stdint.h>
#include<intrin.h>
#include <xmmintrin.h>
#include<chrono>

using namespace std;

// 添加CPU周期计数函数
#if defined(_MSC_VER)

uint64_t rdtsc() {
    return __rdtsc();
}
#elif defined(__GNUC__)
uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}
#endif

uint32_t FK[4] = { 0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc };
uint32_t CK[32] = {
   0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269, 0x70777e85, 0x8c939aa1,
   0xa8afb6bd, 0xc4cbd2d9, 0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249,
   0x50575e65, 0x6c737a81, 0x888f969d, 0xa4abb2b9, 0xc0c7ced5, 0xdce3eaf1,
   0xf8ff060d, 0x141b2229, 0x30373e45, 0x4c535a61, 0x686f767d, 0x848b9299,
   0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209, 0x10171e25, 0x2c333a41,
   0x484f565d, 0x646b7279 };
uint8_t Sbox[256] = {
   0xD6, 0x90, 0xE9, 0xFE, 0xCC, 0xE1, 0x3D, 0xB7, 0x16, 0xB6, 0x14, 0xC2,
   0x28, 0xFB, 0x2C, 0x05, 0x2B, 0x67, 0x9A, 0x76, 0x2A, 0xBE, 0x04, 0xC3,
   0xAA, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99, 0x9C, 0x42, 0x50, 0xF4,
   0x91, 0xEF, 0x98, 0x7A, 0x33, 0x54, 0x0B, 0x43, 0xED, 0xCF, 0xAC, 0x62,
   0xE4, 0xB3, 0x1C, 0xA9, 0xC9, 0x08, 0xE8, 0x95, 0x80, 0xDF, 0x94, 0xFA,
   0x75, 0x8F, 0x3F, 0xA6, 0x47, 0x07, 0xA7, 0xFC, 0xF3, 0x73, 0x17, 0xBA,
   0x83, 0x59, 0x3C, 0x19, 0xE6, 0x85, 0x4F, 0xA8, 0x68, 0x6B, 0x81, 0xB2,
   0x71, 0x64, 0xDA, 0x8B, 0xF8, 0xEB, 0x0F, 0x4B, 0x70, 0x56, 0x9D, 0x35,
   0x1E, 0x24, 0x0E, 0x5E, 0x63, 0x58, 0xD1, 0xA2, 0x25, 0x22, 0x7C, 0x3B,
   0x01, 0x21, 0x78, 0x87, 0xD4, 0x00, 0x46, 0x57, 0x9F, 0xD3, 0x27, 0x52,
   0x4C, 0x36, 0x02, 0xE7, 0xA0, 0xC4, 0xC8, 0x9E, 0xEA, 0xBF, 0x8A, 0xD2,
   0x40, 0xC7, 0x38, 0xB5, 0xA3, 0xF7, 0xF2, 0xCE, 0xF9, 0x61, 0x15, 0xA1,
   0xE0, 0xAE, 0x5D, 0xA4, 0x9B, 0x34, 0x1A, 0x55, 0xAD, 0x93, 0x32, 0x30,
   0xF5, 0x8C, 0xB1, 0xE3, 0x1D, 0xF6, 0xE2, 0x2E, 0x82, 0x66, 0xCA, 0x60,
   0xC0, 0x29, 0x23, 0xAB, 0x0D, 0x53, 0x4E, 0x6F, 0xD5, 0xDB, 0x37, 0x45,
   0xDE, 0xFD, 0x8E, 0x2F, 0x03, 0xFF, 0x6A, 0x72, 0x6D, 0x6C, 0x5B, 0x51,
   0x8D, 0x1B, 0xAF, 0x92, 0xBB, 0xDD, 0xBC, 0x7F, 0x11, 0xD9, 0x5C, 0x41,
   0x1F, 0x10, 0x5A, 0xD8, 0x0A, 0xC1, 0x31, 0x88, 0xA5, 0xCD, 0x7B, 0xBD,
   0x2D, 0x74, 0xD0, 0x12, 0xB8, 0xE5, 0xB4, 0xB0, 0x89, 0x69, 0x97, 0x4A,
   0x0C, 0x96, 0x77, 0x7E, 0x65, 0xB9, 0xF1, 0x09, 0xC5, 0x6E, 0xC6, 0x84,
   0x18, 0xF0, 0x7D, 0xEC, 0x3A, 0xDC, 0x4D, 0x20, 0x79, 0xEE, 0x5F, 0x3E,
   0xD7, 0xCB, 0x39, 0x48 };

#define rotl32(value,shift) ((value<<shift)|value>>(32-shift))

//初始密钥
#define LOAD_KEY(index)\
	do{\
		k[index]=(key[index<<2]<<24)|(key[(index<<2)+1]<<16)|(key[(index<<2)+2]<<8)|(key[(index<<2)+3]);\
		k[index]=k[index]^FK[index];\
	}while(0)

//一轮轮密钥生成
#define KEY_GEN_1R(index)\
	do{\
		temp=k[1]^k[2]^k[3]^CK[index];\
		temp=(Sbox[temp>>24]<<24)|(Sbox[(temp>>16)&0xff]<<16)|(Sbox[(temp>>8)&0xff]<<8)|(Sbox[temp&0xff]);		\
		rk[index]=k[0]^temp^rotl32(temp,13)^rotl32(temp,23);\
		k[0]=k[1];\
		k[1]=k[2];\
		k[2]=k[3];\
		k[3]=rk[index];\
	}while(0)


void SM4_KEY_GEN(uint8_t* key, uint32_t* rk) {
    uint32_t k[4];
    uint32_t temp;
    LOAD_KEY(0);
    LOAD_KEY(1);
    LOAD_KEY(2);
    LOAD_KEY(3);
    for (int i = 0; i < 32; i++) {
        KEY_GEN_1R(i);
    }
}

//一轮SM4加密
void SM4_1R(int index, uint32_t* A, uint32_t* rk, bool is_enc, uint32_t temp) {
    uint32_t k = (is_enc == 1 ? rk[index] : rk[31 - index]);
    temp = A[1] ^ A[2] ^ A[3] ^ k;
    temp = (Sbox[temp >> 24] << 24) | (Sbox[(temp >> 16) & 0xff] << 16) | (Sbox[(temp >> 8) & 0xff] << 8) | (Sbox[temp & 0xff]);
    temp = A[0] ^ temp ^ rotl32(temp, 2) ^ rotl32(temp, 10) ^ rotl32(temp, 18) ^ rotl32(temp, 24);
    A[0] = A[1];
    A[1] = A[2];
    A[2] = A[3];
    A[3] = temp;
}

void SM4(uint8_t* input, uint8_t* output, uint32_t* rk, bool is_enc) {
    uint32_t A[4];
    uint32_t temp = 0;
    A[0] = ((uint32_t)input[0] << 24) | ((uint32_t)input[1] << 16) | ((uint32_t)input[2] << 8) | input[3];
    A[1] = ((uint32_t)input[4] << 24) | ((uint32_t)input[5] << 16) | ((uint32_t)input[6] << 8) | input[7];
    A[2] = ((uint32_t)input[8] << 24) | ((uint32_t)input[9] << 16) | ((uint32_t)input[10] << 8) | input[11];
    A[3] = ((uint32_t)input[12] << 24) | ((uint32_t)input[13] << 16) | ((uint32_t)input[14] << 8) | input[15];
    for (int i = 0; i < 32; i++) {
        SM4_1R(i, A, rk, is_enc, temp);
    }
    uint32_t tmp = A[0];
    A[0] = A[3];
    A[3] = tmp;
    tmp = A[1];
    A[1] = A[2];
    A[2] = tmp;

    output[0] = (A[0] >> 24) & 0xFF; output[1] = (A[0] >> 16) & 0xFF;
    output[2] = (A[0] >> 8) & 0xFF; output[3] = A[0] & 0xFF;
    output[4] = (A[1] >> 24) & 0xFF; output[5] = (A[1] >> 16) & 0xFF;
    output[6] = (A[1] >> 8) & 0xFF; output[7] = A[1] & 0xFF;
    output[8] = (A[2] >> 24) & 0xFF; output[9] = (A[2] >> 16) & 0xFF;
    output[10] = (A[2] >> 8) & 0xFF; output[11] = A[2] & 0xFF;
    output[12] = (A[3] >> 24) & 0xFF; output[13] = (A[3] >> 16) & 0xFF;
    output[14] = (A[3] >> 8) & 0xFF; output[15] = A[3] & 0xFF;
}



void gmul(const uint8_t* x, const uint8_t* y, uint8_t* z) {
    uint8_t v[16];
    uint8_t r[16];
    int i, j;

    memcpy(v, y, 16);
    memset(z, 0, 16);
    memset(r, 0, 16);
    r[0] = 0xe1; // 约化多项式(0x87左移1位)

    for (i = 0; i < 16; i++) {
        uint8_t x_byte = x[i];
        for (j = 0; j < 8; j++) {
            if (x_byte & (1 << (7 - j))) {
                for (int k = 0; k < 16; k++) {
                    z[k] ^= v[k];
                }
            }

            uint8_t carry = v[15] & 1;
            for (int k = 15; k > 0; k--) {
                v[k] = (v[k] >> 1) | ((v[k - 1] & 1) << 7);
            }
            v[0] >>= 1;

            if (carry) {
                for (int k = 0; k < 16; k++) {
                    v[k] ^= r[k];
                }
            }
        }
    }
}

// GHASH函数
void ghash(const uint8_t* H, const uint8_t* data, size_t data_len, uint8_t* out_tag) {
    uint8_t X[16] = { 0 }; // 初始为0

    for (size_t i = 0; i < data_len; i += 16) {
        uint8_t block[16] = { 0 };
        size_t block_len = (data_len - i) < 16 ? (data_len - i) : 16;
        memcpy(block, data + i, block_len);

        // X ^= block
        for (int j = 0; j < 16; j++) {
            X[j] ^= block[j];
        }

        // X = X * H
        gmul(X, H, X);
    }

    memcpy(out_tag, X, 16);
}

// 128位计数器递增(大端序)
void increment_counter(uint8_t* counter) {
    for (int i = 15; i >= 0; i--) {
        counter[i]++;
        if (counter[i] != 0) break;
    }
}

// SM4-GCM加密
void sm4_gcm_encrypt(const uint8_t* plaintext, size_t plaintext_len,
    const uint8_t* key, const uint8_t* iv, size_t iv_len,
    const uint8_t* aad, size_t aad_len,
    uint8_t* ciphertext, uint8_t* tag, size_t tag_len) {
    uint32_t rk[32];
    SM4_KEY_GEN((uint8_t*)key, rk);

    uint8_t H[16] = { 0 };
    uint8_t J0[16] = { 0 };
    uint8_t counter[16] = { 0 };
    uint8_t E0[16] = { 0 };
    uint8_t auth_tag[16] = { 0 };

    // 步骤1: 生成哈希子密钥 H = SM4(0^128)
    SM4(H, H, rk, 1);

    // 步骤2: 生成J0(初始计数器块)
    if (iv_len == 12) {
        memcpy(J0, iv, 12);
        J0[15] = 1;
    }

    memcpy(counter, J0, 16);
    increment_counter(counter);

    // 步骤3: 计数器模式加密明文
    for (size_t i = 0; i < plaintext_len; i += 16) {
        uint8_t enc_counter[16] = { 0 };
        SM4(counter, enc_counter, rk, 1);

        size_t block_len = (plaintext_len - i) < 16 ? (plaintext_len - i) : 16;
        for (size_t j = 0; j < block_len; j++) {
            ciphertext[i + j] = plaintext[i + j] ^ enc_counter[j];
        }

        increment_counter(counter);
    }

    // 步骤4: 计算认证标签
    // GHASH(AAD || 密文 || AAD长度 || 密文长度)
    uint8_t len_block[16];
    uint64_t aad_bits = aad_len * 8;
    uint64_t ciphertext_bits = plaintext_len * 8;

    memset(len_block, 0, 16);
    memcpy(len_block, &aad_bits, 8);
    memcpy(len_block + 8, &ciphertext_bits, 8);

    // 计算GHASH
    uint8_t X[16] = { 0 };
    ghash(aad, ciphertext, ciphertext_bits, X);

    // 处理AAD
    for (size_t i = 0; i < aad_len; i += 16) {
        uint8_t block[16] = { 0 };
        size_t block_len = (aad_len - i) < 16 ? (aad_len - i) : 16;
        memcpy(block, aad + i, block_len);

        for (int j = 0; j < 16; j++) {
            X[j] ^= block[j];
        }
        gmul(X, H, X);
    }

    // 处理密文
    for (size_t i = 0; i < plaintext_len; i += 16) {
        uint8_t block[16] = { 0 };
        size_t block_len = (plaintext_len - i) < 16 ? (plaintext_len - i) : 16;
        memcpy(block, ciphertext + i, block_len);

        for (int j = 0; j < 16; j++) {
            X[j] ^= block[j];
        }
        gmul(X, H, X);
    }

    // 处理长度
    for (int j = 0; j < 16; j++) {
        X[j] ^= len_block[j];
    }
    gmul(X, H, X);

    // 计算 E0 = SM4(J0)
    SM4(J0, E0, rk, 1);

    // 最终标签 = GHASH ^ E0
    for (int i = 0; i < tag_len && i < 16; i++) {
        tag[i] = X[i] ^ E0[i];
    }
}

// SM4-GCM解密(与加密类似但验证标签)
int sm4_gcm_decrypt(const uint8_t* ciphertext, size_t ciphertext_len,
    const uint8_t* key, const uint8_t* iv, size_t iv_len,
    const uint8_t* aad, size_t aad_len,
    const uint8_t* tag, size_t tag_len,
    uint8_t* plaintext) {
    uint32_t rk[32];
    SM4_KEY_GEN((uint8_t*)key, rk);

    uint8_t H[16] = { 0 };
    uint8_t J0[16] = { 0 };
    uint8_t counter[16] = { 0 };
    uint8_t E0[16] = { 0 };
    uint8_t computed_tag[16] = { 0 };

    // 步骤1: 生成哈希子密钥 H = SM4(0^128)
    SM4(H, H, rk, 1);

    // 步骤2: 生成J0(初始计数器块)
    if (iv_len == 12) {
        memcpy(J0, iv, 12);
        J0[15] = 1;
    }
    
    memcpy(counter, J0, 16);
    increment_counter(counter);

    // 步骤3: 先计算认证标签(在解密前)
    uint8_t X[16] = { 0 };
    uint8_t len_block[16];
    uint64_t aad_bits = aad_len * 8;
    uint64_t ciphertext_bits = ciphertext_len * 8;

    memset(len_block, 0, 16);
    memcpy(len_block, &aad_bits, 8);
    memcpy(len_block + 8, &ciphertext_bits, 8);

    // 处理AAD
    for (size_t i = 0; i < aad_len; i += 16) {
        uint8_t block[16] = { 0 };
        size_t block_len = (aad_len - i) < 16 ? (aad_len - i) : 16;
        memcpy(block, aad + i, block_len);

        for (int j = 0; j < 16; j++) {
            X[j] ^= block[j];
        }
        gmul(X, H, X);
    }

    // 处理密文
    for (size_t i = 0; i < ciphertext_len; i += 16) {
        uint8_t block[16] = { 0 };
        size_t block_len = (ciphertext_len - i) < 16 ? (ciphertext_len - i) : 16;
        memcpy(block, ciphertext + i, block_len);

        for (int j = 0; j < 16; j++) {
            X[j] ^= block[j];
        }
        gmul(X, H, X);
    }

    // 处理长度
    for (int j = 0; j < 16; j++) {
        X[j] ^= len_block[j];
    }
    gmul(X, H, X);

    // 计算 E0 = SM4(J0)
    SM4(J0, E0, rk, 1);

    // 最终计算标签 = GHASH ^ E0
    for (int i = 0; i < tag_len && i < 16; i++) {
        computed_tag[i] = X[i] ^ E0[i];
    }

    // 验证标签
    for (int i = 0; i < tag_len; i++) {
        if (computed_tag[i] != tag[i]) {
            return -1; // 认证失败
        }
    }

    // 步骤4: 计数器模式解密密文(与加密相同)
    memcpy(counter, J0, 16);
    increment_counter(counter);

    for (size_t i = 0; i < ciphertext_len; i += 16) {
        uint8_t enc_counter[16] = { 0 };
        SM4(counter, enc_counter, rk, 1);

        size_t block_len = (ciphertext_len - i) < 16 ? (ciphertext_len - i) : 16;
        for (size_t j = 0; j < block_len; j++) {
            plaintext[i + j] = ciphertext[i + j] ^ enc_counter[j];
        }

        increment_counter(counter);
    }

    return 0; // 成功
}

int main() {
    // 测试SM4-GCM
    printf("=====================SM4-GCM测试=====================\n");

    uint8_t key[16] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                       0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10 };
    uint8_t iv[12] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                      0xfe, 0xdc, 0xba, 0x98 };
    uint8_t aad[16] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                       0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10 };
    uint8_t plaintext[32] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                            0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
                            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                            0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10 };
    uint8_t ciphertext[32] = { 0 };
    uint8_t decrypted[32] = { 0 };
    uint8_t tag[16] = { 0 };

    printf("明文:\n");
    for (int i = 0; i < 32; i++) {
        printf("%02x ", plaintext[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }

    sm4_gcm_encrypt(plaintext, 32, key, iv, 12, aad, 16, ciphertext, tag, 16);

    printf("\n密文:\n");
    for (int i = 0; i < 32; i++) {
        printf("%02x ", ciphertext[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }

    printf("\n标签:\n");
    for (int i = 0; i < 16; i++) {
        printf("%02x ", tag[i]);
    }
    printf("\n");

    int result = sm4_gcm_decrypt(ciphertext, 32, key, iv, 12, aad, 16, tag, 16, decrypted);

    printf("\n解密%s\n", result == 0 ? "成功" : "失败");

    if (result == 0) {
        printf("解密后的明文:\n");
        for (int i = 0; i < 32; i++) {
            printf("%02x ", decrypted[i]);
            if ((i + 1) % 16 == 0) printf("\n");
        }
    }

    // 测试错误标签(应该失败)
    uint8_t wrong_tag[16];
    memcpy(wrong_tag, tag, 16);
    wrong_tag[0] ^= 0x01; // 翻转一个比特

    result = sm4_gcm_decrypt(ciphertext, 32, key, iv, 12, aad, 16, wrong_tag, 16, decrypted);
    printf("\n使用错误标签测试: 解密%s\n", result == 0 ? "成功" : "失败(符合预期)");


    // ===================== 性能测试 =====================
    printf("\n=====================SM4-GCM性能测试=====================\n");

    // 预热缓存
    for (int i = 0; i < 10; i++) {
        sm4_gcm_encrypt(plaintext, 32, key, iv, 12, aad, 16, ciphertext, tag, 16);
        sm4_gcm_decrypt(ciphertext, 32, key, iv, 12, aad, 16, tag, 16, decrypted);
    }

    // 测试加密性能
    const int TEST_ROUNDS = 100;
    size_t total_bytes = 32 * TEST_ROUNDS; // 每次处理32字节

    // 加密性能测试
    uint64_t start_enc = rdtsc();
    for (int i = 0; i < TEST_ROUNDS; i++) {
        sm4_gcm_encrypt(plaintext, 32, key, iv, 12, aad, 16, ciphertext, tag, 16);
    }
    uint64_t end_enc = rdtsc();
    uint64_t enc_cycles = end_enc - start_enc;
    double enc_cycles_per_byte = (double)enc_cycles / total_bytes;

    printf("加密 %d 次总周期数: %llu\n", TEST_ROUNDS, enc_cycles);
    printf("加密总处理数据量: %zu bytes\n", total_bytes);
    printf("加密性能指标: %.2f cycles/byte\n", enc_cycles_per_byte);

    // 解密性能测试
    uint64_t start_dec = rdtsc();
    for (int i = 0; i < TEST_ROUNDS; i++) {
        sm4_gcm_decrypt(ciphertext, 32, key, iv, 12, aad, 16, tag, 16, decrypted);
    }
    uint64_t end_dec = rdtsc();
    uint64_t dec_cycles = end_dec - start_dec;
    double dec_cycles_per_byte = (double)dec_cycles / total_bytes;

    printf("\n解密 %d 次总周期数: %llu\n", TEST_ROUNDS, dec_cycles);
    printf("解密总处理数据量: %zu bytes\n", total_bytes);
    printf("解密性能指标: %.2f cycles/byte\n", dec_cycles_per_byte);
    return 0;
}