#include<stdio.h>
#include<stdint.h>
#include<intrin.h>
#include <xmmintrin.h>
#include<chrono>

using namespace std;

// 添加CPU周期计数函数
#if defined(_MSC_VER)
#include <intrin.h>
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
#define rotl_epi32(value,shift) _mm_xor_si128(_mm_slli_epi32(value,shift),_mm_srli_epi32(value,32-shift))


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


#define xor4(x1,x2,x3,x4) _mm_xor_si128(_mm_xor_si128(x1,x2),_mm_xor_si128(x3,x4))
#define xor6(x1,x2,x3,x4,x5,x6) _mm_xor_si128(_mm_xor_si128(x1,x2),xor4(x3,x4,x5,x6))

// SM4 S盒到AES S盒的仿射变换矩阵

__m128i sm4_to_aes_matrix = _mm_set1_epi64x(0b00000110'00010111'00001010'00110101'00111010'01110010'10011011'00001101);
// AES S盒到SM4 S盒的仿射变换矩阵

__m128i aes_to_sm4_matrix = _mm_set1_epi64x(0b10011100'00111010'10001100'11000100'01110100'11000011'01100001'10101000);

// 仿射变换常数
__m128i sm4_to_aes_constant = _mm_set1_epi8(0b00100011);
__m128i aes_to_sm4_constant = _mm_set1_epi8(0b00111011);


// 使用GF2P8AFFINEQB和AES-NI实现的SM4 S盒
__m128i SM4_BOX_TO_AES_GFNI(__m128i x) {

    // 1. SM4输入 -> AES输入
    __m128i aes_input = _mm_gf2p8affine_epi64_epi8(x, sm4_to_aes_matrix, 0);

    aes_input = _mm_xor_si128(aes_input, sm4_to_aes_constant);

    // 2. 使用AES-NI计算AES S盒
    // 这里使用AESENCLAST指令，因为AES S盒 = AESENCLAST(x, 0)
    __m128i aes_output = _mm_aesenclast_si128(aes_input, _mm_setzero_si128());

    // 3. AES输出 -> SM4输出
    __m128i sm4_output = _mm_gf2p8affine_epi64_epi8(aes_output, aes_to_sm4_matrix, 0);
    sm4_output = _mm_xor_si128(sm4_output, aes_to_sm4_constant);

    return sm4_output;
}

#define GFNI_SM4_1R(index)\
	do{\
	__m128i k=_mm_set1_epi32((is_enc==1?rk[index]:rk[31-index]));\
	temp=xor4(A[1],A[2],A[3],k);\
    temp=SM4_BOX_TO_AES_GFNI(temp);\
    temp=xor6(A[0],temp,rotl_epi32(temp,2),rotl_epi32(temp,10),rotl_epi32(temp,18),rotl_epi32(temp,24));\
    A[0]=A[1];\
    A[1]=A[2];\
    A[2]=A[3];\
    A[3]=temp;\
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

void SM4_GFNI(uint8_t* input, uint8_t* output, uint32_t* rk, bool is_enc) {
    __m128i temp;
    __m128i A[4];
    __m128i vindex;

    temp = _mm_loadu_si128((__m128i*)input);
    vindex = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);

    //小端序-->大端序

    A[0] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(temp, temp), _mm_unpacklo_epi32(temp, temp));
    A[1] = _mm_unpackhi_epi64(_mm_unpacklo_epi32(temp, temp), _mm_unpacklo_epi32(temp, temp));
    A[2] = _mm_unpacklo_epi64(_mm_unpackhi_epi32(temp, temp), _mm_unpackhi_epi32(temp, temp));
    A[3] = _mm_unpackhi_epi64(_mm_unpackhi_epi32(temp, temp), _mm_unpackhi_epi32(temp, temp));

    A[0] = _mm_shuffle_epi8(A[0], vindex);
    A[1] = _mm_shuffle_epi8(A[1], vindex);
    A[2] = _mm_shuffle_epi8(A[2], vindex);
    A[3] = _mm_shuffle_epi8(A[3], vindex);

    for (int i = 0; i < 32; i++) {

        GFNI_SM4_1R(i);
    }
    //大端序-->小端序

    A[0] = _mm_shuffle_epi8(A[0], vindex);
    A[1] = _mm_shuffle_epi8(A[1], vindex);
    A[2] = _mm_shuffle_epi8(A[2], vindex);
    A[3] = _mm_shuffle_epi8(A[3], vindex);

    _mm_storeu_si128((__m128i*)output, _mm_unpacklo_epi64(_mm_unpacklo_epi32(A[3], A[2]), _mm_unpacklo_epi32(A[1], A[0])));

}


// 在GF(2^128)域中乘法，使用约化多项式x^128 + x^7 + x^2 + x + 1
static const __m128i gcm_reduction_poly = _mm_setr_epi32(0x00000001, 0x00000000, 0x00000000, 0xC2000000);

// 使用PCLMULQDQ指令的GF(2^128)乘法
__m128i gfmul(__m128i a, __m128i b) {
    __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;

    // Karatsuba乘法
    tmp3 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp3 = _mm_xor_si128(tmp3, tmp5);
    tmp2 = _mm_xor_si128(tmp4, _mm_clmulepi64_si128(a, b, 0x11));

    // 模约减
    tmp5 = _mm_clmulepi64_si128(tmp3, gcm_reduction_poly, 0x01);
    tmp4 = _mm_slli_si128(tmp5, 8);
    tmp5 = _mm_srli_si128(tmp5, 8);
    tmp3 = _mm_xor_si128(tmp3, tmp4);
    tmp2 = _mm_xor_si128(tmp2, tmp5);

    tmp5 = _mm_clmulepi64_si128(tmp3, gcm_reduction_poly, 0x00);
    return _mm_xor_si128(tmp2, tmp5);
}


// 计算 GHASH
__m128i ghash(__m128i H, const uint8_t* aad, size_t aad_len,
    const uint8_t* ciphertext, size_t ciphertext_len) {
    __m128i X = _mm_setzero_si128();

    // 处理 AAD
    for (size_t i = 0; i < aad_len; i += 16) {
        __m128i block = _mm_setzero_si128();
        size_t block_len = (aad_len - i < 16) ? (aad_len - i) : 16;
        memcpy(&block, aad + i, block_len);
        X = _mm_xor_si128(X, block);
        X = gfmul(X, H);
    }

    // 处理密文
    for (size_t i = 0; i < ciphertext_len; i += 16) {
        __m128i block = _mm_setzero_si128();
        size_t block_len = (ciphertext_len - i < 16) ? (ciphertext_len - i) : 16;
        memcpy(&block, ciphertext + i, block_len);
        X = _mm_xor_si128(X, block);
        X = gfmul(X, H);
    }

    // 处理长度块（AAD长度 || 密文长度），单位：比特
    __m128i len_block = _mm_setzero_si128();
    uint64_t aad_bits = aad_len * 8;
    uint64_t ciphertext_bits = ciphertext_len * 8;
    memcpy(&len_block, &aad_bits, 8);
    memcpy((uint8_t*)&len_block + 8, &ciphertext_bits, 8);

    X = _mm_xor_si128(X, len_block);
    X = gfmul(X, H);

    return X;
}

// 128位计数器递增(大端序)
void increment_counter(__m128i* counter) {
    uint64_t* c = (uint64_t*)counter;
    if (++c[1] == 0) {
        ++c[0];
    }
}

// SM4-GCM加密
void sm4_gcm_encrypt(const uint8_t* plaintext, size_t plaintext_len,
    const uint8_t* key, const uint8_t* iv, size_t iv_len,
    const uint8_t* aad, size_t aad_len,
    uint8_t* ciphertext, uint8_t* tag, size_t tag_len) {
    uint32_t rk[32];
    SM4_KEY_GEN((uint8_t*)key, rk);

    __m128i H = _mm_setzero_si128();
    __m128i J0 = _mm_setzero_si128();
    __m128i counter = _mm_setzero_si128();
    __m128i E0 = _mm_setzero_si128();
    __m128i X = _mm_setzero_si128();

    // 步骤1: 生成哈希子密钥 H = SM4(0^128)
    uint8_t zero_block[16] = { 0 };
    SM4_GFNI(zero_block, (uint8_t*)&H, rk, 1);

    // 步骤2: 生成J0(初始计数器块)
    if (iv_len == 12) {
        // 96位IV: J0 = IV || 0^31 || 1
        memcpy(&J0, iv, 12);
        ((uint8_t*)&J0)[15] = 1;
    }
    else {
        __m128i X = ghash(H, aad, aad_len, ciphertext, plaintext_len);

    }
    counter = J0;
    increment_counter(&counter);

    // 步骤3: 计数器模式加密明文
    for (size_t i = 0; i < plaintext_len; i += 16) {
        __m128i enc_counter = _mm_setzero_si128();
        SM4_GFNI((uint8_t*)&counter, (uint8_t*)&enc_counter, rk, 1);

        size_t block_len = (plaintext_len - i) < 16 ? (plaintext_len - i) : 16;
        __m128i plain_block = _mm_setzero_si128();
        memcpy(&plain_block, plaintext + i, block_len);

        __m128i cipher_block = _mm_xor_si128(plain_block, enc_counter);
        memcpy(ciphertext + i, &cipher_block, block_len);

        increment_counter(&counter);
    }

    // 步骤4: 计算认证标签
    // GHASH(AAD || 密文 || len(AAD) || len(密文))

    // 处理AAD
    for (size_t i = 0; i < aad_len; i += 16) {
        __m128i block = _mm_setzero_si128();
        size_t block_len = (aad_len - i) < 16 ? (aad_len - i) : 16;
        memcpy(&block, aad + i, block_len);

        X = _mm_xor_si128(X, block);
        X = gfmul(X, H);
    }

    // 处理密文
    for (size_t i = 0; i < plaintext_len; i += 16) {
        __m128i block = _mm_setzero_si128();
        size_t block_len = (plaintext_len - i) < 16 ? (plaintext_len - i) : 16;
        memcpy(&block, ciphertext + i, block_len);

        X = _mm_xor_si128(X, block);
        X = gfmul(X, H);
    }

    // 处理长度
    __m128i len_block = _mm_setzero_si128();
    uint64_t aad_bits = aad_len * 8;
    uint64_t ciphertext_bits = plaintext_len * 8;
    memcpy(&len_block, &aad_bits, 8);
    memcpy((uint8_t*)&len_block + 8, &ciphertext_bits, 8);

    X = _mm_xor_si128(X, len_block);
    X = gfmul(X, H);

    // 计算 E0 = SM4(J0)
    SM4_GFNI((uint8_t*)&J0, (uint8_t*)&E0, rk, 1);

    // 最终标签 = GHASH ^ E0
    __m128i final_tag = _mm_xor_si128(X, E0);

    // 复制标签到输出
    size_t copy_len = tag_len < 16 ? tag_len : 16;
    memcpy(tag, &final_tag, copy_len);
}

// SM4-GCM解密
int sm4_gcm_decrypt(const uint8_t* ciphertext, size_t ciphertext_len,
    const uint8_t* key, const uint8_t* iv, size_t iv_len,
    const uint8_t* aad, size_t aad_len,
    const uint8_t* tag, size_t tag_len,
    uint8_t* plaintext) {
    uint32_t rk[32];
    SM4_KEY_GEN((uint8_t*)key, rk);

    __m128i H = _mm_setzero_si128();
    __m128i J0 = _mm_setzero_si128();
    __m128i X = _mm_setzero_si128();

    // 步骤1: 生成哈希子密钥 H = SM4(0^128)
    uint8_t zero_block[16] = { 0 };
    SM4_GFNI(zero_block, (uint8_t*)&H, rk, 1);

    // 步骤2: 生成J0(初始计数器块)
    if (iv_len == 12) {
        // 96位IV: J0 = IV || 0^31 || 1
        memcpy(&J0, iv, 12);
        ((uint8_t*)&J0)[15] = 1;
    }
    else {
        __m128i X = ghash(H, aad, aad_len, ciphertext, ciphertext_len);
    }

    // 步骤3: 先计算认证标签(在解密前)

    // 处理AAD
    for (size_t i = 0; i < aad_len; i += 16) {
        __m128i block = _mm_setzero_si128();
        size_t block_len = (aad_len - i) < 16 ? (aad_len - i) : 16;
        memcpy(&block, aad + i, block_len);

        X = _mm_xor_si128(X, block);
        X = gfmul(X, H);
    }

    // 处理密文
    for (size_t i = 0; i < ciphertext_len; i += 16) {
        __m128i block = _mm_setzero_si128();
        size_t block_len = (ciphertext_len - i) < 16 ? (ciphertext_len - i) : 16;
        memcpy(&block, ciphertext + i, block_len);

        X = _mm_xor_si128(X, block);
        X = gfmul(X, H);
    }

    // 处理长度
    __m128i len_block = _mm_setzero_si128();
    uint64_t aad_bits = aad_len * 8;
    uint64_t ciphertext_bits = ciphertext_len * 8;
    memcpy(&len_block, &aad_bits, 8);
    memcpy((uint8_t*)&len_block + 8, &ciphertext_bits, 8);

    X = _mm_xor_si128(X, len_block);
    X = gfmul(X, H);

    // 计算 E0 = SM4(J0)
    __m128i E0 = _mm_setzero_si128();
    SM4_GFNI((uint8_t*)&J0, (uint8_t*)&E0, rk, 1);

    // 最终计算标签 = GHASH ^ E0
    __m128i computed_tag = _mm_xor_si128(X, E0);

    // 验证标签
    __m128i received_tag = _mm_setzero_si128();
    memcpy(&received_tag, tag, tag_len < 16 ? tag_len : 16);

    // 比较标签
    __m128i eq = _mm_xor_si128(computed_tag, received_tag);
    if (!_mm_test_all_zeros(eq, eq)) {
        return -1; // 认证失败
    }

    // 步骤4: 计数器模式解密密文
    __m128i counter = J0;
    increment_counter(&counter);

    for (size_t i = 0; i < ciphertext_len; i += 16) {
        __m128i enc_counter = _mm_setzero_si128();
        SM4_GFNI((uint8_t*)&counter, (uint8_t*)&enc_counter, rk, 1);

        size_t block_len = (ciphertext_len - i) < 16 ? (ciphertext_len - i) : 16;
        __m128i cipher_block = _mm_setzero_si128();
        memcpy(&cipher_block, ciphertext + i, block_len);

        __m128i plain_block = _mm_xor_si128(cipher_block, enc_counter);
        memcpy(plaintext + i, &plain_block, block_len);

        increment_counter(&counter);
    }

    return 0; // 成功
}

int main() {
    // 测试SM4-GCM
    printf("=====================SM4-GCM-OPT测试=====================\n");

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

    // 性能测试
    printf("\n=====================SM4-GCM-OPT性能测试=====================\n");

    // 预热缓存
    for (int i = 0; i < 10; i++) {
        sm4_gcm_encrypt(plaintext, 32, key, iv, 12, aad, 16, ciphertext, tag, 16);
    }

    // 测量加密100次的总周期数
    uint64_t start_cycles = rdtsc();
    for (int i = 0; i < 100; i++) {
        sm4_gcm_encrypt(plaintext, 32, key, iv, 12, aad, 16, ciphertext, tag, 16);
    }
    uint64_t end_cycles = rdtsc();
    uint64_t total_cycles = end_cycles - start_cycles;

    // 计算总处理的数据量(bytes)
    size_t total_bytes = 100 * 32; // 100次加密，每次32字节

    // 计算cycles/byte
    double cycles_per_byte = (double)total_cycles / total_bytes;
    printf("加密100次总周期数: %llu\n", total_cycles);
    printf("总处理数据量: %zu bytes\n", total_bytes);
    printf("性能指标: %.2f cycles/byte\n", cycles_per_byte);

    int TEST_ROUNDS = 100;
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