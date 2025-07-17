#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <xmmintrin.h>
#include <intrin.h>

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
#define FF0(x, y, z) ((x) ^ (y) ^ (z))
#define FF1(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define GG0(x, y, z) ((x) ^ (y) ^ (z))
#define GG1(x, y, z) (((x) & (y)) | ((~(x)) & (z)))
#define rotl32(value, shift) ((value << shift) | (value>> (32 - shift)))
#define P0(x) ((x) ^ rotl32((x), 9) ^ rotl32((x), 17))
#define P1(x) ((x) ^ rotl32((x), 15) ^ rotl32((x), 23))

#define SIMD_rotl32(x, j) _mm_xor_si128(_mm_slli_epi32(x, j), _mm_srli_epi32(x, 32 - j))
#define SIMD_P1(x) _mm_xor_si128(_mm_xor_si128(x, SIMD_rotl32(x, 15)), SIMD_rotl32(x, 23))
#define to_BE32(x) ((x & 0xff000000) >> 24) | ((x & 0x00ff0000) >> 8) | ((x & 0x0000ff00) << 8) | ((x & 0x000000ff) << 24)
#define xor3(x, y, z) _mm_xor_si128(_mm_xor_si128((x), (y)), (z))

#define CF1(i) \
    do { \
        temp = rotl32(A, 12) + E + rotl32(0x79cc4519, i); \
        SS1 = rotl32(temp, 7); \
        SS2 = SS1 ^ rotl32(A, 12); \
        TT1 = FF0(A, B, C) + D + SS2 + W1[i]; \
        TT2 = GG0(E, F, G) + H + SS1 + W[i]; \
        D = C; \
        C = rotl32(B, 9); \
        B = A; \
        A = TT1; \
        H = G; \
        G = rotl32(F, 19); \
        F = E; \
        E = P0(TT2); \
    } while (0)

#define CF2(i) \
    do { \
        temp = rotl32(A, 12) + E + rotl32(0x7a879d8a, i % 32); \
        SS1 = rotl32(temp, 7); \
        SS2 = SS1 ^ rotl32(A, 12); \
        TT1 = FF1(A, B, C) + D + SS2 + W1[i]; \
        TT2 = GG1(E, F, G) + H + SS1 + W[i]; \
        D = C; \
        C = rotl32(B, 9); \
        B = A; \
        A = TT1; \
        H = G; \
        G = rotl32(F, 19); \
        F = E; \
        E = P0(TT2); \
    } while (0)


//SIMD生成W[68]，每次生成128bit，也就是W[4*j]，W[4*j+1]，W[4*j+2]，W[4*j+3]
void GEN_W(int j, uint32_t* W) {
    W[4 * j] = P1(W[4 * j - 16] ^ W[4 * j - 9] ^ (rotl32(W[4 * j - 3], 15))) ^ rotl32(W[4 * j - 13], 7) ^ W[4 * j - 6];
    __m128i W_j_16 = _mm_setr_epi32(W[4 * j - 16], W[4 * j - 15], W[4 * j - 14], W[4 * j - 13]);
    __m128i W_j_9 = _mm_setr_epi32(W[4 * j - 9], W[4 * j - 8], W[4 * j - 7], W[4 * j - 6]);
    __m128i W_j_3 = _mm_setr_epi32(W[4 * j - 3], W[4 * j - 2], W[4 * j - 1], W[4 * j]);
    __m128i W_j_13 = _mm_setr_epi32(W[4 * j - 13], W[4 * j - 12], W[4 * j - 11], W[4 * j - 10]);
    __m128i W_j_6 = _mm_setr_epi32(W[4 * j - 6], W[4 * j - 5], W[4 * j - 4], W[4 * j - 3]);
    __m128i re = xor3(
        SIMD_P1(xor3(W_j_16, W_j_9, SIMD_rotl32(W_j_3, 15))),
        SIMD_rotl32(W_j_13, 7),
        W_j_6
    );
    _mm_storeu_si128((__m128i*) & W[4 * j], re);
}


//同理生成W1[64]
void GEN_W1(int j, uint32_t* W1, uint32_t* W) {
    _mm_storeu_si128((__m128i*) & W1[4 * j], _mm_xor_si128(_mm_loadu_si128((__m128i*) & W[4 * j]), _mm_loadu_si128((__m128i*) & W[4 * j + 4])));

}

//消息填充
uint32_t PADDING(uint8_t* m, size_t len, uint8_t* buffer) {
    uint32_t bit_len = 8 * len;
    uint32_t left_bit = bit_len % 512;
    int k = left_bit > 448 ? 2 : 1;
    //填充后的字节数
    uint32_t after_padding_len = (len/64)*64 + k * 64;
    int i;
    __m128i* p = (__m128i*)m;
    __m128i* q = (__m128i*)buffer;
    for (i = 0; i < len / 16; i++) {
        _mm_storeu_si128(q + i, _mm_loadu_si128(p + i));
    }
    memcpy(buffer + (len / 16) * 16, m + (len / 16) * 16, len % 16);
    buffer[len] = 0x80;
    //填充0
    memset(&buffer[len + 1], 0, after_padding_len - (len + 1) - 8);
    //填充长度
    uint64_t bit_len_be = ((bit_len & 0xFF00000000000000ull) >> 56) |
        ((bit_len & 0x00FF000000000000ull) >> 40) |
        ((bit_len & 0x0000FF0000000000ull) >> 24) |
        ((bit_len & 0x000000FF00000000ull) >> 8) |
        ((bit_len & 0x00000000FF000000ull) << 8) |
        ((bit_len & 0x0000000000FF0000ull) << 24) |
        ((bit_len & 0x000000000000FF00ull) << 40) |
        ((bit_len & 0x00000000000000FFull) << 56);
    memcpy(&buffer[after_padding_len - 8], &bit_len_be, 8);
    return after_padding_len;
}


uint32_t IV[8] = { 0x7380166f, 0x4914b2b9, 0x172442d7, 0xda8a0600, 0xa96f30bc, 0x163138aa, 0xe38dee4d, 0xb0fb0e4e };

void SIMD_CF(uint32_t* digest, uint8_t* p) {
    uint32_t W[68] = {0};
    uint32_t W1[64] = {0};
    uint32_t temp, SS1, SS2, TT1, TT2;

    //转成大端存储
    for (int i = 0; i < 16; i++) {
        W[i] = to_BE32(((uint32_t*)p)[i]);
    }
    for (int i = 4; i <= 16; i++) {
        GEN_W(i, W);
    }
    for (int i = 0; i < 16; i++) {
        GEN_W1(i, W1, W);
    }

    uint32_t A = digest[0], B = digest[1], C = digest[2], D = digest[3];
    uint32_t E = digest[4], F = digest[5], G = digest[6], H = digest[7];

    for (int i = 0; i < 16; i++) {
        CF1(i);
    }
    for (int i = 16; i < 64; i++) {
        CF2(i);
    }
    digest[0] ^= A;
    digest[1] ^= B;
    digest[2] ^= C;
    digest[3] ^= D;
    digest[4] ^= E;
    digest[5] ^= F;
    digest[6] ^= G;
    digest[7] ^= H;
}

void SIMD_SM3(uint8_t* m, uint32_t* hash, size_t len, uint8_t* buffer) {
    uint32_t n = PADDING(m, len, buffer) / 64;
    uint32_t IV_copy[8];
    memcpy(IV_copy, IV, sizeof(IV));
    for (int i = 0; i < n; i++) {
        SIMD_CF(IV_copy, buffer + i * 64);
    }
    for (int i = 0; i < 8; i++) {
        hash[i] = IV_copy[i];
    }
}
uint32_t PADDING_attack(uint8_t* m, size_t len, uint8_t* buffer) {
    uint32_t bit_len = 8 * len;
    len = len - 64;
    uint32_t left_bit = bit_len % 512;
    int k = left_bit > 448 ? 2 : 1;
    //填充后的字节数
    uint32_t after_padding_len = (len / 64) * 64 + k * 64;
    int i;
    __m128i* p = (__m128i*)m;
    __m128i* q = (__m128i*)buffer;
    for (i = 0; i < len / 16; i++) {
        _mm_storeu_si128(q + i, _mm_loadu_si128(p + i));
    }
    memcpy(buffer + (len / 16) * 16, m + (len / 16) * 16, len % 16);
    buffer[len] = 0x80;
    //填充0
    memset(&buffer[len + 1], 0, after_padding_len - (len + 1) - 8);
    //填充长度
    uint64_t bit_len_be = ((bit_len & 0xFF00000000000000ull) >> 56) |
        ((bit_len & 0x00FF000000000000ull) >> 40) |
        ((bit_len & 0x0000FF0000000000ull) >> 24) |
        ((bit_len & 0x000000FF00000000ull) >> 8) |
        ((bit_len & 0x00000000FF000000ull) << 8) |
        ((bit_len & 0x0000000000FF0000ull) << 24) |
        ((bit_len & 0x000000000000FF00ull) << 40) |
        ((bit_len & 0x00000000000000FFull) << 56);
    memcpy(&buffer[after_padding_len - 8], &bit_len_be, 8);
    return after_padding_len;
}

uint32_t PADDING_AND_APPEND(uint8_t* m, size_t len, uint8_t* buffer, uint8_t* append, size_t append_len) {
    PADDING(m, len, buffer);
    memcpy(buffer + PADDING(m, len, buffer), append, append_len);
    //append后消息的总字节数
    return PADDING(m, len, buffer) + append_len;
}

void SIMD_SM3_attack(uint8_t* m, uint32_t* hash, size_t len, uint8_t* buffer,uint32_t * IV1) {
    uint32_t n = PADDING_attack(m, len, buffer) / 64;
    for (int i = 0; i < n; i++) {
        SIMD_CF(IV1, buffer + i * 64);
    }
    for (int i = 0; i < 8; i++) {
        hash[i] =IV1[i];
    }
}



int main() {
    //计算H(M)
    uint8_t m[] = "abc";
    int len = strlen((char*)m);

    uint32_t hash[8];
    uint8_t buffer[2 << 12] = { 0 };  // 确保足够大

    SIMD_SM3(m, hash, len, buffer);

 
    printf("\n原始消息的哈希值为:\n");

    for (int i = 0; i < 8; i++) {
        printf("%08x ", hash[i]);
    }
    printf("\n");


    //计算IV=H(M)时的H(append)
    uint8_t append[] = "abc";
    int append_len = strlen((char*)append);
 
    uint32_t hash1[8];
    uint8_t buffer1[2 << 12] = { 0 };

    SIMD_SM3_attack(append, hash1, append_len+64, buffer1, hash);
    
    printf("\n输入IV为上述哈希值时，附加消息的哈希值为:\n");

    for (int i = 0; i < 8; i++) {
        printf("%08x ", hash1[i]);
    }
    printf("\n");


    uint8_t forge_buffer[2 << 12] = { 0 };
    uint32_t forge_len=PADDING_AND_APPEND(m, len, forge_buffer, append, append_len);


    uint8_t buffer2[2 << 12] = { 0 };
    
    uint32_t hash2[8];

    SIMD_SM3(forge_buffer, hash2, forge_len, buffer2);


    printf("\n伪造消息的哈希值为:\n");

    for (int i = 0; i < 8; i++) {
        printf("%08x ", hash2[i]);
    }
    printf("\n");


    


    return 0;
}