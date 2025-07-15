#include<stdio.h>
#include<stdint.h>
#include<string.h>
#include<chrono>
#include <intrin.h>

// ���CPU���ڼ�������
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

#define FF0(x,y,z) ((x)^(y)^(z))
#define FF1(x,y,z) (((x)&(y))|((x)&(z))|((y)&(z)))

#define GG0(x,y,z) ((x)^(y)^(z))
#define GG1(x,y,z) (((x)&(y))|((~(x))&(z)))

#define rotl32(value,shift) ((value<<shift)|(value>>(32-shift)))
#define P0(x) ((x)^rotl32((x),9)^rotl32((x),17))
#define P1(x) ((x)^rotl32((x),15)^rotl32((x),23))

#define T0 0x79CC4519
#define T1 0x7A879D8A

//����ת��Ϊ��˴洢
#define to_BE32(x) ((x & 0xff000000) >> 24) |((x & 0x00ff0000) >> 8) |((x & 0x0000ff00) << 8) |((x & 0x000000ff) << 24)
#define to_BE64(x) ((x & 0xff00000000000000ull) >> 56) |\
((x & 0x00ff000000000000ull) >> 40) |\
((x & 0x0000ff0000000000ull) >> 24) |\
((x & 0x000000ff00000000ull) >> 8) |\
((x & 0x00000000ff000000ull) << 8) |\
((x & 0x0000000000ff0000ull) << 24) |\
((x & 0x000000000000ff00ull) << 40) |\
((x & 0x00000000000000ffull) << 56)

//���鳤��64B=512b
#define block_size 64
//ÿ��ѹ�����Ϊ32B=256b
#define digest_size 32

typedef struct sm3_ctx {
    //�洢��ǰժҪֵ
    uint32_t digest[digest_size / sizeof(uint32_t)];
    //�Ѿ�����Ŀ���
    int num_block;
    uint8_t buffer[block_size];
    //buffer��δ������ֽ���
    int num;
}SM3_CTX;

void SM3_INIT(SM3_CTX* ctx) {
    //��ʼIV
    ctx->digest[0] = 0x7380166F;
    ctx->digest[1] = 0x4914B2B9;
    ctx->digest[2] = 0x172442D7;
    ctx->digest[3] = 0xDA8A0600;
    ctx->digest[4] = 0xA96F30BC;
    ctx->digest[5] = 0x163138AA;
    ctx->digest[6] = 0xE38DEE4D;
    ctx->digest[7] = 0xB0FB0E4E;

    ctx->num_block = 0;
    ctx->num = 0;
}

void CF(uint32_t digest[block_size / sizeof(uint32_t)], const uint8_t block[block_size]) {
    uint32_t A = digest[0];
    uint32_t B = digest[1];
    uint32_t C = digest[2];
    uint32_t D = digest[3];
    uint32_t E = digest[4];
    uint32_t F = digest[5];
    uint32_t G = digest[6];
    uint32_t H = digest[7];

    //��Ϣ��չ
    uint32_t W[68], W1[64];
    const uint32_t* p = (const uint32_t*)(block);
    //ÿ����ת��Ϊ��˴洢
    for (int i = 0; i < 16; i++) {
        W[i] = to_BE32( p[i]);
    }
    for (int i = 16; i < 68; i++) {
        W[i] = P1(W[i - 16] ^ W[i - 9] ^ rotl32(W[i - 3], 15)) ^ rotl32(W[i - 13], 7) ^ W[i - 6];
    }
    for (int i = 0; i < 64; i++) {
        W1[i] = W[i] ^ W[i + 4];
    }

    //ѹ��
    uint32_t SS1, SS2, TT1, TT2;

    for (int i = 0; i < 16; i++) {
        SS1 = rotl32((rotl32(A, 12) + E + rotl32(T0, i)), 7);
        SS2 = SS1 ^ rotl32(A, 12);
        TT1 = FF0(A, B, C) + D + SS2 + W1[i];
        TT2 = GG0(E, F, G) + H + SS1 + W[i];
        D = C,C = rotl32(B, 9), B = A,A = TT1;
        H = G, G = rotl32(F, 19), F = E, E = P0(TT2);
    }
    for (int i = 16; i < 64; i++) {
        SS1 = rotl32((rotl32(A, 12) + E + rotl32(T1, i)), 7);
        SS2 = SS1 ^ rotl32(A, 12);
        TT1 = FF1(A, B, C) + D + SS2 + W1[i];
        TT2 = GG1(E, F, G) + H + SS1 + W[i];
        D = C, C = rotl32(B, 9),B = A,A = TT1;
        H = G, G = rotl32(F, 19), F = E, E = P0(TT2);
    }

    digest[0] ^= A, digest[1] ^= B, digest[2] ^= C, digest[3] ^= D;
    digest[4] ^= E, digest[5] ^= F, digest[6] ^= G, digest[7] ^= H;
}

void SM3_UPDATE(SM3_CTX* ctx, const uint8_t* data, size_t dlen) {
    //����֮ǰδ�����Ŀ�
    if (ctx->num) {
        unsigned int n = block_size - ctx->num;
        //�����ݲ��������Ϊ����
        if (dlen < n) {
            //�ŵ�buffer�еȴ�����
            memcpy(ctx->buffer + ctx->num, data, dlen);
            ctx->num += dlen;
            return;
        }
        else {
            //���Ϊ������д���
            memcpy(ctx->buffer + ctx->num, data, n);
            CF(ctx->digest, ctx->buffer);
            ctx->num_block++;
            data += n;
            dlen -= n;
        }
    }
    while (dlen >= block_size) {
        CF(ctx->digest, data);
        ctx->num_block++;
        data += block_size;
        dlen -= block_size;
    }
    ctx->num = dlen;
    if (dlen) {
        memcpy(ctx->buffer, data, dlen);
    }
}

void SM3_FINAL(SM3_CTX* ctx, uint8_t* digest) {
    uint32_t* p = (uint32_t*)digest;
    uint64_t* len = (uint64_t*)(ctx->buffer + block_size - 8);

    //��Ϣĩβ���1������0
    ctx->buffer[ctx->num] = 0x80;

    //��ʣ��ռ��ܹ����9�ֽ�(0x80+8�ֽ���Ϣ����)
    if (ctx->num + 9 <= block_size) {
        memset(ctx->buffer + ctx->num + 1, 0, block_size - ctx->num - 9);
    }
    else {
        //���Ϊ���鴦��
        memset(ctx->buffer + ctx->num + 1, 0, block_size - ctx->num - 1);
        CF(ctx->digest, ctx->buffer);
        memset(ctx->buffer, 0, block_size - 8);
    }
    //�����Ϣ����
    len[0] =(uint64_t)(ctx->num_block) * 512 + (ctx->num << 3);
    len[0] = to_BE64(len[0]);
    CF(ctx->digest, ctx->buffer);
    for (uint32_t i = 0; i < 8; i++){
        p[i] =to_BE32(ctx->digest[i]);
    }
    memset(ctx, 0, sizeof(SM3_CTX));
}


void SM3(const uint8_t* message, size_t mlen, uint8_t res[block_size]) {
    SM3_CTX ctx;
    SM3_INIT(&ctx);
    SM3_UPDATE(&ctx, message, mlen);
    SM3_FINAL(&ctx, res);
}

int main() {
    uint8_t hash[32] = {};
    uint8_t message[61] = "SDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDU";
    size_t len = strlen((char*)message);
    const int iterations = 100; // ����100��

    // Ԥ�ȣ�������������
    SM3(message, len, hash);

    // ��ʼ���ڼ���
    uint64_t start_cycles = rdtsc();

    for (int i = 0; i < iterations; i++) {
        SM3(message, len, hash);
    }

    uint64_t end_cycles = rdtsc();
    uint64_t total_cycles = end_cycles - start_cycles;

    size_t total_bytes = len * iterations;
    double cycles_per_byte = (double)total_cycles / total_bytes;

    printf("��ϣ100����������: %llu\n", total_cycles);
    printf("�ܴ����ֽ���: %zu bytes\n", total_bytes);
    printf("����ָ��: %.2f cycles/byte\n", cycles_per_byte);

    // ���һ�ι�ϣ���������֤
    printf("\n��ϣֵΪ:\n");
    for (int i = 0; i < 32; i++)
        printf("%02x ", hash[i]);
    printf("\n");

    return 0;
}