#include <iostream>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <chrono>
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

void SM3_openssl(const std::string& input) {
    // 初始化 OpenSSL
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    // 创建消息摘要上下文
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    const EVP_MD* md = EVP_sm3();

    // 初始化 SM3 哈希计算
    EVP_DigestInit_ex(mdctx, md, NULL);

    // 更新哈希计算（输入数据）
    EVP_DigestUpdate(mdctx, input.c_str(), input.length());

    // 存储哈希结果的缓冲区
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hashLen = 0;

    // 开始周期计数
    uint64_t start_cycles = rdtsc();

    for (int i = 0; i < 100; i++) {
        EVP_DigestFinal_ex(mdctx, hash, &hashLen);
    }

    uint64_t end_cycles = rdtsc();
    uint64_t total_cycles = end_cycles - start_cycles;

    size_t total_bytes = 6000;
    double cycles_per_byte = (double)total_cycles / total_bytes;

    printf("哈希100次总周期数: %llu\n", total_cycles);
    printf("总处理字节数: %zu bytes\n", total_bytes);
    printf("性能指标: %.2f cycles/byte\n", cycles_per_byte);

    // 完成哈希计算
    EVP_DigestFinal_ex(mdctx, hash, &hashLen);


    printf("哈希值为：\n", input.c_str());
    for (unsigned int i = 0; i < hashLen; i++) {
        printf("%02x ", hash[i]); 
    }
    printf("\n");

    // 释放上下文
    EVP_MD_CTX_free(mdctx);

    // 清理 OpenSSL 资源
    EVP_cleanup();
    ERR_free_strings();
}

int main() {
    std::string input = "SDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDUSDU";
    SM3_openssl(input);

    return 0;
}