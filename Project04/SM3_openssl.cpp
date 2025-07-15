#include <iostream>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <chrono>

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

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    // 完成哈希计算
    EVP_DigestFinal_ex(mdctx, hash, &hashLen);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("SM3_openssl哈希耗时: %lld μs\n", total_time.count());

    // 使用 printf 按字节输出哈希值（带空格分隔）
    printf("消息 \"%s\" 的哈希值为：\n", input.c_str());
    for (unsigned int i = 0; i < hashLen; i++) {
        printf("%02x ", hash[i]); // %02x 保证两位十六进制，空格分隔
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