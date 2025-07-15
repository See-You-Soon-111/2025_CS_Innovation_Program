#include<stdio.h>
#include<stdint.h>
#include<intrin.h>
#include <xmmintrin.h>
#include<chrono>

using namespace std;

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

//��ʼ��Կ
#define LOAD_KEY(index)\
	do{\
		k[index]=(key[index<<2]<<24)|(key[(index<<2)+1]<<16)|(key[(index<<2)+2]<<8)|(key[(index<<2)+3]);\
		k[index]=k[index]^FK[index];\
	}while(0)

//һ������Կ����
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

//һ��SM4����
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


int main() {
	uint8_t input[16] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10 };
	uint8_t key[16] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10 };
	uint32_t rk[32];

	SM4_KEY_GEN(key, rk);

	printf("=====================SM4�ӽ���=====================\n");
	printf("���ģ�\n");
	SM4(input, input, rk, 1);
	for (int i = 0; i < 16; i++) {
		printf("%02x ", input[i]);
	}
	printf("\n");


	printf("���ģ�\n");
	SM4(input, input, rk, 0);
	for (int i = 0; i < 16; i++) {
		printf("%02x ", input[i]);
	}
	printf("\n");

	printf("=====================SM4���ܲ���=====================\n");

	// Ԥ�Ȼ���
	for (int i = 0; i < 10; i++) {
		SM4(input, input, rk, 1);
	}

	// ��������100�ε���������
	uint64_t start_cycles = rdtsc();
	for (int i = 0; i < 100; i++) {
		SM4(input, input, rk, 1);
	}
	uint64_t end_cycles = rdtsc();
	uint64_t total_cycles = end_cycles - start_cycles;

	// �����ܴ����������(bytes)
	size_t total_bytes = 100 * 16; // 100�μ��ܣ�ÿ��16�ֽ�

	// ����cycles/byte
	double cycles_per_byte = (double)total_cycles / total_bytes;

	printf("����100����������: %llu\n", total_cycles);
	printf("�ܴ���������: %zu bytes\n", total_bytes);
	printf("����ָ��: %.2f cycles/byte\n", cycles_per_byte);

	printf("\n");
	// ��������100�ε���������
	uint64_t start_cycles1 = rdtsc();
	for (int i = 0; i < 100; i++) {
		SM4(input, input, rk, 0);
	}
	uint64_t end_cycles1 = rdtsc();
	uint64_t total_cycles1 = end_cycles1 - start_cycles1;

	// �����ܴ����������(bytes)
	size_t total_bytes1 = 100 * 16; // 100�ν��ܣ�ÿ��16�ֽ�

	// ����cycles/byte
	double cycles_per_byte1 = (double)total_cycles1 / total_bytes1;

	printf("����100����������: %llu\n", total_cycles1);
	printf("�ܴ���������: %zu bytes\n", total_bytes1);
	printf("����ָ��: %.2f cycles/byte\n", cycles_per_byte1);

	return 0;
}