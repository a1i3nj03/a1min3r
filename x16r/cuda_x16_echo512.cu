/* echo512-80 cuda
* tpruvot 2018 - GPL code
* sp 2018
* random nonces in bench are invalid ...
*/

#include <stdio.h>
#include <memory.h>
#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define AESx(x) (x ##UL) /* SPH_C32(x) */

extern __device__ __device_builtin__ void __threadfence_block(void);
#define xor4_32(a,b,c,d) ((a ^ b) ^ (c ^ d));
#define INTENSIVE_GMF
#include "x11/cuda_x11_aes_sp.cuh"

__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c)
{
	asm("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a) : "r"(b), "r"(c));	// 0xEA = (F0 ^ CC) ^ AA
	return a;
}

__device__ static __forceinline__ void echo_round_sp(const uint32_t sharedMemory[1024 * 8], uint32_t *W, uint32_t &k0) {
	// Big Sub Words
#pragma unroll 16
	for (int idx = 0; idx < 16; idx++)
		AES_2ROUND_32(sharedMemory, W[(idx << 2) + 0], W[(idx << 2) + 1], W[(idx << 2) + 2], W[(idx << 2) + 3], k0);

	// Shift Rows
#pragma unroll 4
	for (int i = 0; i < 4; i++) {
		uint32_t t[4];
		/// 1, 5, 9, 13
		t[0] = W[i + 4];
		t[1] = W[i + 8];
		t[2] = W[i + 24];
		t[3] = W[i + 60];
		W[i + 4] = W[i + 20];
		W[i + 8] = W[i + 40];
		W[i + 24] = W[i + 56];
		W[i + 60] = W[i + 44];

		W[i + 20] = W[i + 36];
		W[i + 40] = t[1];
		W[i + 56] = t[2];
		W[i + 44] = W[i + 28];

		W[i + 28] = W[i + 12];
		W[i + 12] = t[3];
		W[i + 36] = W[i + 52];
		W[i + 52] = t[0];
	}
	// Mix Columns
#pragma unroll 4
	for (int i = 0; i < 4; i++) {
#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16) {
			uint32_t a[4];
			a[0] = W[idx + i];
			a[1] = W[idx + i + 4];
			a[2] = W[idx + i + 8];
			a[3] = W[idx + i + 12];

			uint32_t ab = a[0] ^ a[1];
			uint32_t bc = a[1] ^ a[2];
			uint32_t cd = a[2] ^ a[3];

			uint32_t t, t2, t3;
			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[idx + i] = (bc^ a[3] ^ abx);
			W[idx + i + 4] = xor3(a[0], cd, bcx);
			W[idx + i + 8] = xor3(ab, a[3], cdx);
			W[idx + i + 12] = xor3(ab, a[2], xor3(abx, bcx, cdx));
		}
	}
}
__device__ static __forceinline__ void echo_round_first_sp(const uint32_t sharedMemory[1024 * 8], uint32_t *W, uint32_t &k0)
{
	// Big Sub Words
#pragma unroll
	for (int idx = 8; idx < 13; idx++)
		AES_2ROUND_32(sharedMemory, W[(idx << 2) + 0], W[(idx << 2) + 1], W[(idx << 2) + 2], W[(idx << 2) + 3], k0);

	k0 += 3;

	// Shift Rows
#pragma unroll 4
	for (int i = 0; i < 4; i++) {
		uint32_t t[4];
		/// 1, 5, 9, 13
		t[0] = W[i + 4];
		t[1] = W[i + 8];
		t[2] = W[i + 24];
		t[3] = W[i + 60];
		W[i + 4] = W[i + 20];
		W[i + 8] = W[i + 40];
		W[i + 24] = W[i + 56];
		W[i + 60] = W[i + 44];

		W[i + 20] = W[i + 36];
		W[i + 40] = t[1];
		W[i + 56] = t[2];
		W[i + 44] = W[i + 28];

		W[i + 28] = W[i + 12];
		W[i + 12] = t[3];
		W[i + 36] = W[i + 52];
		W[i + 52] = t[0];
	}
	// Mix Columns
#pragma unroll 4
	for (int i = 0; i < 4; i++) {
#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16) {
			uint32_t a[4];
			a[0] = W[idx + i];
			a[1] = W[idx + i + 4];
			a[2] = W[idx + i + 8];
			a[3] = W[idx + i + 12];

			uint32_t ab = a[0] ^ a[1];
			uint32_t bc = a[1] ^ a[2];
			uint32_t cd = a[2] ^ a[3];

			uint32_t t, t2, t3;
			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[idx + i] = (bc^ a[3] ^ abx);
			W[idx + i + 4] = xor3(a[0], cd, bcx);
			W[idx + i + 8] = xor3(ab, a[3], cdx);
			W[idx + i + 12] = xor3(ab, a[2], xor3(abx, bcx, cdx));
		}
	}
}

__device__ __forceinline__
void cuda_echo_round_80(const uint32_t sharedMemory[1024 * 8], uint32_t *const __restrict__ data, const uint32_t nonce, uint32_t *hash)
{
	uint32_t W[64];

	const uint4 P[11] = { { 0xc2031f3a, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af },
	{ 0x428a9633, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af },
	{ 0xe2eaf6f3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af },
	{ 0xc9f3efc1, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af },
	{ 0x56869a2b, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af },
	{ 0x789c801f, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af },
	{ 0x81cbd7b1, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af },
	{ 0x4a7b67ca, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af },
	{ 0x83d3d3ab, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968 },
	{ 0x5d99993f, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751 },
	{ 0x57706cdc, 0xe4736c70, 0xf53fa165, 0xd6be2d00 } };


	W[51] = cuda_swab32(nonce);
	W[52] = 0x80;

	uint32_t k0 = 640; // bitlen
	uint4 *W4 = (uint4 *)&W[0];
#pragma unroll
	for (int i = 32; i<51; i++) W[i] = data[i - 32];

	W4[0] = P[0];
	W4[1] = P[1];
	W4[2] = P[2];
	W4[3] = P[3];
	W4[4] = P[4];
	W4[5] = P[5];
	W4[6] = P[6];
	W4[7] = P[7];

	W4[13] = P[8];
	W4[14] = P[9];
	W4[15] = P[10];

	k0 += 8;
	echo_round_first_sp(sharedMemory, W, k0);

	for (int i = 1; i < 10; i++)
		echo_round_sp(sharedMemory, W, k0);

	uint32_t Z[16];

	Z[0] = 512 ^ data[0] ^ W[0] ^ W[0 + 32];
	Z[1] = data[1] ^ W[1] ^ W[1 + 32];
	Z[2] = data[2] ^ W[2] ^ W[2 + 32];
	Z[3] = data[3] ^ W[3] ^ W[3 + 32];
	Z[4] = 512 ^ data[4] ^ W[4] ^ W[4 + 32];
	Z[5] = data[5] ^ W[5] ^ W[5 + 32];
	Z[6] = data[6] ^ W[6] ^ W[6 + 32];
	Z[7] = data[7] ^ W[7] ^ W[7 + 32];
	*(uint2x4*)&hash[0] = *(uint2x4*)&Z[0];

	Z[8] = 512 ^ data[8] ^ W[8] ^ W[8 + 32];
	Z[9] = data[9] ^ W[9] ^ W[9 + 32];
	Z[10] = data[10] ^ W[10] ^ W[10 + 32];
	Z[11] = data[11] ^ W[11] ^ W[11 + 32];
	Z[12] = 512 ^ data[12] ^ W[12] ^ W[12 + 32];
	Z[13] = data[13] ^ W[13] ^ W[13 + 32];
	Z[14] = data[14] ^ W[14] ^ W[14 + 32];
	Z[15] = data[15] ^ W[15] ^ W[15 + 32];

	*(uint2x4*)&hash[8] = *(uint2x4*)&Z[8];
}

__host__
void x16_echo512_cuda_init(int thr_id, const uint32_t threads)
{
	//	aes_cpu_init(thr_id);
}

__constant__ static uint32_t c_PaddedMessage80[20];

__host__
void x16_echo512_setBlock_80(int thr_id, void *endiandata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, endiandata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

__global__ __launch_bounds__(256, 3) /* will force 72 registers */
void x16_echo512_gpu_hash_80_sp(uint32_t threads, uint32_t startNonce, uint64_t *g_hash)
{
	__shared__  uint32_t sharedMemory[1024 * 8];

	aes_gpu_init256_32(sharedMemory);
//	__threadfence_block();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t hashPosition = thread;
		uint32_t *pHash = (uint32_t*)&g_hash[hashPosition << 3];

		__syncthreads();
		cuda_echo_round_80(sharedMemory, c_PaddedMessage80, startNonce + thread, pHash);
	}
}

__host__
void x16_echo512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x16_echo512_gpu_hash_80_sp << <grid, block >> >(threads, startNonce, (uint64_t*)d_hash);
}
