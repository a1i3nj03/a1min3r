/***************************************************************************************************
* SIMD512 SM3+ CUDA IMPLEMENTATION (require cuda_x11_simd512_func.cuh)
*** Based on Alexis78's very good simd modifications ***
*/

#include <stdio.h>
#include <intrin.h>
#include "miner.h"
#include "cuda_helper_alexis.h"
//#include "cuda_helper.h"
#include "cuda_vectors_alexis.h"

#if 0
#define TPB 128//128U//96U//128
#define LSB	2//5
#define TPB_FINAL	128//256U
#define LSB_FINAL	2//1//1
#define LSB_1		2//2//2//1
#define LSB_2		4//2//2//1

#define TPB52_1		192//128
#define TPB52_2		256//128

#define TPB_INI1		128
#define LSB_INI1		8
#define TPB_INI2		128
#define LSB_INI2		2//8//4

#define TPB50_1		128
#define TPB50_2		128
#else
#define TPB 128//128U//96U//128
#define LSB	6//5
#define TPB_FINAL	192//256U
#define LSB_FINAL	2//1//1
#define LSB_1		3//2//2//1
#define LSB_2		4//2//2//1

#define TPB52_1		128//128
#define TPB52_2		128//128

#define TPB_INI1	128
#define LSB_INI1	8//4//8
#define TPB_INI2	128
#define LSB_INI2	2//8//4

#define TPB50_1		128
#define TPB50_2		128
#endif

#define DYNAMIC_RADIX
#ifdef DYNAMIC_RADIX
#define SIMD_RADIX	64U//128U//80U//128U//64U // for reducing memory needed for SIMD
uint16_t gpu_radix[MAX_GPUS];
#else
#define SIMD_RADIX	64U//128U//80U//128U//64U // for reducing memory needed for SIMD
#endif
//uint32_t *d_state[MAX_GPUS];
extern uint4 *d_temp4[MAX_GPUS];

#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>
// texture bound to d_temp4[thr_id], for read access in Compaction kernel
//texture<uint4, 1, cudaReadModeElementType> texRef1D_128; //*************
texture<uint4, 1, cudaReadModeElementType> texRef1D_128;

__align__(64) __constant__ static uint32_t c_simd[8]; // before #include

#include "alt_simd_fn.cuh"

// texture bound to d_temp4[thr_id], for read access in Compaction kernel
//texture<uint4, 1, cudaReadModeElementType> texRef1D_128;


/***************************************************/
__global__
__launch_bounds__(TPB52_2, LSB_1)
static void x11_simd512_gpu_compress_64_maxwell(uint32_t threads, uint32_t *g_hash, const uint4 *const __restrict__ g_fft4, volatile int *order)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thr_offset = thread << 6;
	uint32_t IV[32];
#ifdef A1MIN3R_MOD
	if (*order) { return; }
#endif
	if (thread < threads) 
	{

		uint32_t *Hash = &g_hash[thread << 4];
		//		Compression1(Hash, thread, g_fft4, g_state);
		uint32_t A[32];
#if __CUDA_ARCH__ >= 750
		* (uint2x4*)&A[0] = __ldg4((uint2x4*)&Hash[0]) ^ __ldg4((uint2x4*)&c_IV_512[0]);
		*(uint2x4*)&A[8] = __ldg4((uint2x4*)&Hash[8]) ^ __ldg4((uint2x4*)&c_IV_512[8]);
		*(uint2x4*)&A[16] = __ldg4((uint2x4*)&c_IV_512[16]);
		*(uint2x4*)&A[24] = __ldg4((uint2x4*)&c_IV_512[24]);

		Round8(A, thr_offset, g_fft4);

		STEP8_IF_IV(0, 32, 4, 13, &A[0], &A[8], &A[16], &A[24]);
		STEP8_IF_IV(8, 33, 13, 10, &A[24], &A[0], &A[8], &A[16]);
		STEP8_IF_IV(16, 34, 10, 25, &A[16], &A[24], &A[0], &A[8]);
		STEP8_IF_IV(24, 35, 25, 4, &A[8], &A[16], &A[24], &A[0]);
#else
		*(uint2x4*)&IV[0] = __ldg4((uint2x4*)&c_IV_512[0]);
		*(uint2x4*)&IV[8] = __ldg4((uint2x4*)&c_IV_512[8]);
		*(uint2x4*)&IV[16] = __ldg4((uint2x4*)&c_IV_512[16]);
		*(uint2x4*)&IV[24] = __ldg4((uint2x4*)&c_IV_512[24]);

		*(uint2x4*)&A[0] = __ldg4((uint2x4*)&Hash[0]);
		*(uint2x4*)&A[8] = __ldg4((uint2x4*)&Hash[8]);

#pragma unroll 16
		for (uint32_t i = 0; i<16; i++)
			A[i] = A[i] ^ IV[i];

#pragma unroll 16
		for (uint32_t i = 16; i<32; i++)
			A[i] = IV[i];

		Round8(A, thr_offset, g_fft4);

		STEP8_IF(&IV[0], 32, 4, 13, &A[0], &A[8], &A[16], &A[24]);
		STEP8_IF(&IV[8], 33, 13, 10, &A[24], &A[0], &A[8], &A[16]);
		STEP8_IF(&IV[16], 34, 10, 25, &A[16], &A[24], &A[0], &A[8]);
		STEP8_IF(&IV[24], 35, 25, 4, &A[8], &A[16], &A[24], &A[0]);
#endif
#pragma unroll 32
		for (uint32_t i = 0; i<32; i++) {
			IV[i] = A[i];
		}

		A[0] ^= 512;

		Round8_0_final(A, 3, 23, 17, 27);
		Round8_1_final(A, 28, 19, 22, 7);
		Round8_2_final(A, 29, 9, 15, 5);
		Round8_3_final(A, 4, 13, 10, 25);
		STEP8_IF(&IV[0], 32, 4, 13, &A[0], &A[8], &A[16], &A[24]);
		STEP8_IF(&IV[8], 33, 13, 10, &A[24], &A[0], &A[8], &A[16]);
		STEP8_IF(&IV[16], 34, 10, 25, &A[16], &A[24], &A[0], &A[8]);
		STEP8_IF(&IV[24], 35, 25, 4, &A[8], &A[16], &A[24], &A[0]);

		*(uint2x4*)&Hash[0] = *(uint2x4*)&A[0];
		*(uint2x4*)&Hash[8] = *(uint2x4*)&A[8];
	}
}

__device__ __forceinline__
static void SIMD_Compress(uint32_t *A, const uint32_t thr_offset, const uint4 *const __restrict__ g_fft4) {

	uint32_t IV[32];

	*(uint2x4*)&IV[0] = *(uint2x4*)&c_IV_512[0];
	*(uint2x4*)&IV[8] = *(uint2x4*)&c_IV_512[8];
	*(uint2x4*)&IV[16] = *(uint2x4*)&c_IV_512[16];
	*(uint2x4*)&IV[24] = *(uint2x4*)&c_IV_512[24];

	Round8(A, thr_offset, g_fft4);

	const uint32_t a[4] = { 4, 13, 10, 25 };

	for (int i = 0; i<4; i++)
		STEP8_IF(&IV[i * 8], 32 + i, a[i], a[(i + 1) & 3], &A[(0 + i * 24) & 31], &A[(8 + i * 24) & 31], &A[(16 + i * 24) & 31], &A[(24 + i * 24) & 31]);

#pragma unroll 32
	for (uint32_t i = 0; i<32; i++) {
		IV[i] = A[i];
	}

	A[0] ^= 512;

	Round8_0_final(A, 3, 23, 17, 27);
	Round8_1_final(A, 28, 19, 22, 7);
	Round8_2_final(A, 29, 9, 15, 5);
	Round8_3_final(A, 4, 13, 10, 25);

	for (int i = 0; i<4; i++)
		STEP8_IF(&IV[i * 8], 32 + i, a[i], a[(i + 1) & 3], &A[(0 + i * 24) & 31], &A[(8 + i * 24) & 31], &A[(16 + i * 24) & 31], &A[(24 + i * 24) & 31]);

}

__host__
void alt_x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, volatile int *order)
{
#ifdef DYNAMIC_RADIX
	uint32_t rem = threads % gpu_radix[thr_id]; // if using non powers of 2.
	threads /= gpu_radix[thr_id];
#else
	uint32_t rem = threads % SIMD_RADIX; // if using non powers of 2.
	threads /= SIMD_RADIX;
#endif
	//	dim3 gridX8(grid.x * 8U);
	uint32_t tpb = TPB52_1;
	const dim3 grid1((8U * threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

	tpb = TPB52_2;
	const dim3 grid2((threads + tpb - 1) / tpb);
	const dim3 block2(tpb);

#ifdef DYNAMIC_RADIX
	for (uint32_t i = 0; i < gpu_radix[thr_id]; i++)
#else
	for (uint32_t i = 0; i < SIMD_RADIX; i++)
#endif
	{
		uint64_t offset = threads * i << 6;
		uint8_t *tmp_hash;
		tmp_hash = (uint8_t*)d_hash; tmp_hash += offset;

		x11_simd512_gpu_expand_64 << <grid1, block1 >> > (threads, (uint32_t*)tmp_hash, d_temp4[thr_id], order);
		x11_simd512_gpu_compress_64_maxwell << < grid2, block2 >> > (threads, (uint32_t*)tmp_hash, d_temp4[thr_id], order);
	}
#if 1
	if (rem)
	{
		threads = rem;
#ifdef DYNAMIC_RADIX
		uint64_t offset = threads * gpu_radix[thr_id] << 6;
#else
		uint64_t offset = threads * SIMD_RADIX << 6;
#endif
		uint8_t *tmp_hash;
		tmp_hash = (uint8_t*)d_hash; tmp_hash += offset;

		x11_simd512_gpu_expand_64 << <grid1, block1 >> > (threads, (uint32_t*)tmp_hash, d_temp4[thr_id], order);
		x11_simd512_gpu_compress_64_maxwell << < grid2, block2 >> > (threads, (uint32_t*)tmp_hash, d_temp4[thr_id], order);
	}
#endif
}

__host__
int alt_x11_simd512_cpu_init(int thr_id, uint32_t threads)
{
	//	int dev_id = device_map[thr_id];
	// cuda_get_arch(thr_id); // should be already done!
	//	if (device_sm[dev_id] < 300 || cuda_arch[dev_id] < 300) {
	//		x11_simd512_cpu_init_sm2(thr_id);
	//		return 0;
	//	}
	//2097152
#if 0
	if (threads > 2097152)
	{
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_temp4[thr_id], 32 * sizeof(uint4)*(threads >> 1)), (int)err); /* todo: prevent -i 21 */
		CUDA_CALL_OR_RET_X(cudaMalloc((&d_temp4[thr_id]) + 32 * (threads >> 1), 32 * sizeof(uint4)*(threads >> 1)), (int)err); /* todo: prevent -i 21 */
	}
	else
#endif
#ifdef DYNAMIC_RADIX

	unsigned long lz;

	if (!_BitScanReverse(&lz, threads))
	{
		lz = 0;
	}
	switch (lz)//(threads)
	{
	case 27:
		gpu_radix[thr_id] = 256;
		break;
	case 26:
		gpu_radix[thr_id] = 256;//180;//180;//170;//160;//192;
		break;
	case 25:
		gpu_radix[thr_id] = 64;//32;//32;//64;//128;//
		break;
	case 24:
		gpu_radix[thr_id] = 16;//32;//64;
		break;
	case 23:
		gpu_radix[thr_id] = 8;//16//42;
		break;
	case 22:
		gpu_radix[thr_id] = 4;//4;//32;
		break;
	case 21:
		gpu_radix[thr_id] = 2;//2;//4;
		break;
	case 20:
		gpu_radix[thr_id] = 1;//4;
		break;
	case 19:
		gpu_radix[thr_id] = 1;//4;
		break;
	case 1 << 18:
	case 1 << 17:
	case 1 << 16:
	case 1 << 15:
	case 1 << 14:
	case 1 << 13:
	case 1 << 12:
		gpu_radix[thr_id] = 1;
	default:
		gpu_radix[thr_id] = 256;
		break;
	}

	CUDA_CALL_OR_RET_X(cudaMalloc(&d_temp4[thr_id], 64 * sizeof(uint4) * (threads / gpu_radix[thr_id])), (int)err); /* todo: prevent -i 21 */

#if defined(__x86_64__) || defined(_WIN64) || defined(__aarch64__)
	if (64 * sizeof(uint4) * (threads / gpu_radix[thr_id]) > UINT32_MAX)
#elif defined(__amd64__) || defined(__amd64) || defined(_M_X64) || defined(_M_IA64)
	if (64 * sizeof(uint4) * (threads / gpu_radix[thr_id]) > UINT32_MAX)
#else
	if (64 * sizeof(uint4) * (threads / gpu_radix[thr_id]) > UINT32_MAX)//INT32_MAX)
#endif
	{
		gpulog(LOG_ERR, thr_id, "Error: cannot allocate > INT32_MAX");
		return -1;
	}

#else
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_temp4[thr_id], 64 * sizeof(uint4) * threads / SIMD_RADIX), (int)err); /* todo: prevent -i 21 */
#endif // DYNAMIC_RADIX

#if 0

																											   //CUDA_CALL_OR_RET_X(cudaMalloc(&d_temp4[thr_id], 64 * sizeof(uint4)*threads), (int)err); /* todo: prevent -i 21 */
	CUDA_CALL_OR_RET_X(cudaMalloc(&d_state[thr_id], 32 * sizeof(int) * threads / SIMD_RADIX), (int)err);

#ifndef DEVICE_DIRECT_CONSTANTS
	cudaMemcpyToSymbol(c_perm, h_perm, sizeof(h_perm), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_IV_512, h_IV_512, sizeof(h_IV_512), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_FFT128_8_16_Twiddle, h_FFT128_8_16_Twiddle, sizeof(h_FFT128_8_16_Twiddle), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_FFT256_2_128_Twiddle, h_FFT256_2_128_Twiddle, sizeof(h_FFT256_2_128_Twiddle), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(d_cw0, h_cw0, sizeof(h_cw0), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_cw1, h_cw1, sizeof(h_cw1), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_cw2, h_cw2, sizeof(h_cw2), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_cw3, h_cw3, sizeof(h_cw3), 0, cudaMemcpyHostToDevice);
#endif
#endif
	// Texture for 128-Bit Zugriffe
	cudaChannelFormatDesc channelDesc128 = cudaCreateChannelDesc<uint4>();
	texRef1D_128.normalized = 0;
	texRef1D_128.filterMode = cudaFilterModePoint;
	texRef1D_128.addressMode[0] = cudaAddressModeClamp;

	CUDA_CALL_OR_RET_X(cudaBindTexture(NULL, &texRef1D_128, d_temp4[thr_id], &channelDesc128, 64 * sizeof(uint4) * (threads / gpu_radix[thr_id])), (int)err);

	return 0;
}

__host__
void alt_x11_simd512_cpu_free(int thr_id)
{
	//	int dev_id = device_map[thr_id];
	//	if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300) {
	cudaFree(d_temp4[thr_id]);
	//		cudaFree(d_state[thr_id]);
	//	}
}
