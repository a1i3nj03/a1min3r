#include "x11/cuda_x11.h"

//! alexis functions have order added for jump table
extern void x11_echo512_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash, volatile int *order);
extern void x11_luffa512_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash, volatile int *order);
extern void x11_shavite512_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash, volatile int *order);
extern void x13_fugue512_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash, volatile int *order);
extern void x14_shabal512_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash, volatile int *order);
extern void x13_hamsi512_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash, volatile int *order); ///! changed

extern void x13_hamsi512_cpu_init(int thr_id, uint32_t threads);
//extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
//extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_free(int thr_id);

extern void x14_shabal512_cpu_init(int thr_id, uint32_t threads);
//extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int flag);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, volatile int *order);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void x17_sha512_cpu_init(int thr_id, uint32_t threads);
extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, volatile int *order); // added order

extern void x17_haval256_cpu_init(int thr_id, uint32_t threads);
extern void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, const int outlen);

extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_outputHash, volatile int *order);

// ---- 80 bytes kernels

void quark_bmw512_cpu_setBlock_80(int thr_id, void *pdata);
void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void groestl512_setBlock_80(int thr_id, uint32_t *endiandata);
void groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void skein512_cpu_setBlock_80(int thr_id, void *pdata);
void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, volatile int *order);

extern void qubit_luffa512_cpu_setBlock_80_alexis(int thr_id, void *pdata);
extern void qubit_luffa512_cpu_hash_80_alexis(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash, volatile int *order);

void jh512_setBlock_80(int thr_id, uint32_t *endiandata);
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void keccak512_setBlock_80(int thr_id, uint32_t *endiandata);
void keccak512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void cubehash512_setBlock_80(int thr_id, uint32_t* endiandata);
void cubehash512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void x11_shavite512_setBlock_80(int thr_id, void *pdata);
void x11_shavite512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void x16_shabal512_setBlock_80(int thr_id, void *pdata);
void x16_shabal512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void x16_simd512_setBlock_80(int thr_id, void *pdata);
void x16_simd512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void x16_echo512_cuda_init(int thr_id, const uint32_t threads);
void x16_echo512_setBlock_80(int thr_id, void *pdata);
void x16_echo512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void x16_hamsi512_setBlock_80(int thr_id, void *pdata);
void x16_hamsi512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void x16_fugue512_cpu_init(int thr_id, uint32_t threads);
void x16_fugue512_cpu_free(int thr_id);
void x16_fugue512_setBlock_80(int thr_id, void *pdata);
void x16_fugue512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void x16_whirlpool512_init(int thr_id, uint32_t threads);
void x16_whirlpool512_setBlock_80(int thr_id, void* endiandata);
void x16_whirlpool512_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

void x16_sha512_setBlock_80(void *pdata);
void x16_sha512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash, volatile int *order);

