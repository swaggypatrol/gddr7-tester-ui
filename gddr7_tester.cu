#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef DEFAULT_FRACTION
#define DEFAULT_FRACTION 0.80
#endif

#define CUDA_CHECK(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)

__device__ __forceinline__ uint32_t mix32(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<uint32_t>(x ^ (x >> 32));
}

// ====== Access order mapping (ensures a permutation) ======
struct MapParams {
  unsigned long long step2; // stride-64KiB step (coprime with n_vec)
  unsigned long long step3; // stride-128KiB step (coprime with n_vec)
  unsigned long long step5; // multiplicative permutation step (coprime with n_vec)
  unsigned int block;       // block-xor size in uint4 lanes, recommend 256 (4KiB)
};

__device__ __forceinline__
size_t map_index_perm(size_t i, size_t n_vec, int mode, const MapParams p)
{
  if (mode == 1) {
    // linear: natural order
    return i;
  } else if (mode == 2) {
    // stride ~64KiB: m = i * step (mod n); step coprime with n => permutation
    return (i * p.step2) % n_vec;
  } else if (mode == 3) {
    // stride ~128KiB
    return (i * p.step3) % n_vec;
  } else if (mode == 4) {
    // block-xor (4KiB): flip bits within full blocks; tail blocks use identity mapping
    const unsigned long long B = p.block;
    unsigned long long full = (n_vec / B) * B;  // coverage of full blocks
    if (i < full) {
      unsigned long long g = i / B, r = i % B;
      unsigned long long r2 = r ^ (B >> 1);     // swap half-block
      return g * B + r2;
    } else {
      return i; // tail block: unchanged to preserve one-to-one mapping
    }
  } else {
    // multiplicative permutation (pseudo-random): odd step5 coprime with n
    return (i * p.step5) % n_vec;
  }
}

// ====== Vectorized kernels ======
__global__ void init_pattern_vec(uint4* __restrict__ data4, size_t n_vec, uint32_t seed) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = tid; i < n_vec; i += stride) {
    size_t base = i * 4ULL;
    uint4 v;
    v.x = mix32((uint64_t)(base + 0) ^ seed);
    v.y = mix32((uint64_t)(base + 1) ^ seed);
    v.z = mix32((uint64_t)(base + 2) ^ seed);
    v.w = mix32((uint64_t)(base + 3) ^ seed);
    data4[i] = v;
  }
}

__global__ void verify_and_flip_vec(uint4* __restrict__ data4, size_t n_vec,
                                    uint32_t expect_seed, uint32_t next_seed,
                                    unsigned long long* __restrict__ errcount,
                                    int mode, MapParams mp) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;

  for (size_t i = tid; i < n_vec; i += stride) {
    size_t m = map_index_perm(i, n_vec, mode, mp);  // only changes access order; data locations unchanged
    size_t base = m * 4ULL;
    uint4 v = data4[m];

    uint32_t e0 = mix32((uint64_t)(base + 0) ^ expect_seed);
    uint32_t e1 = mix32((uint64_t)(base + 1) ^ expect_seed);
    uint32_t e2 = mix32((uint64_t)(base + 2) ^ expect_seed);
    uint32_t e3 = mix32((uint64_t)(base + 3) ^ expect_seed);

    if (v.x != e0) atomicAdd(errcount, 1ULL);
    if (v.y != e1) atomicAdd(errcount, 1ULL);
    if (v.z != e2) atomicAdd(errcount, 1ULL);
    if (v.w != e3) atomicAdd(errcount, 1ULL);

    v.x = mix32((uint64_t)(base + 0) ^ next_seed);
    v.y = mix32((uint64_t)(base + 1) ^ next_seed);
    v.z = mix32((uint64_t)(base + 2) ^ next_seed);
    v.w = mix32((uint64_t)(base + 3) ^ next_seed);

    data4[m] = v;
  }
}

// ====== Host: pick coprime steps ======
static inline unsigned long long gcd_ull(unsigned long long a, unsigned long long b) {
  while (b) { unsigned long long t = a % b; a = b; b = t; }
  return a;
}

static inline unsigned long long pick_coprime_step(unsigned long long n, unsigned long long target) {
  if (n <= 1) return 1;
  if (target == 0) target = 1;
  // force odd and search near target for a value coprime with n
  if ((target & 1ULL) == 0) target += 1;
  unsigned long long s = target, delta = 0;
  for (unsigned tries = 0; tries < 100000; ++tries) {
    if (gcd_ull(n, s) == 1ULL) return s;
    // alternate +2 / -2 while expanding outward
    delta += 2;
    unsigned long long up = s + delta;
    if (up > 1 && gcd_ull(n, up) == 1ULL) return up;
    if (s > delta) {
      unsigned long long dn = s - delta;
      if (gcd_ull(n, dn) == 1ULL) return dn;
    }
  }
  // fall back to 1 if no coprime found (extreme case)
  return 1ULL;
}

static inline double toGiB(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char* argv[]) {
  double fraction = (argc >= 2) ? std::atof(argv[1]) : DEFAULT_FRACTION;
  if (fraction <= 0.0 || fraction > 0.90) fraction = DEFAULT_FRACTION;
  int chunk_iters = (argc >= 3) ? std::atoi(argv[2]) : 100;
  if (chunk_iters <= 0) chunk_iters = 100;

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  size_t freeB = 0, totalB = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

  size_t targetB = (size_t)(freeB * fraction);
  size_t n_elems = targetB / sizeof(uint32_t);
  n_elems -= (n_elems & 3ULL); // align to 16B
  targetB  = n_elems * sizeof(uint32_t);
  size_t n_vec = n_elems / 4;

  int eccEnabled = 0;
  if (cudaDeviceGetAttribute(&eccEnabled, cudaDevAttrEccEnabled, 0) != cudaSuccess) eccEnabled = 0;

  printf("Device: %s (cc %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  printf("Global Mem: %.2f GiB free / %.2f GiB total\n", toGiB(freeB), toGiB(totalB));
  printf("ECC enabled: %s\n", eccEnabled ? "Yes" : "No");
  printf("Allocated/Tested: %.2f GiB (fraction=%.2f)\n", toGiB(targetB), fraction);
  if (n_vec == 0) { printf("Nothing to test.\n"); return 0; }

  // choose steps that are coprime with n
  MapParams mp{};
  mp.block = 256; // 4KiB per block (256 * 16B)
  mp.step2 = pick_coprime_step(n_vec, (64ULL * 1024) / 16ULL);   // ~64KiB / 16B
  mp.step3 = pick_coprime_step(n_vec, (128ULL * 1024) / 16ULL);  // ~128KiB / 16B
  mp.step5 = pick_coprime_step(n_vec, 2654435761ULL);            // coprime neighbor of permutation constant
  printf("Permutation steps: mode2=%llu, mode3=%llu, mode5=%llu (n_vec=%llu)\n",
         (unsigned long long)mp.step2, (unsigned long long)mp.step3, (unsigned long long)mp.step5,
         (unsigned long long)n_vec);

  uint32_t* d_data = nullptr;
  unsigned long long* d_err = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, targetB));
  CUDA_CHECK(cudaMalloc(&d_err, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_err, 0, sizeof(unsigned long long)));

  const int threads = 256;
  int blocks = prop.multiProcessorCount * 32;
  if (blocks > 65535) blocks = 65535;

  const uint32_t seedA = 0x12345678u;
  const uint32_t seedB = 0xDEADBEEFu;

  printf("Initializing pattern A...\n");
  init_pattern_vec<<<blocks, threads>>>((uint4*)d_data, n_vec, seedA);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));

  printf("Loop forever. Modes cycle as 1→2→3→4→5\n");
  const double tested_bytes = (double)(n_vec * sizeof(uint4));
  unsigned long long prev_err = 0;
  const int MODE_CNT = 5;

  for (unsigned long long chunk = 1;; ++chunk) {
    int mode = (int)((chunk - 1) % MODE_CNT) + 1;

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < chunk_iters; ++i) {
      verify_and_flip_vec<<<blocks, threads>>>((uint4*)d_data, n_vec, seedA, seedB, d_err, mode, mp);
      CUDA_CHECK(cudaGetLastError());
      verify_and_flip_vec<<<blocks, threads>>>((uint4*)d_data, n_vec, seedB, seedA, d_err, mode, mp);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    unsigned long long total_err = 0;
    CUDA_CHECK(cudaMemcpy(&total_err, d_err, sizeof(total_err), cudaMemcpyDeviceToHost));
    unsigned long long delta_err = total_err - prev_err;
    prev_err = total_err;

    const double bytes_processed = (double)chunk_iters * 2.0 * 2.0 * tested_bytes;
    const double gbps = (bytes_processed / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    printf("[Chunk %llu | Mode %d] Time: %.2f ms | Bandwidth: %.2f GB/s | New errors: %llu | Total errors: %llu\n",
           chunk, mode, ms, gbps, delta_err, total_err);
    fflush(stdout);
  }

  return 0;
}
