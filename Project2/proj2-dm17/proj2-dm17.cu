/*
  Project2 - GPU SDH Implementation
  Divyansh Maurya (U11865935)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

#define BOX_SIZE 23000.0
#define SQRT3 1.7320508075688772

struct bucket {
    unsigned long long d_cnt;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct atom {
    double x, y, z;
};

__host__ __device__ inline double distance3d(double ax, double ay, double az,
                                             double bx, double by, double bz) {
    double dx = ax - bx;
    double dy = ay - by;
    double dz = az - bz;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

void compute_SDH_CPU(bucket* hist, const atom* atoms, long long N,
                     double bucket_width, int num_bins) {
    for (long long i = 0; i < N; i++) {
        for (long long j = i + 1; j < N; j++) {
            double dist = distance3d(atoms[i].x, atoms[i].y, atoms[i].z,
                                     atoms[j].x, atoms[j].y, atoms[j].z);
            int bin = (int)(dist / bucket_width);
            if ((unsigned)bin < (unsigned)num_bins)
                hist[bin].d_cnt++;
        }
    }
}

__global__ void sdh_gpu_kernel(const double* x_coords, const double* y_coords,
                               const double* z_coords,
                               unsigned long long* output_hist,
                               long long N, double bucket_width,
                               int num_bins, int padded_bins,
                               int num_copies) {
    extern __shared__ unsigned char shared_mem[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int warp_lane = tid & 31;

    double* shared_x = (double*)shared_mem;
    double* shared_y = shared_x + block_size;
    double* shared_z = shared_y + block_size;
    unsigned int* shared_hist = (unsigned int*)(shared_z + block_size);

    for (int i = tid; i < num_copies * padded_bins; i += block_size)
        shared_hist[i] = 0u;
    __syncthreads();

    long long idx = (long long)blockIdx.x * block_size + tid;
    bool valid = (idx < N);
    double myx = 0, myy = 0, myz = 0;

    if (valid) {
#if __CUDA_ARCH__ >= 350
        myx = __ldg(&x_coords[idx]);
        myy = __ldg(&y_coords[idx]);
        myz = __ldg(&z_coords[idx]);
#else
        myx = x_coords[idx];
        myy = y_coords[idx];
        myz = z_coords[idx];
#endif
    }

    double inv_width = 1.0 / bucket_width;
    int total_blocks = (N + block_size - 1) / block_size;

    for (int other_block = blockIdx.x + 1; other_block < total_blocks; ++other_block) {
        long long other_start = (long long)other_block * block_size;
        int points_in_block = min((long long)block_size, N - other_start);

        if (tid < points_in_block) {
#if __CUDA_ARCH__ >= 350
            shared_x[tid] = __ldg(&x_coords[other_start + tid]);
            shared_y[tid] = __ldg(&y_coords[other_start + tid]);
            shared_z[tid] = __ldg(&z_coords[other_start + tid]);
#else
            shared_x[tid] = x_coords[other_start + tid];
            shared_y[tid] = y_coords[other_start + tid];
            shared_z[tid] = z_coords[other_start + tid];
#endif
        }
        __syncthreads();

        if (valid) {
#pragma unroll 4
            for (int j = 0; j < points_in_block; j++) {
                double dist = distance3d(myx, myy, myz,
                                         shared_x[j], shared_y[j], shared_z[j]);
                int bin = (int)(dist * inv_width);
                if ((unsigned)bin < (unsigned)num_bins) {
                    int copy_idx = (num_copies == 1) ? 0 : (warp_lane % num_copies);
                    atomicAdd(&shared_hist[copy_idx * padded_bins + bin], 1u);
                }
            }
        }
        __syncthreads();
    }

    long long block_start = (long long)blockIdx.x * block_size;
    int pts_in_block = min((long long)block_size, N - block_start);

    if (tid < pts_in_block) {
        shared_x[tid] = myx;
        shared_y[tid] = myy;
        shared_z[tid] = myz;
    }
    __syncthreads();

    if (valid && tid < pts_in_block) {
        for (int offset = 1; offset < pts_in_block; ++offset) {
            int j = tid + offset;
            if (j >= pts_in_block) j -= pts_in_block;
            if (tid < j) {
                double dist = distance3d(myx, myy, myz,
                                         shared_x[j], shared_y[j], shared_z[j]);
                int bin = (int)(dist * inv_width);
                if ((unsigned)bin < (unsigned)num_bins) {
                    int copy_idx = (num_copies == 1) ? 0 : (warp_lane % num_copies);
                    atomicAdd(&shared_hist[copy_idx * padded_bins + bin], 1u);
                }
            }
        }
    }
    __syncthreads();

    for (int b = tid; b < num_bins; b += block_size) {
        unsigned long long sum = 0ULL;
        for (int c = 0; c < num_copies; c++)
            sum += (unsigned long long)shared_hist[c * padded_bins + b];
        if (sum) atomicAdd(&output_hist[b], sum);
    }
}

int get_num_copies(int bins) {
    if (bins <= 10) return 32;
    if (bins <= 35) return 16;
    if (bins <= 92) return 8;
    if (bins <= 152) return 4;
    if (bins <= 300) return 2;
    return 1;
}



void output_histogram_host(bucket *hist, int num_buckets) {
    int i;
    long long total_cnt = 0;
    for (i = 0; i < num_buckets; i++) {
        if (i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15llu ", (unsigned long long)hist[i].d_cnt);
        total_cnt += hist[i].d_cnt;
        if (i == num_buckets - 1)
            printf("\n T:%lld \n", total_cnt);
        else
            printf("| ");
    }
}

void compare_histograms(bucket *cpu_hist, bucket *gpu_hist, int num_buckets) {
    printf("\nDifference between CPU and GPU histograms:\n");
    for (int i = 0; i < num_buckets; i++) {
        long long diff = cpu_hist[i].d_cnt - gpu_hist[i].d_cnt;
        if (i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15lld ", diff);
        if (i != num_buckets - 1)
            printf("| ");
    }
    printf("\n");
}

/* ----------------------------------------------------------- */

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <num_points> <bucket_width> <block_size>\n", argv[0]);
        return 1;
    }

    long long N = atoll(argv[1]);
    double W = atof(argv[2]);
    int BLOCK_SIZE = atoi(argv[3]);
    if (N <= 0 || W <= 0 || BLOCK_SIZE <= 0 || BLOCK_SIZE > 1024) {
        fprintf(stderr, "Invalid arguments\n");
        return 1;
    }

    int NUM_BINS = (int)(BOX_SIZE * SQRT3 / W) + 1;
    int PADDED_BINS = ((NUM_BINS + 31) / 32) * 32;

    bucket* cpu_hist = (bucket*)calloc(NUM_BINS, sizeof(bucket));
    bucket* gpu_hist = (bucket*)calloc(NUM_BINS, sizeof(bucket));
    atom* atoms = (atom*)malloc(sizeof(atom) * N);
    if (!cpu_hist || !gpu_hist || !atoms) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    srand(1);
    for (long long i = 0; i < N; i++) {
        atoms[i].x = ((double)rand() / RAND_MAX) * BOX_SIZE;
        atoms[i].y = ((double)rand() / RAND_MAX) * BOX_SIZE;
        atoms[i].z = ((double)rand() / RAND_MAX) * BOX_SIZE;
    }

    struct timeval start_cpu, end_cpu;
    gettimeofday(&start_cpu, NULL);
    compute_SDH_CPU(cpu_hist, atoms, N, W, NUM_BINS);
    gettimeofday(&end_cpu, NULL);
    double cpu_time = (end_cpu.tv_sec - start_cpu.tv_sec) + 
                      (end_cpu.tv_usec - start_cpu.tv_usec) / 1e6;
    printf("\n******** Total Running Time of CPU = %.5f sec *******\n", cpu_time);

    double *host_x = (double*)malloc(sizeof(double) * N);
    double *host_y = (double*)malloc(sizeof(double) * N);
    double *host_z = (double*)malloc(sizeof(double) * N);
    for (long long i = 0; i < N; i++) {
        host_x[i] = atoms[i].x;
        host_y[i] = atoms[i].y;
        host_z[i] = atoms[i].z;
    }

    double *dev_x, *dev_y, *dev_z;
    unsigned long long* dev_hist;
    CUDA_CHECK(cudaMalloc(&dev_x, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&dev_y, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&dev_z, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&dev_hist, sizeof(unsigned long long) * NUM_BINS));

    CUDA_CHECK(cudaMemcpy(dev_x, host_x, sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_y, host_y, sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_z, host_z, sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dev_hist, 0, sizeof(unsigned long long) * NUM_BINS));

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t max_shared = prop.sharedMemPerBlock;

    int num_copies = get_num_copies(NUM_BINS);
    size_t shared_needed = 3ULL * BLOCK_SIZE * sizeof(double)
        + (size_t)num_copies * (size_t)PADDED_BINS * sizeof(unsigned int);
    while (num_copies > 1 && shared_needed > max_shared) {
        num_copies >>= 1;
        shared_needed = 3ULL * BLOCK_SIZE * sizeof(double)
            + (size_t)num_copies * (size_t)PADDED_BINS * sizeof(unsigned int);
    }
    if (shared_needed > max_shared) {
        fprintf(stderr, "Error: Need %zu bytes shared memory but only %zu available\n",
                (size_t)shared_needed, (size_t)max_shared);
        return 1;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    sdh_gpu_kernel<<<num_blocks, BLOCK_SIZE, shared_needed>>>(
        dev_x, dev_y, dev_z, dev_hist, N, W, NUM_BINS, PADDED_BINS, num_copies);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    unsigned long long* temp = (unsigned long long*)malloc(sizeof(unsigned long long) * NUM_BINS);
    CUDA_CHECK(cudaMemcpy(temp, dev_hist, sizeof(unsigned long long) * NUM_BINS,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUM_BINS; i++) gpu_hist[i].d_cnt = temp[i];

    printf("\n--- CPU histogram ---\n");
    output_histogram_host(cpu_hist, NUM_BINS);

    printf("\n--- GPU histogram ---\n");
    output_histogram_host(gpu_hist, NUM_BINS);

    printf("\n--- Difference histogram (CPU - GPU) ---\n");
    compare_histograms(cpu_hist, gpu_hist, NUM_BINS);

    printf("\n******** Total Running Time of Kernel = %.5f sec *******\n", ms / 1000.0f);

    CUDA_CHECK(cudaFree(dev_x));
    CUDA_CHECK(cudaFree(dev_y));
    CUDA_CHECK(cudaFree(dev_z));
    CUDA_CHECK(cudaFree(dev_hist));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(temp);
    free(host_x);
    free(host_y);
    free(host_z);
    free(atoms);
    free(cpu_hist);
    free(gpu_hist);
    return 0;
}
