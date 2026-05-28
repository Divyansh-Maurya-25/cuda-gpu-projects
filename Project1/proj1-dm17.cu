/* ==================================================================
   SDH_cuda.cu
   CUDA implementation of Spatial Distance Histogram (SDH)
   Mirrors the provided CPU (SDH_base.cu) behavior.
 
Divyansh Maurya
U11865935
   ==================================================================
*/




#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#define BOX_SIZE 23000.0        /* box side length */
#define SQRT3 1.7320508075688772

typedef struct {
    double x_pos;
    double y_pos;
    double z_pos;
} atom;

typedef struct {
    long long d_cnt;   /* counts (64-bit signed on host/device struct) */
} bucket;

/* CUDA error-check macro */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error %s:%d: '%s'\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Compute Euclidean distance (host helper) */
static inline double p2p_distance_host(const atom *list, long long i, long long j) {
    double dx = list[i].x_pos - list[j].x_pos;
    double dy = list[i].y_pos - list[j].y_pos;
    double dz = list[i].z_pos - list[j].z_pos;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

/* CPU baseline: compute histogram by counting each unordered pair once */
int PDH_baseline(bucket *hist, atom *alist, long long acnt, double res, int num_buckets) {
    long long i, j;
    for (i = 0; i < acnt; ++i) {
        for (j = i + 1; j < acnt; ++j) {
            double dist = p2p_distance_host(alist, i, j);
            int h_pos = (int)(dist / res);
            if (h_pos >= 0 && h_pos < num_buckets) hist[h_pos].d_cnt++;
        }
    }
    return 0;
}

/* Print histogram in same format as provided in instructions */
void output_histogram_host(bucket *hist, int num_buckets) {
    int i;
    long long total_cnt = 0;
    for (i = 0; i < num_buckets; i++) {
        if (i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15lld ", hist[i].d_cnt);
        total_cnt += hist[i].d_cnt;
        if (i == num_buckets - 1)
            printf("\n T:%lld \n", total_cnt);
        else
            printf("| ");
    }
}

/* Compare histograms and print difference in the required format */
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

/* GPU kernel: each thread processes one point i and compares with j>i */
__global__ void sdh_kernel(const atom *d_atoms, bucket *d_hist,
                           long long N, double res, int num_buckets) {
    long long i = (long long)blockIdx.x * blockDim.x + (long long)threadIdx.x;
    if (i >= N) return;

    double xi = d_atoms[i].x_pos;
    double yi = d_atoms[i].y_pos;
    double zi = d_atoms[i].z_pos;

    for (long long j = i + 1; j < N; ++j) {
        double dx = xi - d_atoms[j].x_pos;
        double dy = yi - d_atoms[j].y_pos;
        double dz = zi - d_atoms[j].z_pos;
        double dist = sqrt(dx*dx + dy*dy + dz*dz);
        int h_pos = (int)(dist / res);
        if (h_pos >= 0 && h_pos < num_buckets) {
            atomicAdd((unsigned long long *)&(d_hist[h_pos].d_cnt), (unsigned long long)1ULL);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <num_points> <bucket_width_w>\n", argv[0]);
        return 1;
    }

    long long PDH_acnt = atoll(argv[1]);
    double PDH_res = atof(argv[2]);
    if (PDH_acnt <= 0 || PDH_res <= 0.0) {
        fprintf(stderr, "Invalid arguments: num_points and bucket_width must be positive.\n");
        return 1;
    }

    int num_buckets = (int)(BOX_SIZE * SQRT3 / PDH_res) + 1;

    bucket *hist_cpu = (bucket *)malloc(sizeof(bucket) * num_buckets);
    bucket *hist_gpu  = (bucket *)malloc(sizeof(bucket) * num_buckets);
    atom *atom_list = (atom *)malloc(sizeof(atom) * PDH_acnt);

    for (int i = 0; i < num_buckets; ++i) {
        hist_cpu[i].d_cnt = 0;
        hist_gpu[i].d_cnt = 0;
    }

    srand(1);
    for (long long i = 0; i < PDH_acnt; ++i) {
        atom_list[i].x_pos = ((double)rand() / (double)RAND_MAX) * BOX_SIZE;
        atom_list[i].y_pos = ((double)rand() / (double)RAND_MAX) * BOX_SIZE;
        atom_list[i].z_pos = ((double)rand() / (double)RAND_MAX) * BOX_SIZE;
    }

    /* CPU baseline */
    PDH_baseline(hist_cpu, atom_list, PDH_acnt, PDH_res, num_buckets);

    printf("\n--- CPU histogram ---\n");
    output_histogram_host(hist_cpu, num_buckets);

    /* GPU setup */
    atom *d_atoms = NULL;
    bucket *d_hist = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_atoms, (size_t)PDH_acnt * sizeof(atom)));
    CUDA_CHECK(cudaMalloc((void **)&d_hist, (size_t)num_buckets * sizeof(bucket)));
    CUDA_CHECK(cudaMemcpy(d_atoms, atom_list, (size_t)PDH_acnt * sizeof(atom), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, (size_t)num_buckets * sizeof(bucket)));

    int threadsPerBlock = 128;
    long long blocks_ll = (PDH_acnt + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks_ll <= 0) blocks_ll = 1;
    if (blocks_ll > INT_MAX) {
        fprintf(stderr, "Error: too many blocks required.\n");
        return 1;
    }
    int blocks = (int)blocks_ll;

    sdh_kernel<<<blocks, threadsPerBlock>>>(d_atoms, d_hist, PDH_acnt, PDH_res, num_buckets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hist_gpu, d_hist, (size_t)num_buckets * sizeof(bucket), cudaMemcpyDeviceToHost));

    printf("\n--- GPU histogram ---\n");
    output_histogram_host(hist_gpu, num_buckets);

    printf("\n--- Difference histogram (CPU - GPU) ---\n");
    compare_histograms(hist_cpu, hist_gpu, num_buckets);

    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_atoms));
    free(atom_list);
    free(hist_cpu);
    free(hist_gpu);

    return 0;
}
