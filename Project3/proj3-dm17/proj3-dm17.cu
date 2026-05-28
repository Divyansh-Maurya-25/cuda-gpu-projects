/*
    Project 3: Parallel Bloom Filter with CUDA
    
    Author: Divyansh Maurya [COP 4520]
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <inttypes.h>
#include <string.h>

/* ======================================================================
   SIPHASH MACROS & CONSTANTS
   ====================================================================== */
#ifndef cROUNDS
#define cROUNDS 2
#endif
#ifndef dROUNDS
#define dROUNDS 4
#endif

#define ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))

// Host macros (from professor's code)
#define U32TO8_LE(p, v) \
    (p)[0] = (uint8_t)((v)); \
    (p)[1] = (uint8_t)((v) >> 8); \
    (p)[2] = (uint8_t)((v) >> 16); \
    (p)[3] = (uint8_t)((v) >> 24);

#define U64TO8_LE(p, v) \
    U32TO8_LE((p), (uint32_t)((v))); \
    U32TO8_LE((p) + 4, (uint32_t)((v) >> 32));

#define U8TO64_LE(p) \
    (((uint64_t)((p)[0])) | ((uint64_t)((p)[1]) << 8) | \
     ((uint64_t)((p)[2]) << 16) | ((uint64_t)((p)[3]) << 24) | \
     ((uint64_t)((p)[4]) << 32) | ((uint64_t)((p)[5]) << 40) | \
     ((uint64_t)((p)[6]) << 48) | ((uint64_t)((p)[7]) << 56))

#define SIPROUND \
    do { \
        v0 += v1; v1 = ROTL(v1, 13); v1 ^= v0; v0 = ROTL(v0, 32); \
        v2 += v3; v3 = ROTL(v3, 16); v3 ^= v2; v0 += v3; \
        v3 = ROTL(v3, 21); v3 ^= v0; v2 += v1; \
        v1 = ROTL(v1, 17); v1 ^= v2; v2 = ROTL(v2, 32); \
    } while (0)

/* ======================================================================
   HOST SIPHASH (Reference Implementation)
   ====================================================================== */
int siphash(const void *in, const size_t inlen, const void *k, uint8_t *out, const size_t outlen) {
    const unsigned char *ni = (const unsigned char *)in;
    const unsigned char *kk = (const unsigned char *)k;
    assert((outlen == 8) || (outlen == 16));
    uint64_t v0 = 0x736f6d6570736575ULL;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL;
    uint64_t v3 = 0x7465646279746573ULL;
    uint64_t k0 = U8TO64_LE(kk);
    uint64_t k1 = U8TO64_LE(kk + 8);
    uint64_t m;
    int i;
    const unsigned char *end = ni + inlen - (inlen % sizeof(uint64_t));
    const int left = inlen & 7;
    uint64_t b = ((uint64_t)inlen) << 56;
    v3 ^= k1; v2 ^= k0; v1 ^= k1; v0 ^= k0;
    if (outlen == 16) v1 ^= 0xee;
    for (; ni != end; ni += 8) {
        m = U8TO64_LE(ni);
        v3 ^= m;
        for (i = 0; i < cROUNDS; ++i) SIPROUND;
        v0 ^= m;
    }
    switch (left) {
    case 7: b |= ((uint64_t)ni[6]) << 48;
    case 6: b |= ((uint64_t)ni[5]) << 40;
    case 5: b |= ((uint64_t)ni[4]) << 32;
    case 4: b |= ((uint64_t)ni[3]) << 24;
    case 3: b |= ((uint64_t)ni[2]) << 16;
    case 2: b |= ((uint64_t)ni[1]) << 8;
    case 1: b |= ((uint64_t)ni[0]); break;
    case 0: break;
    }
    v3 ^= b;
    for (i = 0; i < cROUNDS; ++i) SIPROUND;
    v0 ^= b;
    if (outlen == 16) v2 ^= 0xee; else v2 ^= 0xff;
    for (i = 0; i < dROUNDS; ++i) SIPROUND;
    b = v0 ^ v1 ^ v2 ^ v3;
    U64TO8_LE(out, b);
    if (outlen == 8) return 0;
    v1 ^= 0xdd;
    for (i = 0; i < dROUNDS; ++i) SIPROUND;
    b = v0 ^ v1 ^ v2 ^ v3;
    U64TO8_LE(out + 8, b);
    return 0;
}

/* ======================================================================
   DEVICE SIPHASH (GPU Implementation)
   ====================================================================== */

// Helper for endian-safe loading on GPU
__device__ inline uint64_t U8TO64_LE_DEV(const uint8_t *p) {
    return ((uint64_t)p[0]) | ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

// Device version of SipHash returning uint64_t directly
__device__ uint64_t siphash_device(const char *in, int inlen, const uint8_t *k) {
    const uint8_t *ni = (const uint8_t *)in;
    uint64_t v0 = 0x736f6d6570736575ULL;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL;
    uint64_t v3 = 0x7465646279746573ULL;
    
    uint64_t k0 = U8TO64_LE_DEV(k);
    uint64_t k1 = U8TO64_LE_DEV(k + 8);
    uint64_t m;
    int i;
    const uint8_t *end = ni + inlen - (inlen % sizeof(uint64_t));
    const int left = inlen & 7;
    uint64_t b = ((uint64_t)inlen) << 56;

    v3 ^= k1; v2 ^= k0; v1 ^= k1; v0 ^= k0;

    for (; ni != end; ni += 8) {
        m = U8TO64_LE_DEV(ni);
        v3 ^= m;
        for (i = 0; i < cROUNDS; ++i) SIPROUND;
        v0 ^= m;
    }

    switch (left) {
    case 7: b |= ((uint64_t)ni[6]) << 48;
    case 6: b |= ((uint64_t)ni[5]) << 40;
    case 5: b |= ((uint64_t)ni[4]) << 32;
    case 4: b |= ((uint64_t)ni[3]) << 24;
    case 3: b |= ((uint64_t)ni[2]) << 16;
    case 2: b |= ((uint64_t)ni[1]) << 8;
    case 1: b |= ((uint64_t)ni[0]); break;
    case 0: break;
    }

    v3 ^= b;
    for (i = 0; i < cROUNDS; ++i) SIPROUND;
    v0 ^= b;
    v2 ^= 0xff; 
    for (i = 0; i < dROUNDS; ++i) SIPROUND;
    b = v0 ^ v1 ^ v2 ^ v3;
    
    return b;
}

/* ======================================================================
   BLOOM FILTER STRUCTURES & HOST UTILS
   ====================================================================== */

typedef unsigned long long int uint128_t; 
#define MAX_STRING_LENGTH 20

struct bloom_filter{
    uint8_t num_hashes;
    double error;
    uint128_t num_bits;
    uint128_t num_elements;
    int misses;
};

char get_random_character(){
    static const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
    return charset[rand() % (sizeof(charset) - 1)];
}

void generate_flattened_string(int count, int max_string_length, char **flattened, int **positions){
    *positions = (int *)malloc(count * sizeof(int));
    *flattened = (char *)malloc((max_string_length + 1) * count * sizeof(char)); 

    int current_position = 0;
    for (int i = 0; i < count; i++){
        int length = rand() % max_string_length + 1;
        (*positions)[i] = current_position;

        for (int j = 0; j < length; j++){
            (*flattened)[current_position++] = get_random_character();
        }
        (*flattened)[current_position++] = '\0';
    }
    *flattened = (char *)realloc(*flattened, current_position * sizeof(char));
}

void init_filter(struct bloom_filter *bloom, uint64_t elements, double error) {
    bloom->error = error;
    bloom->num_elements = elements;
    bloom->num_bits = ceil((elements * log(error)) / log(1 / pow(2, log(2))));
    bloom->num_hashes = round((bloom->num_bits / elements) * log(2));
    bloom->misses = 0;
}

// HOST (CPU) Add
void add_to_filter(struct bloom_filter *bloom, uint8_t *byte_array, const char *str) {
    uint64_t hash;
    uint8_t out[8], key[16] = {1}; // Initialize key to 1s only at first byte is weird in prof code, but we follow it? 
    // Actually prof code: uint8_t key[16] = {1}; initializes index 0 to 1, rest to 0.
    
    uint8_t len = 0;
    while (str[len] != '\0') { len++; }

    for (uint8_t i = 0; i < bloom->num_hashes; i++) {
        siphash(str, len, key, out, 8);
        memcpy(&hash, out, sizeof(uint64_t));
        byte_array[hash % bloom->num_bits] = 1;

        for (size_t j = 0; j < 16; j++) {
            ((uint8_t*)key)[j] ^= (uint8_t)(hash >> (j % 8));
        }
    }
}

// HOST (CPU) Check
int check_filter(struct bloom_filter *bloom, uint8_t *byte_array, const char *str) {
    uint64_t hash;
    uint8_t out[8], key[16] = {1};

    uint8_t len = 0;
    while (str[len] != '\0') { len++; }

    for (uint8_t i = 0; i < bloom->num_hashes; i++) {
        siphash(str, len, key, out, 8);
        memcpy(&hash, out, sizeof(uint64_t));
        
        if (byte_array[hash % (bloom->num_bits)] == 0) return 0;

        for (size_t j = 0; j < 16; j++) {
            ((uint8_t*)key)[j] ^= (uint8_t)(hash >> (j % 8));
        }
    }
    return 1;
}

/* ======================================================================
   GPU KERNELS
   ====================================================================== */

__global__ void bloom_insert_kernel(char *strings, int *positions, uint8_t *bit_array, 
                                    uint64_t num_bits, uint8_t num_hashes, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        char *str = strings + positions[idx];
        uint8_t len = 0;
        while (str[len] != '\0') len++;

        uint8_t key[16] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        uint64_t hash;

        for (uint8_t i = 0; i < num_hashes; i++) {
            hash = siphash_device(str, len, key);
            bit_array[hash % num_bits] = 1;

            for (int j = 0; j < 16; j++) {
                key[j] ^= (uint8_t)(hash >> (j % 8));
            }
        }
    }
}

__global__ void bloom_check_kernel(char *strings, int *positions, uint8_t *bit_array, 
                                   uint64_t num_bits, uint8_t num_hashes, int num_elements, int *misses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        char *str = strings + positions[idx];
        uint8_t len = 0;
        while (str[len] != '\0') len++;

        uint8_t key[16] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        uint64_t hash;
        bool exists = true;

        for (uint8_t i = 0; i < num_hashes; i++) {
            hash = siphash_device(str, len, key);
            
            if (bit_array[hash % num_bits] == 0) {
                exists = false;
                break;
            }

            for (int j = 0; j < 16; j++) {
                key[j] ^= (uint8_t)(hash >> (j % 8));
            }
        }

        if (!exists) {
            atomicAdd(misses, 1);
        }
    }
}

/* ======================================================================
   TIMING FUNCTIONS
   ====================================================================== */

// 1. CPU Timers (Renamed to avoid conflict with GPU requirement)
struct timeval cpu_start, cpu_stop;
float cpu_elapsed_time;

void start_cpu_timer() {
    gettimeofday(&cpu_start, NULL);
}

void stop_cpu_timer() {
    gettimeofday(&cpu_stop, NULL);
    cpu_elapsed_time = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1000.0;
    cpu_elapsed_time += (cpu_stop.tv_usec - cpu_start.tv_usec) / 1000.0;
}

// 2. GPU Timers (Strict compliance with PDF: "Implement start_timer() and stop_timer()")
cudaEvent_t start_ev, stop_ev;

void start_timer() {
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);
    cudaEventRecord(start_ev, 0);
}

float stop_timer() {
    float milliseconds = 0;
    cudaEventRecord(stop_ev, 0);
    cudaEventSynchronize(stop_ev);
    cudaEventElapsedTime(&milliseconds, start_ev, stop_ev);
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    return milliseconds;
}

/* ======================================================================
   MAIN
   ====================================================================== */

int main(int argc, char **argv) {

    /* 1. Command Line Arguments */
    if (argc != 4) {
        printf("Usage: %s { # of elements } { desired %% error } { block size }\n", argv[0]);
        return -1;
    }

    uint64_t NUMBER_OF_ELEMENTS = atoi(argv[1]);
    double ERROR = atof(argv[2]);
    int BLOCK_SIZE = atoi(argv[3]);

    if ((ERROR >= 1) || (ERROR <= 0)) {
        printf("Invalid. Error must be within 0 and 1.\n");
        return -1;
    }

    uint64_t STRINGS_ADDED = NUMBER_OF_ELEMENTS;

    /* 2. Generate Strings */
    char *strings_h;
    int *positions_h;
    srand(1); // Seed for reproducibility
    generate_flattened_string(NUMBER_OF_ELEMENTS, MAX_STRING_LENGTH, &strings_h, &positions_h);

    /* 3. Run CPU (Host) Implementation */
    struct bloom_filter bf_h;
    init_filter(&bf_h, STRINGS_ADDED, ERROR);
    uint8_t *byte_array_h = (uint8_t*)calloc(bf_h.num_bits, sizeof(uint8_t));

    start_cpu_timer();
    for (int i = 0; i < STRINGS_ADDED; i++) {
        add_to_filter(&bf_h, byte_array_h, strings_h + positions_h[i]);
    }
    for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
        if (check_filter(&bf_h, byte_array_h, strings_h + positions_h[i]) == 0) { bf_h.misses++; }
    }
    stop_cpu_timer();

    printf("[CPU] Insert+Query(or Total time of generation): %0.3f ms\n", cpu_elapsed_time);
    printf("[CPU] False negatives: %d/%ld\n", bf_h.misses, NUMBER_OF_ELEMENTS);

    // Free CPU specific resources (keep strings/positions for GPU)
    free(byte_array_h);

    /* 4. Run GPU (Device) Implementation */
    
    // A. Allocate GPU Memory
    char *d_strings;
    int *d_positions;
    uint8_t *d_byte_array;
    int *d_misses;
    int h_gpu_misses = 0;

    // Calculate total size of strings buffer
    int last_idx = NUMBER_OF_ELEMENTS - 1;
    int total_chars = positions_h[last_idx] + strlen(strings_h + positions_h[last_idx]) + 1;

    cudaMalloc((void**)&d_strings, total_chars * sizeof(char));
    cudaMalloc((void**)&d_positions, NUMBER_OF_ELEMENTS * sizeof(int));
    cudaMalloc((void**)&d_byte_array, bf_h.num_bits * sizeof(uint8_t));
    cudaMalloc((void**)&d_misses, sizeof(int));

    // B. Copy Data to GPU
    cudaMemcpy(d_strings, strings_h, total_chars * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions_h, NUMBER_OF_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_byte_array, 0, bf_h.num_bits * sizeof(uint8_t));
    cudaMemset(d_misses, 0, sizeof(int));

    // C. Execute Kernels
    int GRID_SIZE = (NUMBER_OF_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    start_timer(); // Using GPU Timer as required
    
    // Insert
    bloom_insert_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_strings, d_positions, d_byte_array, 
                                                   bf_h.num_bits, bf_h.num_hashes, NUMBER_OF_ELEMENTS);
    
    // Check
    bloom_check_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_strings, d_positions, d_byte_array, 
                                                  bf_h.num_bits, bf_h.num_hashes, NUMBER_OF_ELEMENTS, d_misses);
    
    // Stop Timer (synchronizes inside)
    float gpu_time = stop_timer();

    // D. Retrieve Results
    cudaMemcpy(&h_gpu_misses, d_misses, sizeof(int), cudaMemcpyDeviceToHost);

    // E. Output and Speedup Calculation
    printf("[GPU] Insert+Query: %0.3f ms (%.1fx speedup)\n", gpu_time, cpu_elapsed_time / gpu_time);
    printf("[GPU] False negatives: %d/%ld\n", h_gpu_misses, NUMBER_OF_ELEMENTS);

    /* 5. Cleanup */
    free(strings_h);
    free(positions_h);
    cudaFree(d_strings);
    cudaFree(d_positions);
    cudaFree(d_byte_array);
    cudaFree(d_misses);

    return 0;
}