/*  
    ======================================================================
    Project 3: COP4520 - CPU Implementation 
    Usage for CPU : /apps/GPU_course/runScript.sh bloomfilter_CPU.cu 10000 0.5 
    Note : CPU code will not have the third argument which is block size
    Usage requirement for GPU : /apps/GPU_course/runScript.sh bloomfilter.cu 10000 0.5 256
    Note : In parallel implemnattaion the block size should be passed as an argumnet in command line
    
    
    SipHash C Implementation: https://github.com/veorq/SipHash/tree/master
    ======================================================================
*/  

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

/* SipHash implementation starts here */
// used by siphash
#include <assert.h>
#include <stddef.h>
#include <inttypes.h>
#include <string.h>

/*
    SipHash reference C implementation  -- this is the same implementation provided in the additional information document 
    Copyright (c) 2012-2022 Jean-Philippe Aumasson <jeanphilippe.aumasson@gmail.com>
    Copyright (c) 2012-2014 Daniel J. Bernstein <djb@cr.yp.to>

    To the extent possible under law, the author(s) have dedicated all copyright
    and related and neighboring rights to this software to the public domain
    worldwide. This software is distributed without any warranty.

    You should have received a copy of the CC0 Public Domain Dedication along
    with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
*/

/* default: SipHash-2-4 */
#ifndef cROUNDS
#define cROUNDS 2
#endif
#ifndef dROUNDS
#define dROUNDS 4
#endif

#define ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))

#define U32TO8_LE(p, v)                                                        \
    (p)[0] = (uint8_t)((v));                                                   \
    (p)[1] = (uint8_t)((v) >> 8);                                              \
    (p)[2] = (uint8_t)((v) >> 16);                                             \
    (p)[3] = (uint8_t)((v) >> 24);

#define U64TO8_LE(p, v)                                                        \
    U32TO8_LE((p), (uint32_t)((v)));                                           \
    U32TO8_LE((p) + 4, (uint32_t)((v) >> 32));

#define U8TO64_LE(p)                                                           \
    (((uint64_t)((p)[0])) | ((uint64_t)((p)[1]) << 8) |                        \
     ((uint64_t)((p)[2]) << 16) | ((uint64_t)((p)[3]) << 24) |                 \
     ((uint64_t)((p)[4]) << 32) | ((uint64_t)((p)[5]) << 40) |                 \
     ((uint64_t)((p)[6]) << 48) | ((uint64_t)((p)[7]) << 56))

#define SIPROUND                                                               \
    do {                                                                       \
        v0 += v1;                                                              \
        v1 = ROTL(v1, 13);                                                     \
        v1 ^= v0;                                                              \
        v0 = ROTL(v0, 32);                                                     \
        v2 += v3;                                                              \
        v3 = ROTL(v3, 16);                                                     \
        v3 ^= v2;                                                              \
        v0 += v3;                                                              \
        v3 = ROTL(v3, 21);                                                     \
        v3 ^= v0;                                                              \
        v2 += v1;                                                              \
        v1 = ROTL(v1, 17);                                                     \
        v1 ^= v2;                                                              \
        v2 = ROTL(v2, 32);                                                     \
    } while (0)
#define TRACE

/*  Computes a SipHash value
    ======================================
    *in: pointer to input data (read-only)
    inlen: input data length in bytes (any size_t value)
    *k: pointer to the key data (read-only), must be 16 bytes
    *out: pointer to output data (write-only), outlen bytes must be allocated
    outlen: length of the output in bytes, must be 8 or 16
*/
int siphash(const void *in, const size_t inlen, const void *k, uint8_t *out,
            const size_t outlen) {

    const unsigned char *ni = (const unsigned char *)in;
    const unsigned char *kk = (const unsigned char *)k;
    assert((outlen == 8) || (outlen == 16));
    uint64_t v0 = UINT64_C(0x736f6d6570736575);
    uint64_t v1 = UINT64_C(0x646f72616e646f6d);
    uint64_t v2 = UINT64_C(0x6c7967656e657261);
    uint64_t v3 = UINT64_C(0x7465646279746573);
    uint64_t k0 = U8TO64_LE(kk);
    uint64_t k1 = U8TO64_LE(kk + 8);
    uint64_t m;
    int i;
    const unsigned char *end = ni + inlen - (inlen % sizeof(uint64_t));
    const int left = inlen & 7;
    uint64_t b = ((uint64_t)inlen) << 56;
    v3 ^= k1;
    v2 ^= k0;
    v1 ^= k1;
    v0 ^= k0;
    if (outlen == 16)
        v1 ^= 0xee;
    for (; ni != end; ni += 8) {
        m = U8TO64_LE(ni);
        v3 ^= m;
        TRACE;
        for (i = 0; i < cROUNDS; ++i)
            SIPROUND;
        v0 ^= m;
    }
    switch (left) {
    case 7:
        b |= ((uint64_t)ni[6]) << 48;
        /* FALLTHRU */
    case 6:
        b |= ((uint64_t)ni[5]) << 40;
        /* FALLTHRU */
    case 5:
        b |= ((uint64_t)ni[4]) << 32;
        /* FALLTHRU */
    case 4:
        b |= ((uint64_t)ni[3]) << 24;
        /* FALLTHRU */
    case 3:
        b |= ((uint64_t)ni[2]) << 16;
        /* FALLTHRU */
    case 2:
        b |= ((uint64_t)ni[1]) << 8;
        /* FALLTHRU */
    case 1:
        b |= ((uint64_t)ni[0]);
        break;
    case 0:
        break;
    }
    v3 ^= b;
    TRACE;
    for (i = 0; i < cROUNDS; ++i)
        SIPROUND;
    v0 ^= b;
    if (outlen == 16)
        v2 ^= 0xee;
    else
        v2 ^= 0xff;
    TRACE;
    for (i = 0; i < dROUNDS; ++i)
        SIPROUND;
    b = v0 ^ v1 ^ v2 ^ v3;
    U64TO8_LE(out, b);
    if (outlen == 8)
        return 0;
    v1 ^= 0xdd;
    TRACE;
    for (i = 0; i < dROUNDS; ++i)
        SIPROUND;
    b = v0 ^ v1 ^ v2 ^ v3;
    U64TO8_LE(out + 8, b);
    return 0;
}
/* SipHash implementation ends here */

typedef unsigned long long int uint128_t;   // used for readability equivalent to long long int

/*  
    These macros go unused currently, but could be used to reduce the space complexity of the array.
    Instead of using an entire byte (8 bits) to store a single value, these macros allow you to
    work with individual bits. This reduces memory usage by packing multiple values into a single 
    byte rather than allocating one byte per value.

    #define set_bit(A,k)    (A[(k)/32] |= (1 << ((k)%32)))
    #define clear_bit(A,k)  (A[(k)/32] &= ~(1 << ((k)%32)))
    #define check_bit(A,k)  (A[(k)/32] & (1 << ((k)%32)))
*/

#define MAX_STRING_LENGTH 20        // max length of each string generated by generate_flattened_string

struct bloom_filter{
    uint8_t num_hashes;             // number of hashes
    double error;                   // desired probability for false positives
    uint128_t num_bits;             // number of bits in array
    uint128_t num_elements;         // number of strings in the array
    int misses;                     // number of misses (default: 0)
};

double ERROR;                                   // false positivity rate of the filter (determined by user)
uint64_t NUMBER_OF_ELEMENTS, STRINGS_ADDED;     // number of strings total and number of strings added to the filter

struct bloom_filter bf_h;               // bloom filters for host (cpu)
char *strings_h;            // 1D-char array for host (cpu)
uint8_t *byte_array_h;   // 1D-unsigned byte array (0 - 255) for host (cpu)
int *positions_h;         // 1D-int array for host (cpu)

float elapsed_time;                 // used to print elapsed time
struct timeval start, stop;            

/* Starts the CPU timer */
void start_timer() {
    gettimeofday(&start, NULL);
}

/* Stops the CPU timer */
void stop_timer() {
    gettimeofday(&stop, NULL);
    elapsed_time = (stop.tv_sec - start.tv_sec) * 1000.0;
    elapsed_time += (stop.tv_usec - start.tv_usec) / 1000.0;
}

/*  Initalize the filter
    =======================================
    *bloom: pointer to bloom filter struct
    elements: number of elements that will be added to the filter
    error: false positivity rate of the filter

    The num_bits and num_hashes is determined by error and elements
    reference -- https://en.wikipedia.org/wiki/Bloom_filter#Optimal_number_of_hash_functions 
*/
void init_filter(struct bloom_filter *bloom, uint64_t elements, double error) {
    bloom->error = error;
    bloom->num_elements = elements;

    bloom->num_bits = ceil((elements * log(error)) / log(1 / pow(2, log(2))));
    bloom->num_hashes = round((bloom->num_bits / elements) * log(2));
    bloom->misses = 0;
}

/*  Adds a string to the bloom filter
    =======================================
    *bloom: pointer to bloom filter struct
    *byte_array: pointer to byte array
    *str: char array (read-only)
*/
void add_to_filter(struct bloom_filter *bloom, uint8_t *byte_array, const char *str) {

    uint64_t hash;
    uint8_t out[8], key[16] = {1};

    // find the string length -- cannot use strlen() because that is __host__ only.
    uint8_t len = 0;
    while (str[len] != '\0') { len++; }

    // generate and add as many hashes as required (determined by function init_filter)
    for (uint8_t i = 0; i < bloom->num_hashes; i++) {
        siphash(str, len, key, out, 8);             // create a new hash from the given string and key
        memcpy(&hash, out, sizeof(uint64_t));       // copy the output to the hash variable
        byte_array[hash % bloom->num_bits] = 1;     // set the index byte to 1

        // regenerate a new key based on the previous hash
        for (size_t j = 0; j < 16; j++) {
            ((uint8_t*)key)[j] ^= (uint8_t)(hash >> (j % 8));
        }
    }
}

/*  Checks a string exists in the bloom filter
    =======================================
    *bloom: pointer to bloom filter struct
    *byte_array: pointer to byte array
    *str: char array (read-only)
*/
int check_filter(struct bloom_filter *bloom, uint8_t *byte_array, const char *str) {

    uint64_t hash;
    uint8_t out[8], key[16] = {1};

    // find the string length -- cannot use strlen() because that is __host__ only.
    uint8_t len = 0;
    while (str[len] != '\0') { len++; }

    // generate and check as many hashes as required (determined by function init_filter)
    for (uint8_t i = 0; i < bloom->num_hashes; i++) {
        siphash(str, len, key, out, 8);             // create a new hash from the given string and key
        memcpy(&hash, out, sizeof(uint64_t));       // copy the output to the hash variable
        
        /*  if byte_array is set the 1, then the string may exist in the filter (not guaranteed)
            if byte_array is set to 0, then the string does not exist in the filter (guaranteed). */
        if (byte_array[hash % (bloom->num_bits)] == 0) return 0;

        // regenerate a new key based on the previous hash
        for (size_t j = 0; j < 16; j++) {
            ((uint8_t*)key)[j] ^= (uint8_t)(hash >> (j % 8));
        }
    }
    return 1;
}

// Returns random alphanumeric character
char get_random_character(){
    static const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
    return charset[rand() % (sizeof(charset) - 1)];
}

/*  Create 1D char array, where the position
    is used to find the index where the string
    starts.
        + Saves on malloc time.
    =======================================
    count: number of strings generated
    max_string_length: max length of each string generated
    **flattened: pointer to char array (pass by reference so that a copy isn't required)
        -- stores all strings into single 1d-array
    **positions: pointer to int array (pass by reference so that a copy isn't required)
        -- stores all string offsets into a single 1d-array
*/
void generate_flattened_string(int count, int max_string_length, char **flattened, int **positions){
    *positions = (int *)malloc(count * sizeof(int));
    *flattened = (char *)malloc((max_string_length + 1) * count * sizeof(char)); // Overestimate

    int current_position = 0;
    for (int i = 0; i < count; i++){
        int length = rand() % max_string_length + 1;
        (*positions)[i] = current_position;

        for (int j = 0; j < length; j++){
            (*flattened)[current_position++] = get_random_character();
        }
        // Every string is null terminated so a simple "flattened + positions[i]" works to reference the i-th string.
        (*flattened)[current_position++] = '\0';
    }

    // Realloc now that the size is known.
    *flattened = (char *)realloc(*flattened, current_position * sizeof(char));
}

int main(int argc, char **argv) {

    /* _____ COMMAND LINE ARGUMENTS _________________________________________________________________________ */

    if (argc != 3) {
        printf("Invalid. Requires 2 arguments.\n");
        printf("{ # of elements } { desired %% error }\n\n");
        return -1;
    }

    NUMBER_OF_ELEMENTS = atoi(argv[1]);
    ERROR = atof(argv[2]);
    if ((ERROR >= 1) || (ERROR <= 0)) {
        printf("Invalid. Error must be within 0 and 1. Currently => %.3lf\n\n", ERROR);
        return -1;
    }

    STRINGS_ADDED = NUMBER_OF_ELEMENTS; // ceil(NUMBER_OF_ELEMENTS / 2);

    /* _____ GENERATE STRINGS _______________________________________________________________________________ */

    srand(1);   // set seed for randomly generated strings
    generate_flattened_string(NUMBER_OF_ELEMENTS, MAX_STRING_LENGTH, &strings_h, &positions_h);

    /* _____ INITIALIZE CPU FILTER __________________________________________________________________________ */

    init_filter(&bf_h, STRINGS_ADDED, ERROR);   // calculate byte array sizes and number of hashes needed
    byte_array_h = (uint8_t*)calloc(bf_h.num_bits, sizeof(uint8_t));

    // total size of the strings_h array
    uint128_t size = positions_h[STRINGS_ADDED - 1] + strlen(strings_h + positions_h[STRINGS_ADDED - 1] + 1); 

    /* _____ TEST CPU CODE  _________________________________________________________________________________ */

    start_timer();
    for (int i = 0; i < STRINGS_ADDED; i++) {
        add_to_filter(&bf_h, byte_array_h, strings_h + positions_h[i]);
    }

    for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
        if (check_filter(&bf_h, byte_array_h, strings_h + positions_h[i]) == 0) { bf_h.misses++; }
    }
    stop_timer();

    printf("[CPU] Insert+Query(or Total time of generation): %0.3f ms\n", elapsed_time);
	  printf("[CPU] False negatives: %d/%ld\n", bf_h.misses, NUMBER_OF_ELEMENTS);

    /* _____ FREE MEMORY  ___________________________________________________________________________________ */

    free(byte_array_h);
    free(strings_h);
    
    return 0;
}