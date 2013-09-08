 /*
 * MD5 OpenCL code is based on Alain Espinosa's OpenCL patches.
 *
 * This software is Copyright (c) 2010, Dhiru Kholia <dhiru.kholia at gmail.com>
 * ,Copyright (c) 2012, magnum
 * and Copyright (c) 2013, Sayantan Datta <std2048 at gmail.com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 */

#include <string.h>
#include <math.h>

#include "arch.h"
#include "params.h"
#include "path.h"
#include "common.h"
#include "formats.h"
#include "common-opencl.h"
#include "config.h"
#include "options.h"
#include "loader.h"
#include "opencl_rawmd5_fmt.h"

#define PLAINTEXT_LENGTH    55 /* Max. is 55 with current kernel, Warning: key length is hardcoded in md5_kernel struct return_key */
#define BUFSIZE             ((PLAINTEXT_LENGTH+3)/4*4)
#define FORMAT_LABEL        "Raw-MD5-opencl"
#define FORMAT_NAME         ""
#define ALGORITHM_NAME      "MD5 OpenCL (inefficient, development use only)"
#define BENCHMARK_COMMENT   ""
#define BENCHMARK_LENGTH    -1
#define CIPHERTEXT_LENGTH   32
#define DIGEST_SIZE         16
#define BINARY_SIZE         16
#define BINARY_ALIGN        4
#define SALT_SIZE           0
#define SALT_ALIGN          1
#define FORMAT_TAG          "$dynamic_0$"
#define TAG_LENGTH          (sizeof(FORMAT_TAG) - 1)

#define RAWMD5_DEBUG 0

cl_mem pinned_saved_keys, pinned_saved_idx, pinned_partial_hashes;
cl_mem buffer_keys, buffer_idx, buffer_out;
static unsigned int *saved_plain;
static uint64_t *saved_idx, key_idx = 0;
static unsigned int num_keys= 0;
static cl_uint *partial_hashes, *res_hashes ;

cl_mem buffer_ld_hashes, buffer_outKeyIdx, buffer_mask_gpu;
static unsigned int *loaded_hashes, cmp_out, *outKeyIdx;
static int loaded_count;

cl_mem buffer_bitmap1, buffer_bitmap2;
static struct bitmap_context_mixed  *bitmap1;
static struct bitmap_context_global *bitmap2;

static struct mask_context msk_ctx;
static unsigned char *mask_offsets;
static struct db_main *DB;
static unsigned int multiplier = 1;

cl_kernel crk_kernel_nnn, crk_kernel_ccc, crk_kernel_cnn, crk_kernel_om, crk_kernel;

static int self_test = 1; // used as a flag
static unsigned int mask_mode = 0;

#define MIN(a, b)               (((a) > (b)) ? (b) : (a))
#define MAX(a, b)               (((a) > (b)) ? (a) : (b))

#define MIN_KEYS_PER_CRYPT      1024
#define MAX_KEYS_PER_CRYPT      (1024 * 2048 * 4)

#define OCL_CONFIG             "rawmd5"
#define STEP                    65536

static int have_full_hashes;

extern void common_find_best_lws(size_t group_size_limit,
        unsigned int sequential_id, cl_kernel crypt_kernel);
extern void common_find_best_gws(int sequential_id, unsigned int rounds, int step,
        unsigned long long int max_run_time);

static int crypt_all_self_test(int *pcount, struct db_salt *_salt);
static int crypt_all(int *pcount, struct db_salt *_salt);
static void load_mask(struct fmt_main *fmt);
static void select_kernel(struct mask_context *msk_ctx);
static char *get_key_self_test(int index);
static char *get_key(int index);

static struct fmt_tests tests[] = {
	{"098f6bcd4621d373cade4e832627b4f6", "test"},
	{FORMAT_TAG "378e2c4a07968da2eca692320136433d","thatsworking"},
	{FORMAT_TAG "8ad8757baa8564dc136c1e07507f4a98","test3"},
	{"d41d8cd98f00b204e9800998ecf8427e", ""},
	{NULL}
};

static void create_clobj(int kpc, struct fmt_main * self)
{
	self->params.min_keys_per_crypt = self->params.max_keys_per_crypt = kpc;

	pinned_saved_keys = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, BUFSIZE * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating page-locked memory pinned_saved_keys");
	saved_plain = clEnqueueMapBuffer(queue[ocl_gpu_id], pinned_saved_keys, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, BUFSIZE * kpc, 0, NULL, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error mapping page-locked memory saved_plain");

	pinned_saved_idx = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(uint64_t) * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating page-locked memory pinned_saved_idx");
	saved_idx = clEnqueueMapBuffer(queue[ocl_gpu_id], pinned_saved_idx, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(uint64_t) * kpc, 0, NULL, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error mapping page-locked memory saved_idx");

	res_hashes = malloc(sizeof(cl_uint) * 3 * kpc);

	pinned_partial_hashes = clCreateBuffer(context[ocl_gpu_id], CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 4 * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating page-locked memory pinned_partial_hashes");
	partial_hashes = (cl_uint *) clEnqueueMapBuffer(queue[ocl_gpu_id], pinned_partial_hashes, CL_TRUE, CL_MAP_READ, 0, 4 * kpc, 0, NULL, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error mapping page-locked memory partial_hashes");

	// create and set arguments
	buffer_keys = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_ONLY, BUFSIZE * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_keys");

	buffer_idx = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_ONLY, sizeof(uint64_t) * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_idx");

	buffer_out = clCreateBuffer(context[ocl_gpu_id], CL_MEM_WRITE_ONLY, DIGEST_SIZE * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_out");

	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 0, sizeof(buffer_keys), (void *) &buffer_keys), "Error setting argument 1");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 1, sizeof(buffer_idx), (void *) &buffer_idx), "Error setting argument 2");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 2, sizeof(buffer_out), (void *) &buffer_out), "Error setting argument 3");

	global_work_size = kpc;
}

static void release_clobj(void)
{
	if (self_test)
		HANDLE_CLERROR(clEnqueueUnmapMemObject(queue[ocl_gpu_id], pinned_partial_hashes, partial_hashes, 0,NULL,NULL), "Error Unmapping partial_hashes");
	else
		MEM_FREE(partial_hashes);

	HANDLE_CLERROR(clEnqueueUnmapMemObject(queue[ocl_gpu_id], pinned_saved_keys, saved_plain, 0, NULL, NULL), "Error Unmapping saved_plain");
	HANDLE_CLERROR(clEnqueueUnmapMemObject(queue[ocl_gpu_id], pinned_saved_idx, saved_idx, 0, NULL, NULL), "Error Unmapping saved_idx");

	HANDLE_CLERROR(clReleaseMemObject(buffer_keys), "Error Releasing buffer_keys");
	HANDLE_CLERROR(clReleaseMemObject(buffer_idx), "Error Releasing buffer_idx");
	HANDLE_CLERROR(clReleaseMemObject(buffer_out), "Error Releasing buffer_out");
	HANDLE_CLERROR(clReleaseMemObject(pinned_saved_keys), "Error Releasing pinned_saved_keys");
	HANDLE_CLERROR(clReleaseMemObject(pinned_partial_hashes), "Error Releasing pinned_partial_hashes");
	MEM_FREE(res_hashes);
}

static void done(void)
{
	release_clobj();
	if(!self_test) {
		HANDLE_CLERROR(clReleaseMemObject(buffer_outKeyIdx), "Error Releasing cmp_out");
		HANDLE_CLERROR(clReleaseMemObject(buffer_ld_hashes), "Error Releasing loaded hashes");
		HANDLE_CLERROR(clReleaseMemObject(buffer_bitmap1), "Error Releasing loaded hashes");
		HANDLE_CLERROR(clReleaseMemObject(buffer_bitmap2), "Error Releasing loaded hashes");

		MEM_FREE(outKeyIdx);
		MEM_FREE(loaded_hashes);
		MEM_FREE(mask_offsets);
		MEM_FREE(bitmap1);
		MEM_FREE(bitmap2);
	}

	HANDLE_CLERROR(clReleaseMemObject(buffer_mask_gpu), "Error Releasing mask buffer.");
	HANDLE_CLERROR(clReleaseKernel(crypt_kernel), "Release self_test kernel");
	HANDLE_CLERROR(clReleaseKernel(crk_kernel_nnn), "Release cracking kernel");
	HANDLE_CLERROR(clReleaseKernel(crk_kernel_ccc), "Release cracking kernel");
	HANDLE_CLERROR(clReleaseKernel(crk_kernel_cnn), "Release cracking kernel");
	HANDLE_CLERROR(clReleaseKernel(crk_kernel_om), "Release cracking kernel");
	HANDLE_CLERROR(clReleaseProgram(program[ocl_gpu_id]), "Release Program");
}

static void init(struct fmt_main *self)
{
	opencl_init("$JOHN/kernels/md5_kernel.cl", ocl_gpu_id, NULL);
	crypt_kernel = clCreateKernel(program[ocl_gpu_id], "md5_self_test", &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating kernel. Double-check kernel name?");

	crk_kernel_om = clCreateKernel(program[ocl_gpu_id], "md5_om", &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating kernel. Double-check kernel name?");

	crk_kernel_nnn = clCreateKernel(program[ocl_gpu_id], "md5_nnn", &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating kernel. Double-check kernel name?");

	crk_kernel_ccc = clCreateKernel(program[ocl_gpu_id], "md5_ccc", &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating kernel. Double-check kernel name?");

	crk_kernel_cnn = clCreateKernel(program[ocl_gpu_id], "md5_cnn", &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating kernel. Double-check kernel name?");

	/* Read LWS/GWS prefs from config or environment */
	opencl_get_user_preferences(OCL_CONFIG);

	/* Round off to nearest power of 2 */
	if(local_work_size)
		local_work_size = pow(2, ceil(log(local_work_size)/log(2)));
	if(!global_work_size)
		global_work_size = MAX_KEYS_PER_CRYPT;
	if(!local_work_size)
		local_work_size = LWS;

	buffer_mask_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_ONLY, sizeof(struct mask_context) , NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer mask gpu\n");

	if(options.mask) {
		int i;
		mask_mode = 1;
		local_work_size = LWS;
		if (!getenv("GWS"))
			global_work_size /= 2;
		load_mask(self);
		multiplier = 1;
		for (i = 0; i < msk_ctx.count; i++)
			multiplier *= msk_ctx.ranges[msk_ctx.activeRangePos[i]].count;
#if RAWMD5_DEBUG
		fprintf(stderr, "Multiply the end c/s with:%d\n", multiplier);
#endif
	}

	create_clobj((global_work_size + local_work_size - 1) / local_work_size * local_work_size , self);

	if (options.verbosity > 2)
		fprintf(stderr,
		        "Local worksize (LWS) %zd, global worksize (GWS) %zd\n",
		        local_work_size, global_work_size);
	self->params.min_keys_per_crypt = local_work_size;
	self->params.max_keys_per_crypt = global_work_size;
	self->methods.crypt_all = crypt_all_self_test;
	self->methods.get_key = get_key_self_test;
}

static int valid(char *ciphertext, struct fmt_main *self)
{
	char *p, *q;

	p = ciphertext;
	if (!strncmp(p, FORMAT_TAG, TAG_LENGTH))
		p += TAG_LENGTH;

	q = p;
	while (atoi16[ARCH_INDEX(*q)] != 0x7F) {
		if (*q >= 'A' && *q <= 'F') /* support lowercase only */
			return 0;
		q++;
	}
	return !*q && q - p == CIPHERTEXT_LENGTH;
}

static char *split(char *ciphertext, int index, struct fmt_main *self)
{
	static char out[TAG_LENGTH + CIPHERTEXT_LENGTH + 1];

	if (!strncmp(ciphertext, FORMAT_TAG, TAG_LENGTH))
		return ciphertext;

	memcpy(out, FORMAT_TAG, TAG_LENGTH);
	memcpy(out + TAG_LENGTH, ciphertext, CIPHERTEXT_LENGTH + 1);
	return out;
}

static void *get_binary(char *ciphertext)
{
	static unsigned char out[DIGEST_SIZE];
	char *p;
	int i;
	p = ciphertext + TAG_LENGTH;
	for (i = 0; i < sizeof(out); i++) {
		out[i] = (atoi16[ARCH_INDEX(*p)] << 4) | atoi16[ARCH_INDEX(p[1])];
		p += 2;
	}
	return out;
}

static int get_hash_0(int index) {  return partial_hashes[index] & 0xf;        }
static int get_hash_1(int index) {  return partial_hashes[index] & 0xff;       }
static int get_hash_2(int index) {  return partial_hashes[index] & 0xfff;      }
static int get_hash_3(int index) {  return partial_hashes[index] & 0xffff;     }
static int get_hash_4(int index) {  return partial_hashes[index] & 0xfffff;    }
static int get_hash_5(int index) {  return partial_hashes[index] & 0xffffff;   }
static int get_hash_6(int index) {  return partial_hashes[index] & 0x7ffffff;  }

static void clear_keys(void)
{
	key_idx = 0;
	num_keys = 0;
}

static void setKernelArgs(cl_kernel *kernel) {
	int argIdx = 0;

	HANDLE_CLERROR(clSetKernelArg(*kernel, argIdx++, sizeof(buffer_keys), &buffer_keys), "Error setting argument 1");
	HANDLE_CLERROR(clSetKernelArg(*kernel, argIdx++, sizeof(buffer_idx), &buffer_idx), "Error setting argument 2");
	HANDLE_CLERROR(clSetKernelArg(*kernel, argIdx++, sizeof(buffer_ld_hashes), &buffer_ld_hashes), "Error setting argument 4");
	HANDLE_CLERROR(clSetKernelArg(*kernel, argIdx++, sizeof(buffer_outKeyIdx), &buffer_outKeyIdx), "Error setting argument 5");
	if(mask_mode)
		HANDLE_CLERROR(clSetKernelArg(*kernel, argIdx++, sizeof(buffer_mask_gpu), &buffer_mask_gpu), "Error setting argument 7");
	HANDLE_CLERROR(clSetKernelArg(*kernel, argIdx++, sizeof(buffer_bitmap1), &buffer_bitmap1), "Error setting argument 8");
	HANDLE_CLERROR(clSetKernelArg(*kernel, argIdx++, sizeof(buffer_bitmap2), &buffer_bitmap2), "Error setting argument 9");
}

static void opencl_md5_reset(struct db_main *db) {


	if(db) {
		unsigned int length = 0;

		db->format->params.min_keys_per_crypt = db->format->params.max_keys_per_crypt;

		HANDLE_CLERROR(clEnqueueUnmapMemObject(queue[ocl_gpu_id], pinned_partial_hashes, partial_hashes, 0,NULL,NULL), "Error Unmapping partial_hashes");
		loaded_hashes = (unsigned int*)mem_alloc(((db->password_count) * 4 + 1)*sizeof(unsigned int));
		partial_hashes = (unsigned int*)mem_alloc(((db->password_count) + 1)*sizeof(unsigned int));
		outKeyIdx     = (unsigned int*)mem_alloc((db->password_count) * sizeof(unsigned int) * 2);
		mask_offsets  = (unsigned char*) mem_calloc(db->format->params.max_keys_per_crypt);
		bitmap1       = (struct bitmap_context_mixed*)mem_alloc(sizeof(struct bitmap_context_mixed));
		bitmap2       = (struct bitmap_context_global*)mem_alloc(sizeof(struct bitmap_context_global));

		buffer_ld_hashes = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, ((db->password_count) * 4 + 1)*sizeof(int), NULL, &ret_code);
		HANDLE_CLERROR(ret_code, "Error creating buffer arg loaded_hashes\n");

		buffer_bitmap1 = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, sizeof(struct bitmap_context_mixed), NULL, &ret_code);
		HANDLE_CLERROR(ret_code, "Error creating buffer arg loaded_hashes\n");

		buffer_bitmap2 = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, sizeof(struct bitmap_context_global), NULL, &ret_code);
		HANDLE_CLERROR(ret_code, "Error creating buffer arg loaded_hashes\n");

		length = ((db->format->params.max_keys_per_crypt) > ((db->password_count) * sizeof(unsigned int) * 2)) ?
			  (db->format->params.max_keys_per_crypt) : ((db->password_count) * sizeof(unsigned int) * 2);
		/* buffer_outKeyIdx is multiplexed for use as mask_offset input and keyIdx output */
		buffer_outKeyIdx = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, length, NULL, &ret_code);
		HANDLE_CLERROR(ret_code, "Error creating buffer cmp_out\n");

		self_test = 0;

		if (mask_mode) {
			setKernelArgs(&crk_kernel_nnn);
			setKernelArgs(&crk_kernel_ccc);
			setKernelArgs(&crk_kernel_cnn);
			select_kernel(&msk_ctx);

			DB = db;
		}

		else {
			setKernelArgs(&crk_kernel_om);
			crk_kernel = crk_kernel_om;
		}

		db->format->methods.crypt_all = crypt_all;
		db->format->methods.get_key = get_key;

	}
}

static void load_hash(struct db_salt *salt) {

	unsigned int *bin, i;
	struct db_password *pw;

	loaded_count = (salt->count);
	loaded_hashes[0] = loaded_count;
	pw = salt -> list;
	i = 0;
	do {
		bin = (unsigned int *)pw -> binary;
		// Potential segfault if removed
		if(bin != NULL) {
			loaded_hashes[i + 1] = bin[0];
			loaded_hashes[i + loaded_count + 1] = bin[1];
			loaded_hashes[i + 2 * loaded_count + 1] = bin[2];
			loaded_hashes[i + 3 * loaded_count + 1] = bin[3];
			i++ ;
		}
	} while ((pw = pw -> next)) ;

	if(i != (salt->count)) {
		fprintf(stderr, "Something went wrong while loading hashes to gpu..Exiting..\n");
		exit(0);
	}

	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], buffer_ld_hashes, CL_TRUE, 0, (i * 4 + 1) * sizeof(unsigned int) , loaded_hashes, 0, NULL, NULL), "failed in clEnqueueWriteBuffer loaded_hashes");
}

static void load_bitmap(unsigned int num_loaded_hashes, unsigned int index, unsigned int *bitmap, size_t szBmp) {
	unsigned int i, hash;
	memset(bitmap, 0, szBmp);

	for(i = 0; i < num_loaded_hashes; i++) {
		hash = loaded_hashes[index * num_loaded_hashes + i + 1] & (szBmp * 8 - 1);
		// divide by 32 , harcoded here and correct only for unsigned int
		bitmap[hash >> 5] |= (1U << (hash & 31));
	}
}

static void load_hashtable_plus(unsigned int *hashtable, unsigned int *loaded_next_hash, unsigned int idx, unsigned int num_loaded_hashes, unsigned int szHashTbl) {
	unsigned int i;
#if RAWMD5_DEBUG
	unsigned int counter = 0;
#endif
	memset(hashtable, 0xFF, szHashTbl * sizeof(unsigned int));
	memset(loaded_next_hash, 0xFF, num_loaded_hashes * sizeof(unsigned int));

	for (i = 0; i < num_loaded_hashes; ++i) {
		unsigned int hash = loaded_hashes[i + idx*num_loaded_hashes + 1] & (szHashTbl - 1);
		loaded_next_hash[i] = hashtable[hash];
#if RAWMD5_DEBUG
		if(!(hashtable[hash]^0xFFFFFFFF)) counter++;
#endif
		hashtable[hash] = i;
	}
#if RAWMD5_DEBUG
	fprintf(stderr, "Hash Table Effectiveness:%lf%%\n", ((double)counter/(double)num_loaded_hashes)*100);
#endif
}

static void check_mask_rawmd5(struct mask_context *msk_ctx) {
	int i, j, k ;

	if(msk_ctx -> count > PLAINTEXT_LENGTH) msk_ctx -> count = PLAINTEXT_LENGTH;
	if(msk_ctx -> count > MASK_RANGES_MAX) {
		fprintf(stderr, "MASK parameters are too small...Exiting...\n");
		exit(EXIT_FAILURE);

	}

  /* Assumes msk_ctx -> activeRangePos[] is sorted. Check if any range exceeds md5 key limit */
	for( i = 0; i < msk_ctx->count; i++)
		if(msk_ctx -> activeRangePos[i] >= PLAINTEXT_LENGTH) {
			msk_ctx->count = i;
			break;
		}
	j = 0;
	i = 0;
	k = 0;
 /* Append non-active portion to activeRangePos[] for ease of computation inside GPU */
	while((j <= msk_ctx -> activeRangePos[k]) && (k < msk_ctx -> count)) {
		if(j == msk_ctx -> activeRangePos[k]) {
			k++;
			j++;
			continue;
		}
		msk_ctx -> activeRangePos[msk_ctx -> count + i] = j;
		i++;
		j++;
	}
	while ((i+msk_ctx->count) < MASK_RANGES_MAX) {
		msk_ctx -> activeRangePos[msk_ctx -> count + i] = j;
		i++;
		j++;
	}

	for(i = msk_ctx->count; i < MASK_RANGES_MAX; i++)
		msk_ctx->ranges[msk_ctx -> activeRangePos[i]].count = 0;

	/* Sort active ranges in descending order of charchter count */
	if(msk_ctx->ranges[msk_ctx -> activeRangePos[0]].count < msk_ctx->ranges[msk_ctx -> activeRangePos[1]].count) {
		i = msk_ctx -> activeRangePos[1];
		msk_ctx -> activeRangePos[1] = msk_ctx -> activeRangePos[0];
		msk_ctx -> activeRangePos[0] = i;
	}

	if(msk_ctx->ranges[msk_ctx -> activeRangePos[0]].count < msk_ctx->ranges[msk_ctx -> activeRangePos[2]].count) {
		i = msk_ctx -> activeRangePos[2];
		msk_ctx -> activeRangePos[2] = msk_ctx -> activeRangePos[0];
		msk_ctx -> activeRangePos[0] = i;
	}

	if(msk_ctx->ranges[msk_ctx -> activeRangePos[1]].count < msk_ctx->ranges[msk_ctx -> activeRangePos[2]].count) {
		i = msk_ctx -> activeRangePos[2];
		msk_ctx -> activeRangePos[2] = msk_ctx -> activeRangePos[1];
		msk_ctx -> activeRangePos[1] = i;
	}
	/* Consecutive charchters in ranges that have all the charchters consective are
	 * arranged in ascending order. This is to make password generation on host and device
	 * match each other for md5_ccc and md5_cnn kernels*/
	for( i = 0; i < msk_ctx->count; i++)
		if(msk_ctx->ranges[msk_ctx -> activeRangePos[i]].start != 0) {
			for (j = 0; j < msk_ctx->ranges[msk_ctx -> activeRangePos[i]].count; j++)
				msk_ctx->ranges[msk_ctx -> activeRangePos[i]].chars[j] =
					msk_ctx->ranges[msk_ctx -> activeRangePos[i]].start + j;

		}

}

static void load_mask(struct fmt_main *fmt) {

	if (!fmt->private.msk_ctx) {
		fprintf(stderr, "No given mask.Exiting...\n");
		exit(EXIT_FAILURE);
	}
	memcpy(&msk_ctx, fmt->private.msk_ctx, sizeof(struct mask_context));
	check_mask_rawmd5(&msk_ctx);
#if RAWMD5_DEBUG
	int i, j;
	for(i = 0; i < MASK_RANGES_MAX; i++)
	    printf("%d ",msk_ctx.activeRangePos[i]);
	printf("\n");
	for(i = 0; i < MASK_RANGES_MAX; i++)
	    printf("%d ",msk_ctx.ranges[msk_ctx.activeRangePos[i]].count);
	printf("\n");

	/*
	for(i = 0; i < msk_ctx.count; i++)
	  printf(" %d ", msk_ctx.activeRangePos[i]);*/
	for(i = 0; i < msk_ctx.count; i++){
			for(j = 0; j < msk_ctx.ranges[msk_ctx.activeRangePos[i]].count; j++)
				printf("%c ",msk_ctx.ranges[msk_ctx.activeRangePos[i]].chars[j]);
			printf("\n");
			//checkRange(&msk_ctx, msk_ctx.activeRangePos[i]) ;
			printf("START:%c",msk_ctx.ranges[msk_ctx.activeRangePos[i]].start);
			printf("\n");
	}
#endif
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], buffer_mask_gpu, CL_TRUE, 0, sizeof(struct mask_context), &msk_ctx, 0, NULL, NULL ), "Failed Copy data to gpu");
}

/* crk_kernel_ccc: optimized for kernel with all 3 ranges consecutive.
 * crk_kernel_nnn: optimized for kernel with no consecutive ranges.
 * crk_kernel_cnn: optimized for kernel with 1st range being consecutive and remaining ranges non-consecutive.
 *
 * select_kernel() assumes that the active ranges are arranged according to decreasing character count, which is taken
 * care of inside check_mask_rawmd5().
 *
 * crk_kernel_ccc used for mask types: ccc, cc, c.
 * crk_kernel_nnn used for mask types: nnn, nnc, ncn, ncc, nc, nn, n.
 * crk_kernel_cnn used for mask types: cnn, cnc, ccn, cn.
 */

static void select_kernel(struct mask_context *msk_ctx) {

	if (!(msk_ctx->ranges[msk_ctx->activeRangePos[0]].start)) {
		crk_kernel = crk_kernel_nnn;
		fprintf(stderr,"Using kernel md5_nnn...\n" );
		return;
	}

	else {
		crk_kernel = crk_kernel_ccc;

		if ((msk_ctx->count) > 1) {
			if (!(msk_ctx->ranges[msk_ctx->activeRangePos[1]].start)) {
				crk_kernel = crk_kernel_cnn;
				fprintf(stderr,"Using kernel md5_cnn...\n" );
				return;
			}

			else {
				crk_kernel = crk_kernel_ccc;

				/* For type ccn */
				if ((msk_ctx->count) == 3)
					if (!(msk_ctx->ranges[msk_ctx->activeRangePos[2]].start))  {
						crk_kernel = crk_kernel_cnn;
						if ((msk_ctx->ranges[msk_ctx->activeRangePos[2]].count) > 64) {
							fprintf(stderr,"Raw-MD5-opencl failed processing mask type ccn.\n" );
						}
						fprintf(stderr,"Using kernel md5_cnn...\n" );
						return;
					}

				fprintf(stderr,"Using kernel md5_ccc...\n" );
				return;
			}
		}

		fprintf(stderr,"Using kernel md5_ccc...\n" );
		return;
	}
}

static void set_key(char *_key, int index)
{
	const ARCH_WORD_32 *key = (ARCH_WORD_32*)_key;
	int len = strlen(_key);

	saved_idx[index] = (key_idx << 6) | len;

	while (len > 4) {
		saved_plain[key_idx++] = *key++;
		len -= 4;
	}
	if (len)
		saved_plain[key_idx++] = *key & (0xffffffffU >> (32 - (len << 3)));

	num_keys++;
}

static char *get_key_self_test(int index)
{
	static char out[PLAINTEXT_LENGTH + 20];
	int i;
	int  len = saved_idx[index] & 63;
	char *key = (char*)&saved_plain[saved_idx[index] >> 6];

	for (i = 0; i < len; i++)
		out[i] = *key++;
	out[i] = 0;

	return out;
}

static void passgen(int ctr, int offset, char *key) {
	int i, j, k;

	offset = msk_ctx.flg_wrd ? offset : 0;

	i =  ctr % msk_ctx.ranges[msk_ctx.activeRangePos[0]].count;
	key[msk_ctx.activeRangePos[0] + offset] = msk_ctx.ranges[msk_ctx.activeRangePos[0]].chars[i];

	if (msk_ctx.ranges[msk_ctx.activeRangePos[1]].count) {
		j = (ctr / msk_ctx.ranges[msk_ctx.activeRangePos[0]].count) % msk_ctx.ranges[msk_ctx.activeRangePos[1]].count;
		key[msk_ctx.activeRangePos[1] + offset] = msk_ctx.ranges[msk_ctx.activeRangePos[1]].chars[j];
	}
	if (msk_ctx.ranges[msk_ctx.activeRangePos[2]].count) {
		k = (ctr / (msk_ctx.ranges[msk_ctx.activeRangePos[0]].count * msk_ctx.ranges[msk_ctx.activeRangePos[1]].count)) % msk_ctx.ranges[msk_ctx.activeRangePos[2]].count;
		key[msk_ctx.activeRangePos[2] + offset] = msk_ctx.ranges[msk_ctx.activeRangePos[2]].chars[k];
	}
}

static char *get_key(int index)
{
	static char out[PLAINTEXT_LENGTH + 1];
	int i;
	int  len, ctr = 0, mask_offset = 0, flag = 0;
	char *key;

	if((index < loaded_count) && cmp_out) {
		ctr = outKeyIdx[index + loaded_count];
		/* outKeyIdx contains all zero when no new passwords are cracked.
		 * Hence during status checks even if index is less than loaded count
		 * correct range of passwords is displayed.
		 */
		index = outKeyIdx[index] & 0x7fffffff;
		mask_offset = mask_offsets[index];
		flag = 1;
	}
	index = (index > num_keys)? (num_keys?num_keys-1:0): index;
	len = saved_idx[index] & 63;
	key = (char*)&saved_plain[saved_idx[index] >> 6];

	for (i = 0; i < len; i++)
		out[i] = *key++;

	if(cmp_out && mask_mode && flag)
		passgen(ctr, mask_offset, out);

	out[i] = 0;

	return out;
}

static int crypt_all_self_test(int *pcount, struct db_salt *salt)
{
	int count = *pcount;

	global_work_size = (count + local_work_size - 1) / local_work_size * local_work_size;

	// copy keys to the device
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], buffer_keys, CL_TRUE, 0, 4 * key_idx, saved_plain, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_keys");
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], buffer_idx, CL_TRUE, 0, sizeof(uint64_t) * global_work_size, saved_idx, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_idx");

	HANDLE_CLERROR(clEnqueueNDRangeKernel(queue[ocl_gpu_id], crypt_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL), "failed in clEnqueueNDRangeKernel");

	// read back partial hashes
	HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], buffer_out, CL_TRUE, 0, sizeof(cl_uint) * global_work_size, partial_hashes, 0, NULL, NULL), "failed in reading data back");
	have_full_hashes = 0;

	return count;
}

static int crypt_all(int *pcount, struct db_salt *salt)
{
	int count = *pcount;
	unsigned int i;
	cl_event evnt;

	global_work_size = (count + local_work_size - 1) / local_work_size * local_work_size;

	if(mask_mode)
		*pcount *= multiplier;

	if(loaded_count != (salt->count)) {
		load_hash(salt);
		load_bitmap(loaded_count, 0, &bitmap1[0].bitmap0[0], (BITMAP_SIZE_1 / 8));
		load_bitmap(loaded_count, 1, &bitmap1[0].bitmap1[0], (BITMAP_SIZE_1 / 8));
		load_bitmap(loaded_count, 2, &bitmap1[0].bitmap2[0], (BITMAP_SIZE_1 / 8));
		load_bitmap(loaded_count, 3, &bitmap1[0].bitmap3[0], (BITMAP_SIZE_1 / 8));
		load_bitmap(loaded_count, 0, &bitmap1[0].gbitmap0[0], (BITMAP_SIZE_3 / 8));
		load_hashtable_plus(&bitmap2[0].hashtable0[0], &bitmap1[0].loaded_next_hash[0], 2, loaded_count, HASH_TABLE_SIZE_0);
		HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], buffer_bitmap1, CL_TRUE, 0, sizeof(struct bitmap_context_mixed), bitmap1, 0, NULL, NULL ), "Failed Copy data to gpu");
		HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], buffer_bitmap2, CL_TRUE, 0, sizeof(struct bitmap_context_global), bitmap2, 0, NULL, NULL ), "Failed Copy data to gpu");
	}
	// copy keys to the device
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], buffer_keys, CL_TRUE, 0, 4 * key_idx, saved_plain, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_keys");
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], buffer_idx, CL_TRUE, 0, sizeof(uint64_t) * global_work_size, saved_idx, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_idx");

	if(msk_ctx.flg_wrd)
		HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], buffer_outKeyIdx, CL_TRUE, 0,
			(DB->format->params.max_keys_per_crypt), mask_offset_buffer, 0, NULL, NULL),
			"failed in clEnqueWriteBuffer buffer_outKeyIdx");

	HANDLE_CLERROR(clEnqueueNDRangeKernel(queue[ocl_gpu_id], crk_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &evnt), "failed in clEnqueueNDRangeKernel");

	HANDLE_CLERROR(clWaitForEvents(1, &evnt), "Wait for event failed");

	// read back compare results
	HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], buffer_outKeyIdx, CL_TRUE, 0, sizeof(cl_uint) * loaded_count, outKeyIdx, 0, NULL, NULL), "failed in reading cmp data back");

	cmp_out = 0;

	// If a positive match is found outKeyIdx[i] contains 0xffffffff else contains 0
	for(i = 0; i < (loaded_count & (~cmp_out)); i++)
		cmp_out = outKeyIdx[i]?0xffffffff:0;

	have_full_hashes = 0;

	// If any positive match is found
	if(cmp_out) {
		//HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], buffer_out, CL_TRUE, 0, sizeof(cl_uint) * loaded_count, partial_hashes, 0, NULL, NULL), "failed in reading hashes back");
		HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], buffer_outKeyIdx, CL_TRUE, 0, sizeof(cl_uint) * loaded_count * 2, outKeyIdx, 0, NULL, NULL), "failed in reading cmp data back");
		HANDLE_CLERROR(clFinish(queue[ocl_gpu_id]), "cl finish failed");
		for(i = 0; i < loaded_count; i++) {
			if(outKeyIdx[i])
				partial_hashes[i] = loaded_hashes[i+1];
			else partial_hashes[i] = 0;
		}
		if(msk_ctx.flg_wrd)
			memcpy(mask_offsets, mask_offset_buffer, (DB->format->params.max_keys_per_crypt));
		return loaded_count;
	}

	else {
		HANDLE_CLERROR(clFinish(queue[ocl_gpu_id]), "cl finish failed");
		return 0;
	}
}

static int cmp_all(void *binary, int count)
{
	if(self_test) {
		unsigned int i;
		unsigned int b = ((unsigned int *) binary)[0];

		for (i = 0; i < count; i++)
			if (b == partial_hashes[i])
				return 1;
		return 0;
	}

	else return 1;
}

static int cmp_one(void *binary, int index)
{
	return (((unsigned int*)binary)[0] == partial_hashes[index]);
}

static int cmp_exact(char *source, int index)
{
	unsigned int *t = (unsigned int *) get_binary(source);
	unsigned int count = self_test ? global_work_size: loaded_count;
	if (self_test) {
		if (!have_full_hashes) {
			clEnqueueReadBuffer(queue[ocl_gpu_id], buffer_out, CL_TRUE,
				sizeof(cl_uint) * (count),
				sizeof(cl_uint) * 3 * count,
				res_hashes, 0, NULL, NULL);
				have_full_hashes = 1;
		}

		if (t[1]!=res_hashes[index])
			return 0;
		if (t[2]!=res_hashes[1*count+index])
			return 0;
		if (t[3]!=res_hashes[2*count+index])
			return 0;
		return 1;
	}

	else {
		if(!outKeyIdx[index]) return 0;
		if (t[1]!=loaded_hashes[index + count + 1])
			return 0;
		if (t[2]!=loaded_hashes[2 * count + index +1])
			return 0;
		if (t[3]!=loaded_hashes[3*count + index + 1])
			return 0;
		return 1;
	}
}

struct fmt_main fmt_opencl_rawMD5 = {
	{
		FORMAT_LABEL,
		FORMAT_NAME,
		ALGORITHM_NAME,
		BENCHMARK_COMMENT,
		BENCHMARK_LENGTH,
		PLAINTEXT_LENGTH,
		BINARY_SIZE,
		BINARY_ALIGN,
		SALT_SIZE,
		SALT_ALIGN,
		MAX_KEYS_PER_CRYPT,
		MAX_KEYS_PER_CRYPT,
		(26*26*10),
		FMT_CASE | FMT_8_BIT,
		tests
	}, {
		init,
		done,
		opencl_md5_reset,
		fmt_default_prepare,
		valid,
		split,
		(void *(*)(char *))get_binary,
		fmt_default_salt,
		fmt_default_source,
		{
			fmt_default_binary_hash_0,
			fmt_default_binary_hash_1,
			fmt_default_binary_hash_2,
			fmt_default_binary_hash_3,
			fmt_default_binary_hash_4,
			fmt_default_binary_hash_5,
			fmt_default_binary_hash_6
		},
		fmt_default_salt_hash,
		fmt_default_set_salt,
		set_key,
		get_key,
		clear_keys,
		crypt_all_self_test,
		{
			get_hash_0,
			get_hash_1,
			get_hash_2,
			get_hash_3,
			get_hash_4,
			get_hash_5,
			get_hash_6
		},
		cmp_all,
		cmp_one,
		cmp_exact
	}
};
