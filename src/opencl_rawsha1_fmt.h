#define LWS 		    64
#define BITMAP_SIZE_0 	    0x80000000
#define BITMAP_SIZE_1	    0x2000
#define BITMAP_SIZE_3       0x4000000
#define HASH_TABLE_SIZE_0   0x8000000
#define MAX_LOADED_HASHES   0x2000000

struct bitmap_context_mixed{
	unsigned int bitmap0[BITMAP_SIZE_1>>5];
	unsigned int bitmap1[BITMAP_SIZE_1>>5];
	unsigned int bitmap2[BITMAP_SIZE_1>>5];
	unsigned int bitmap3[BITMAP_SIZE_1>>5];
	unsigned int gbitmap0[BITMAP_SIZE_3>>5]; //4 MB
	unsigned int loaded_next_hash[MAX_LOADED_HASHES]; //128 MB
};

struct bitmap_context_global {
	unsigned int hashtable0[HASH_TABLE_SIZE_0]; //512 MB
};