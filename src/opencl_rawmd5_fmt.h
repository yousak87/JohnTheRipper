#define LWS 		    64
#define BITMAP_SIZE_0 	    0x80000000
#define BITMAP_SIZE_1	    0x2000
#define BITMAP_SIZE_2	    0x10000

struct bitmap_ctx{
	unsigned int bitmap0[BITMAP_SIZE_1>>5];
	unsigned int bitmap1[BITMAP_SIZE_1>>5];
	unsigned int bitmap2[BITMAP_SIZE_1>>5];
	unsigned int bitmap3[BITMAP_SIZE_1>>5];
	unsigned int gbitmap0[BITMAP_SIZE_0>>5];
	//unsigned int gbitmap1[BITMAP_SIZE_0>>5];

};
