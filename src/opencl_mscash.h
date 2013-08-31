/*
 * This software is Copyright (c) 2013 Sayantan Datta <std2048 at gmail dot com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification, are permitted.
 * This is format is based on mscash-cuda by Lukas Odzioba
 * <lukas dot odzioba at gmail dot com>
 */
#ifndef _MSCASH_H
#define _MSCASH_H

#define LWS 		    	64

#define	KEYS_PER_CRYPT		(1024 * 2048)

#define BINARY_SIZE		16
#define BINARY_ALIGN		4
#define PLAINTEXT_LENGTH	27
#define SALT_LENGTH		19
#define SALT_SIZE		sizeof(mscash_salt)

#define MIN_KEYS_PER_CRYPT	KEYS_PER_CRYPT
#define MAX_KEYS_PER_CRYPT	KEYS_PER_CRYPT

#define MAX(x,y)		((x) > (y) ? (x) : (y))
#define MIN(x,y)		((x) < (y) ? (x) : (y))
#define SHOW(x)			(printf("%s = %08x\n",#x,(x)))
# define SWAP(n) \
    (((n) << 24) | (((n) & 0xff00) << 8) | (((n) >> 8) & 0xff00) | ((n) >> 24))

#define INIT_A			0x67452301
#define INIT_B			0xefcdab89
#define INIT_C			0x98badcfe
#define INIT_D			0x10325476

#define SQRT_2			0x5a827999
#define SQRT_3			0x6ed9eba1

#define BITMAP_SIZE_0 	    	0x80000000
#define BITMAP_SIZE_1	    	0x2000
#define BITMAP_SIZE_3       	0x4000000
#define MAX_LOADED_HASHES	0x1000000
#define HASH_TABLE_SIZE_0   	0x2000000

struct bitmap_context_mixed {
	unsigned int bitmap0[BITMAP_SIZE_1>>5];
	unsigned int bitmap1[BITMAP_SIZE_1>>5];
	unsigned int bitmap2[BITMAP_SIZE_1>>5];
	unsigned int bitmap3[BITMAP_SIZE_1>>5];
	unsigned int gbitmap0[BITMAP_SIZE_3>>5]; //4 MB
	unsigned int loaded_next_hash[MAX_LOADED_HASHES]; //64 MB
};

struct bitmap_context_global {
	unsigned int hashtable0[HASH_TABLE_SIZE_0]; //128 MB
};

static
#ifdef _OPENCL_COMPILER
__constant
#endif
const char mscash_prefix[] = "M$";

#endif
