/* Work group size is hardcoded because it is
 * used inside the kernel for allocating local memory.*/
#define WORK_GROUP_SIZE		64

/* Set it to 1 when very few(< 100) salts are
 * loaded for cracking */
#define HARDCODE_SALT 		0

/* Currently there are no devices which benifits
 * from a fully unrolled kernel */
#define FULL_UNROLL		0

/* If 1, only mask mode kernels are built. */
 #define MASK_MODE_ONLY          0
