/*
 * This file is part of John the Ripper password cracker,
 * Copyright (c) 1996-98 by Solar Designer
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * This software was written by Jim Fougeron jfoug AT cox dot net
 * in 2014. No copyright is claimed, and the software is hereby
 * placed in the public domain. In case this attempt to disclaim
 * copyright and place the software in the public domain is deemed
 * null and void, then the software is Copyright (c) 2014 Jim Fougeron
 * and it is hereby released to the general public under the following
 * terms:
 *
 * This software may be modified, redistributed, and used for any
 * purpose, in source and binary forms, with or without modification.
 */

#ifndef _JOHN_FORMAT_ALIAS_H
#define _JOHN_FORMAT_ALIAS_H

struct fmt_main;

extern void fmt_alias_init();
extern struct fmt_main *fmt_alias_check(struct fmt_main *pFmt, const char *hash, char *fields[10]);
extern struct fmt_main *alias_format_by_idx(char *ciphertext, struct fmt_main *pFmt, int idx);
extern int dynamics_equal(const char *sig1, const char *sig2);

#endif
