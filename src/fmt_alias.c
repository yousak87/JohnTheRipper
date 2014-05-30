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

#include <stdio.h>
#include <string.h>

#include "params.h"
#include "memory.h"
#include "formats.h"
#include "config.h"
#include "fmt_externs.h"
#include "dynamic.h"
#include "fmt_alias.h"

struct alias_list {
	struct alias_list *next;
	struct fmt_main *fmt;
};
static struct alias_list *a_list[250];  // up to 250 aliases at this time.
static int eqDynas[250][16]; // up to 16 equal dyna's per alias type
static int nAlias, nCurList=-1, bInit;

//#define ALIAS_DBG
#ifdef ALIAS_DBG
#define PRINTF(a) printf(a)
#define PRINTF2(a,b) printf(a,b)
#define PRINTF3(a,b,c) printf(a,b,c)
#else
#define PRINTF(a)
#define PRINTF2(a,b)
#define PRINTF3(a,b,c)
#endif

static void fmt_alias_load(const char *alias_list) {
	struct fmt_main *p;
	struct alias_list *tail; 
	char *cp, *Buf = (char*)malloc(strlen(alias_list)+1);
	int cnt = 0;
	int nDynaCnt = 0;

	// TBD  We need some way (option) to force NOT performing alias stuff.
	// there are times when a user would NOT want alias code to happen.

	PRINTF ("In fmt_alias_init\n");
	strcpy(Buf, alias_list);

	cp = strtok(Buf, " ,");
	a_list[nAlias] = NULL;
	while (cp) {
		p = fmt_alias_list;
		while (p && strcasecmp(p->params.label, cp)) {
			p = p->alias_next;
		}
		if (!p) {
			PRINTF2("COULD NOT FIND first alias %s\n", cp);
		}
		else {
			tail = (struct alias_list *)mem_alloc_tiny(sizeof(struct alias_list),MEM_ALIGN_WORD);
			tail->next = a_list[nAlias];
			tail->fmt = p;
			a_list[nAlias] = tail;
			++cnt;
			if ( (p->params.flags & FMT_DYNAMIC) == FMT_DYNAMIC) {
				eqDynas[nAlias][nDynaCnt++] = dyna_get_number(p);
			}
		}
		cp = strtok(NULL, " ,");
	}
	eqDynas[nAlias][nDynaCnt] = -1;
	if (cnt > 1)
		++nAlias;
	free(Buf);
}

void fmt_alias_init() {
	struct cfg_list *list;
	struct cfg_line *line;

	if (bInit) return;
	bInit = 1;
	if ((list = cfg_get_list("List.Alias:", "Aliases"))) {
		if ((line = list->head)) {
			while (line) {
				fmt_alias_load(line->data);
				line = line->next;
			}
		}
	}
}

struct fmt_main *fmt_alias_check(struct fmt_main *pFmt, const char *hash, char *fields[10]) {
	int i, valid;
	struct alias_list *p;
	char *prepared;

	PRINTF3 ("In fmt_alias_check, checking %s for %s\n", hash, pFmt->params.label);
	if (nCurList < 0) {
		if (nCurList == -2) {
			//printf ("No alias\n");
			return 0; // our format has no alias.
		}
		// find our format in the alias list.  If we do not find it,
		// set nCurList = -2, so we ignore alias logic for this format.
		PRINTF2 ("nAlias=%d\n", nAlias);
		for (i = 0; i < nAlias; ++i) {
			p = a_list[i];
			while (p) {
				if (!strcmp(pFmt->params.label, p->fmt->params.label)) {
					// found it!
					nCurList = i;
					PRINTF2 ("Found our format in alias[%d]\n", i);
					break;
				}
				p = p->next;
			}
			if (nCurList != -1) {
				break;
			}
		}
		if (nCurList == -1) {
			PRINTF2 ("Could not find alias  nAlias=%d\n", nAlias);
			nCurList = -2;
			return 0;
		}
	}
	// Ok, we know that our format has an alias, and now we 'check'.
	PRINTF2 ("Checking alias[%d] for a valid format\n", nCurList);
	p = a_list[nCurList];
	while (p) {
		if (strcmp(pFmt->params.label, p->fmt->params.label)) {
			prepared = p->fmt->methods.prepare(fields, p->fmt);
			if (prepared) {
				valid = p->fmt->methods.valid(prepared, p->fmt);
				if (valid)
					return p->fmt;
			}
		}
		p = p->next;
	}
	return 0;
}

int dynamics_equal(const char *sig1, const char *sig2) {
	int n1, n2, i, cnt, idx;

	n1 = -1;
	sscanf(sig1, "$dynamic_%d$", &n1);
	if (n1 == -1) return 0;
	PRINTF3 ("In dynamics_equal, checking %s for %s\n", sig1, sig2);
	n2 = -1;
	sscanf(sig2, "$dynamic_%d$", &n2);
	if (n2 == -1) return 0;
	for (idx = 0; idx < nAlias; ++idx) {
		cnt = 0;
		for (i = 0; eqDynas[idx][i] != -1; ++i) {
			if (eqDynas[idx][i] == n1) ++cnt;
			if (eqDynas[idx][i] == n2) ++cnt;
		}
		if (cnt == 1) return 0;
		if (cnt == 2) return 1;
	}
	return 0;
}

struct fmt_main *alias_format_by_idx(char *ciphertext, struct fmt_main *pFmt, int idx) {
	int i;
	struct alias_list *p;
	int nCurList = -1;

	for (i = 0; i < nAlias; ++i) {
		p = a_list[i];
		while (p) {
			if (!strcmp(pFmt->params.label, p->fmt->params.label)) {
				// found it!
				nCurList = i;
				PRINTF2 ("Found our format in alias[%d]\n", i);
				break;
			}
			p = p->next;
		}
		if (nCurList != -1) {
			int j = 0;
			p = a_list[i];
			if (!p)
				return 0;
			while (j < idx) {
				p = p->next;
				if (!p)
					return 0;
				++j;
			}
			return p->fmt;
		}
	}
	return 0;
}
