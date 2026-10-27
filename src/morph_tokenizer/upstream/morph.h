#ifndef NIYAH_MORPH_H
#define NIYAH_MORPH_H
#include "lexer.h"

/* Versioned annotation IDs, NOT a trained model vocabulary. */
#define MORPH_SCHEMA_VERSION 1u
#define MORPH_MAX_PARTS 8u
#define MORPH_MAX_ANALYSES 64u
#define MORPH_MAX_BASES 96u

typedef enum {
    M_STEM = 0, M_CONJ_W = 1, M_CONJ_F = 2,
    M_PREP_B = 3, M_PREP_K = 4, M_PREP_L = 5, M_DET_AL = 6,
    M_FUTURE_S = 7, M_FUTURE_SAWFA = 8,
    M_IMPF_T = 9, M_IMPF_N = 10, M_IMPF_Y = 11, M_IMPF_A = 12,
    M_PRON_H = 13, M_PRON_HA = 14, M_PRON_K = 15,
    M_PRON_KUM = 16, M_PRON_NA = 17
} MorphId;

typedef enum {
    MP_NONE = 0, MP_FAAIL = 1, MP_MAFOOL = 2, MP_MAFAL = 3,
    MP_MUFTAAL = 4, MP_MUSTAFIL = 5, MP_AFAL = 6,
    MP_AFAAL = 7, MP_FUOOL = 8, MP_FIAAL = 9, MP_MAFAAIL = 10
} MorphPattern;

enum {
    MF_RAW = 1u, MF_SEGMENTED = 2u, MF_VERB_CANDIDATE = 4u,
    MF_PATTERN_CANDIDATE = 8u, MF_PLURAL_CANDIDATE = 16u,
    MF_PRON_AMBIGUOUS = 32u, MF_PRON_OBJECT_CANDIDATE = 64u,
    MF_ARTICLE_CONTRACTED = 128u, MF_ASCII_AL = 256u,
    MF_FUTURE_PARTICLE = 512u
};

typedef struct { u32 offset, len; MorphId id; } MorphPart;
typedef struct {
    MorphPart parts[MORPH_MAX_PARTS];
    u32 count, flags;
    MorphPattern pattern;
    u32 root[3], root_len;
    u32 verb_id; /* 0 = unknown; nonzero is only a candidate lexical link. */
} MorphAnalysis;
typedef struct {
    MorphAnalysis analyses[MORPH_MAX_ANALYSES];
    u32 count, truncated;
} MorphResult;
typedef enum { MORPH_OK = 0, MORPH_INVALID_UTF8 = 1,
               MORPH_TOO_LONG = 2, MORPH_BAD_ARGUMENT = 3 } MorphStatus;

/* No mutation/allocation. Every analysis partitions the ORIGINAL word bytes.
 * No analysis is chosen as correct. The unmodified candidate is always first.
 * Input and output must be non-NULL, non-overlapping; input length is explicit.
 * On TOO_LONG the intact fallback is still returned. Invalid UTF-8 has no result.
 * Roots/patterns are hypotheses, not semantic or contextual disambiguation. */
MorphStatus morph_analyze(const u8 *word, u32 len, MorphResult *out);
const char *morph_id_name(MorphId id);
const char *morph_pattern_name(MorphPattern pattern);
#endif
