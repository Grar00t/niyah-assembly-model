#include "morph.h"

/* Literal Unicode scalars and 1/2/3 captures form an anchored, regex-like table.
 * Repeated captures must agree. No byte-regex engine or libc is needed. */
static const struct { const char *shape; MorphPattern id; u32 flags; } PATTERNS[] = {
    { "1ا23", MP_FAAIL, 0 }, { "م12و3", MP_MAFOOL, 0 },
    { "م123", MP_MAFAL, 0 }, { "م1ت23", MP_MUFTAAL, 0 },
    { "مست123", MP_MUSTAFIL, 0 },
    { "أ123", MP_AFAL, MF_PLURAL_CANDIDATE },
    { "أ12ا3", MP_AFAAL, MF_PLURAL_CANDIDATE },
    { "12و3", MP_FUOOL, MF_PLURAL_CANDIDATE },
    { "12ا3", MP_FIAAL, MF_PLURAL_CANDIDATE },
    { "م1ا23", MP_MAFAAIL, MF_PLURAL_CANDIDATE }
};
/* Tiny seed lexicon: stable IDs, not general Arabic verb recognition.
 * Weak/hamzated/irregular roots and inflected subject endings are out of scope. */
static const struct { u32 root[3]; u32 id; } VERBS[] = {
    { {0x643,0x62A,0x628}, 1 }, /* كتب */
    { {0x62F,0x631,0x633}, 2 }, /* درس */
    { {0x639,0x645,0x644}, 3 }, /* عمل */
    { {0x641,0x647,0x645}, 4 }, /* فهم */
    { {0x62F,0x62E,0x644}, 5 }, /* دخل */
    { {0x62E,0x631,0x62C}, 6 }, /* خرج */
    { {0x644,0x639,0x628}, 7 }, /* لعب */
    { {0x62D,0x645,0x644}, 8 }, /* حمل */
    { {0x633,0x645,0x639}, 9 }, /* سمع */
    { {0x634,0x631,0x628}, 10 } /* شرب */
};
static const struct { const char *text; MorphId id; } SUFFIXES[] = {
    { "كم", M_PRON_KUM }, { "ها", M_PRON_HA }, { "نا", M_PRON_NA },
    { "ه", M_PRON_H }, { "ك", M_PRON_K }
};
typedef struct {
    u32 len, n, cp[MORPH_MAX_BASES], boundary[MORPH_MAX_BASES + 1];
} Word;
typedef struct { MorphPart parts[MORPH_MAX_PARTS]; u32 count, pos, flags; } Prefix;

static u32 text_len(const char *s) { u32 n = 0; while (s[n]) n++; return n; }
static int decode(const u8 *s, u32 len, u32 *pos, u32 *cp) {
    u32 i = *pos, c, v, n, j;
    if (i >= len) return 0;
    c = s[i];
    if (c < 0x80) { *cp = c; *pos = i + 1; return 1; }
    if (c >= 0xC2 && c <= 0xDF) { n = 2; v = c & 31u; }
    else if (c >= 0xE0 && c <= 0xEF) { n = 3; v = c & 15u; }
    else if (c >= 0xF0 && c <= 0xF4) { n = 4; v = c & 7u; }
    else return -1;
    if (n > len - i) return -1;
    for (j = 1; j < n; j++) {
        if ((s[i+j] & 0xC0u) != 0x80u) return -1;
        v = (v << 6) | (s[i+j] & 63u);
    }
    if ((n == 2 && v < 0x80) || (n == 3 && v < 0x800) ||
        (n == 4 && v < 0x10000) || (v >= 0xD800 && v <= 0xDFFF) || v > 0x10FFFF) return -1;
    *pos = i + n; *cp = v; return 1;
}
static int mark(u32 c) { return (c >= 0x64B && c <= 0x65F) || c == 0x670 || c == 0x640; }
static int arabic(u32 c) {
    return (c >= 0x621 && c <= 0x63A) || (c >= 0x641 && c <= 0x64A) ||
           (c >= 0x66E && c <= 0x66F) || (c >= 0x671 && c <= 0x6D3) ||
           (c >= 0x6FA && c <= 0x6FC);
}
static int root_letter(u32 c) { return arabic(c); }
static u32 verb_id(const u32 root[3]) {
    u32 i;
    for (i = 0; i < sizeof VERBS / sizeof VERBS[0]; i++)
        if (VERBS[i].root[0] == root[0] && VERBS[i].root[1] == root[1] &&
            VERBS[i].root[2] == root[2]) return VERBS[i].id;
    return 0;
}
static void clear_analysis(MorphAnalysis *a) {
    u32 i;
    a->count = a->flags = a->root_len = a->verb_id = 0; a->pattern = MP_NONE;
    for (i = 0; i < 3; i++) a->root[i] = 0;
    for (i = 0; i < MORPH_MAX_PARTS; i++) {
        a->parts[i].offset = a->parts[i].len = 0; a->parts[i].id = M_STEM;
    }
}
static int same(const MorphAnalysis *a, const MorphAnalysis *b) {
    u32 i;
    if (a->count != b->count || a->flags != b->flags || a->pattern != b->pattern ||
        a->verb_id != b->verb_id || a->root_len != b->root_len) return 0;
    for (i = 0; i < a->count; i++)
        if (a->parts[i].offset != b->parts[i].offset || a->parts[i].len != b->parts[i].len ||
            a->parts[i].id != b->parts[i].id) return 0;
    for (i = 0; i < 3; i++) if (a->root[i] != b->root[i]) return 0;
    return 1;
}
static void emit(MorphResult *out, const MorphAnalysis *a) {
    u32 i;
    for (i = 0; i < out->count; i++) if (same(a, &out->analyses[i])) return;
    if (out->count == MORPH_MAX_ANALYSES) { out->truncated = 1; return; }
    out->analyses[out->count++] = *a;
}
static int add_prefix(Prefix *p, const Word *w, u32 width, MorphId id) {
    MorphPart *part;
    if (!width || p->pos > w->n || width > w->n - p->pos || p->count >= MORPH_MAX_PARTS - 2) return 0;
    part = &p->parts[p->count++]; part->offset = w->boundary[p->pos];
    p->pos += width; part->len = w->boundary[p->pos] - part->offset; part->id = id;
    p->flags |= MF_SEGMENTED; return 1;
}
static int tail_match(const Word *w, u32 begin, const char *s, u32 *end) {
    u32 chars[4], n = 0, p = 0, c, i, len = text_len(s);
    while (decode((const u8 *)s, len, &p, &c) > 0) {
        if (n == 4) return 0;
        chars[n++] = c;
    }
    if (w->n < begin || n > w->n - begin || w->n - begin - n < 3) return 0;
    for (i = 0; i < n; i++) if (chars[i] != w->cp[w->n - n + i]) return 0;
    *end = w->n - n; return 1;
}
static int pattern_match(const Word *w, u32 begin, u32 end, const char *shape, u32 root[3]) {
    u32 p = 0, i = begin, c, len = text_len(shape), seen = 0, j;
    for (j = 0; j < 3; j++) root[j] = 0;
    while (decode((const u8 *)shape, len, &p, &c) > 0) {
        if (i == end) return 0;
        if (c >= '1' && c <= '3') {
            j = c - '1';
            if (!root_letter(w->cp[i]) || ((seen & (1u << j)) && root[j] != w->cp[i])) return 0;
            root[j] = w->cp[i]; seen |= 1u << j;
        } else if (c != w->cp[i]) return 0;
        i++;
    }
    return i == end && seen == 7;
}
static void stem_candidates(const Word *w, const Prefix *p, u32 end, MorphId suffix,
                            int verbal, int article, MorphResult *out) {
    MorphAnalysis a, b;
    u32 i, n = end - p->pos, id = 0;
    if (n < 3) return;
    if (n == 3) id = verb_id(&w->cp[p->pos]);
    /* Present-tense stripping is guarded by the seed lexicon, never initial letters alone. */
    if (verbal && !id) return;
    for (i = 0; i < p->count; i++)
        if (p->parts[i].id == M_PREP_B || p->parts[i].id == M_PREP_K ||
            p->parts[i].id == M_PREP_L) { id = 0; break; }
    clear_analysis(&a); a.flags = p->flags; a.count = p->count;
    for (i = 0; i < p->count; i++) a.parts[i] = p->parts[i];
    a.parts[a.count].offset = w->boundary[p->pos];
    a.parts[a.count].len = w->boundary[end] - w->boundary[p->pos];
    a.parts[a.count++].id = M_STEM;
    if (end < w->n) {
        if (article) return; /* definite article + possessive suffix is not accepted here */
        a.parts[a.count].offset = w->boundary[end];
        a.parts[a.count].len = w->len - w->boundary[end]; a.parts[a.count++].id = suffix;
        a.flags |= MF_SEGMENTED | (verbal ? MF_PRON_OBJECT_CANDIDATE : MF_PRON_AMBIGUOUS);
    }
    if (verbal) {
        a.flags |= MF_VERB_CANDIDATE; a.verb_id = id; a.root_len = 3;
        for (i = 0; i < 3; i++) a.root[i] = w->cp[p->pos+i];
    }
    if (a.flags) emit(out, &a);
    if (id && !article && !verbal) {
        b = a; b.flags |= MF_VERB_CANDIDATE; b.verb_id = id;
        for (i = 0; i < 3; i++) b.root[i] = w->cp[p->pos+i];
        b.root_len = 3; emit(out, &b);
    }
    if (verbal) return;
    for (i = 0; i < sizeof PATTERNS / sizeof PATTERNS[0]; i++) {
        b = a;
        if (!pattern_match(w, p->pos, end, PATTERNS[i].shape, b.root)) continue;
        b.root_len = 3; b.pattern = PATTERNS[i].id;
        b.flags |= MF_PATTERN_CANDIDATE | PATTERNS[i].flags;
        /* Related root is not a claim that the derived surface is a verb. */
        b.verb_id = 0; emit(out, &b);
    }
}
static void endings(const Word *w, const Prefix *p, int verbal, int article, MorphResult *out) {
    u32 i, end;
    stem_candidates(w, p, w->n, M_STEM, verbal, article, out);
    if (article) return;
    for (i = 0; i < sizeof SUFFIXES / sizeof SUFFIXES[0]; i++)
        if (tail_match(w, p->pos, SUFFIXES[i].text, &end))
            stem_candidates(w, p, end, SUFFIXES[i].id, verbal, article, out);
}
static void verb_prefixes(const Word *w, const Prefix *p, MorphResult *out) {
    Prefix v = *p;
    u32 c; MorphId id;
    if (v.pos < w->n && w->cp[v.pos] == 0x633) {
        if (!add_prefix(&v, w, 1, M_FUTURE_S)) return;
    }
    if (v.pos >= w->n) return;
    c = w->cp[v.pos];
    if (c == 0x62A) id = M_IMPF_T;
    else if (c == 0x646) id = M_IMPF_N;
    else if (c == 0x64A) id = M_IMPF_Y;
    else if (c == 0x623 || c == 0x627) id = M_IMPF_A;
    else return;
    if (!add_prefix(&v, w, 1, id)) return;
    endings(w, &v, 1, 0, out);
}
static void nominal_prefixes(const Word *w, const Prefix *p, int prep, MorphResult *out) {
    Prefix v = *p;
    endings(w, p, 0, 0, out);
    if (p->pos + 2 <= w->n && w->cp[p->pos] == 0x627 && w->cp[p->pos+1] == 0x644) {
        if (add_prefix(&v, w, 2, M_DET_AL)) endings(w, &v, 0, 1, out);
    } else if (prep && p->count && p->parts[p->count-1].id == M_PREP_L &&
               p->pos < w->n && w->cp[p->pos] == 0x644) {
        if (add_prefix(&v, w, 1, M_DET_AL)) {
            v.flags |= MF_ARTICLE_CONTRACTED; endings(w, &v, 0, 1, out);
        }
    }
}
static void after_conjunction(const Word *w, const Prefix *p, MorphResult *out) {
    Prefix v = *p; MorphId id;
    nominal_prefixes(w, p, 0, out); verb_prefixes(w, p, out);
    if (p->pos >= w->n) return;
    if (w->cp[p->pos] == 0x628) id = M_PREP_B;
    else if (w->cp[p->pos] == 0x643) id = M_PREP_K;
    else if (w->cp[p->pos] == 0x644) id = M_PREP_L;
    else return;
    if (add_prefix(&v, w, 1, id)) nominal_prefixes(w, &v, 1, out);
}
MorphStatus morph_analyze(const u8 *word, u32 len, MorphResult *out) {
    Word w; Prefix p; MorphAnalysis a;
    u32 pos = 0, c, start, i; int all_arabic = 1, overflow = 0, rc;
    if (!out) return MORPH_BAD_ARGUMENT;
    out->count = out->truncated = 0;
    if (!word) return MORPH_BAD_ARGUMENT;
    w.len = len; w.n = 0;
    while (pos < len) {
        start = pos; rc = decode(word, len, &pos, &c);
        if (rc < 0) return MORPH_INVALID_UTF8;
        if (mark(c) && w.n) continue;
        if (!arabic(c)) all_arabic = 0;
        if (w.n == MORPH_MAX_BASES) { overflow = 1; continue; }
        w.boundary[w.n] = start; w.cp[w.n++] = c;
    }
    w.boundary[w.n] = len;
    clear_analysis(&a); a.flags = MF_RAW;
    if (len) { a.count = 1; a.parts[0].len = len; a.parts[0].id = M_STEM; }
    emit(out, &a);
    if (overflow) { out->truncated = 1; return MORPH_TOO_LONG; }
    /* Canonical ID normalizes the prefix; source bytes are NEVER rewritten.
     * ASCII Al-/al-/AL- is an explicit transliteration convention, not a detector. */
    if (len > 3 && (word[0] == 'A' || word[0] == 'a') &&
        (word[1] == 'L' || word[1] == 'l') && word[2] == '-') {
        clear_analysis(&a); a.flags = MF_SEGMENTED | MF_ASCII_AL; a.count = 2;
        a.parts[0].id = M_DET_AL; a.parts[0].len = 3;
        a.parts[1].id = M_STEM; a.parts[1].offset = 3; a.parts[1].len = len - 3;
        emit(out, &a); return MORPH_OK;
    }
    if (!all_arabic || w.n < 3) return MORPH_OK;
    if (w.n == 3 && w.cp[0] == 0x633 && w.cp[1] == 0x648 && w.cp[2] == 0x641) {
        clear_analysis(&a); a.flags = MF_FUTURE_PARTICLE; a.count = 1;
        a.parts[0].id = M_FUTURE_SAWFA; a.parts[0].len = len; emit(out, &a);
    }
    p.count = p.pos = p.flags = 0;
    for (i = 0; i < MORPH_MAX_PARTS; i++) {
        p.parts[i].offset = p.parts[i].len = 0; p.parts[i].id = M_STEM;
    }
    after_conjunction(&w, &p, out);
    if (w.cp[0] == 0x648 || w.cp[0] == 0x641) {
        if (add_prefix(&p, &w, 1, w.cp[0] == 0x648 ? M_CONJ_W : M_CONJ_F))
            after_conjunction(&w, &p, out);
    }
    return MORPH_OK;
}
const char *morph_id_name(MorphId id) {
    static const char *const names[] = {
        "STEM", "CONJ_W", "CONJ_F", "PREP_B", "PREP_K", "PREP_L", "DET_AL",
        "FUTURE_S", "FUTURE_SAWFA", "IMPF_T", "IMPF_N", "IMPF_Y", "IMPF_A",
        "PRON_H", "PRON_HA", "PRON_K", "PRON_KUM", "PRON_NA"
    };
    return (u32)id < sizeof names / sizeof names[0] ? names[id] : "?";
}
const char *morph_pattern_name(MorphPattern id) {
    static const char *const names[] = {
        "NONE", "FAAIL", "MAFOOL", "MAFAL", "MUFTAAL", "MUSTAFIL",
        "AFAL", "AFAAL", "FUOOL", "FIAAL", "MAFAAIL"
    };
    return (u32)id < sizeof names / sizeof names[0] ? names[id] : "?";
}
