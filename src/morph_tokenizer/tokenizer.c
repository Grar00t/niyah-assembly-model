/* Exact surface-token codec; all I/O uses Linux x86-64 syscalls.
 * Morphology annotations are candidates. Compression selects a boundary
 * analysis; it does not assert that analysis is linguistically correct. */
#include "upstream/morph.h"
#include "vocab.h"

#define MAX_BYTES 1048576u
#define HEADER_BYTES 64u
#define MAX_ENCODED (HEADER_BYTES + 2u * MAX_BYTES)
static u8 input[MAX_ENCODED + 1u];
static u8 output[MAX_ENCODED];
static MorphResult analyses;
static const u8 magic[8] = {'N','Y','M','O','R','P','H',0};

void *memcpy(void *dst, const void *src, u64 n) {
    u8 *d = dst; const u8 *s = src; while (n--) *d++ = *s++; return dst;
}
void *memset(void *dst, int value, u64 n) {
    u8 *d = dst; while (n--) *d++ = (u8)value; return dst;
}
static long call3(long nr, long a, long b, long c) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "a"(nr), "D"(a), "S"(b), "d"(c) : "rcx", "r11", "memory");
    return ret;
}
static int equal(const u8 *a, const u8 *b, u64 n) {
    u64 i; for (i=0; i<n; i++) if (a[i] != b[i]) return 0; return 1;
}
static int arg(const char *a, const char *b) {
    while (*a && *b && *a == *b) { a++; b++; } return *a == *b;
}
static int write_all(long fd, const u8 *p, u64 n) {
    while (n) {
        long rc = call3(1, fd, (long)p, (long)n);
        if (rc == -4) continue;
        if (rc <= 0) return 3;
        p += rc; n -= (u64)rc;
    }
    return 0;
}
static void put32(u8 *p, u32 x) { u32 i; for (i=0;i<4;i++) p[i]=(u8)(x>>(8*i)); }
static void put64(u8 *p, u64 x) { u32 i; for (i=0;i<8;i++) p[i]=(u8)(x>>(8*i)); }
static u32 get32(const u8 *p) { u32 i,x=0; for (i=0;i<4;i++) x|=(u32)p[i]<<(8*i); return x; }
static u64 get64(const u8 *p) { u32 i; u64 x=0; for (i=0;i<8;i++) x|=(u64)p[i]<<(8*i); return x; }
/* Strict decoder used only to locate boundaries. Invalid bytes use fallback. */
static u32 scalar(const u8 *p, u32 n, u32 *cp) {
    u32 c, k, i, v;
    if (!n) return 0;
    c=p[0];
    if (c<128) { *cp=c; return 1; }
    if (c>=0xc2 && c<=0xdf) { k=2; v=c&31; }
    else if (c>=0xe0 && c<=0xef) { k=3; v=c&15; }
    else if (c>=0xf0 && c<=0xf4) { k=4; v=c&7; }
    else return 0;
    if (k>n) return 0;
    for (i=1;i<k;i++) { if ((p[i]&192)!=128) return 0; v=(v<<6)|(p[i]&63); }
    if ((k==3 && v<0x800) || (k==4 && v<0x10000) || v>0x10ffff || (v>=0xd800 && v<=0xdfff)) return 0;
    *cp=v; return k;
}
static int separator(u32 c) {
    if (c<=0x20 || (c>=0x7f && c<=0xa0)) return 1;
    if (c<128) return !((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9'));
    return c==0x1680 || (c>=0x2000 && c<=0x206f) || c==0x3000 ||
           c==0x060c || c==0x061b || c==0x061e || c==0x061f || c==0x06d4;
}
static u32 piece_id(const u8 *p, u32 n) {
    u32 i;
    for (i=0;i<SURFACE_PIECE_COUNT;i++)
        if (n==piece_lengths[i] && equal(p,(const u8 *)pieces[i],n)) return 256+i;
    return 0xffffffffu;
}
static u32 cost(const u8 *p, u32 n) { return piece_id(p,n)==0xffffffffu ? n : 1; }
static void token(u32 *count, u32 id) {
    u32 pos=HEADER_BYTES+2*(*count); output[pos]=(u8)id; output[pos+1]=(u8)(id>>8); ++*count;
}
static void surface(const u8 *p, u32 n, u32 *count) {
    u32 i, id=piece_id(p,n);
    if (id!=0xffffffffu) { token(count,id); return; }
    for (i=0;i<n;i++) token(count,p[i]);
}
static void word(const u8 *p, u32 n, u32 *count) {
    u32 i,j,best=0,best_cost=cost(p,n),cursor,total;
    MorphStatus rc=morph_analyze(p,n,&analyses);
    if (rc!=MORPH_OK || !analyses.count) { surface(p,n,count); return; }
    for (i=1;i<analyses.count;i++) {
        const MorphAnalysis *a=&analyses.analyses[i]; cursor=0; total=0;
        for (j=0;j<a->count;j++) {
            const MorphPart *part=&a->parts[j];
            if (!part->len || part->offset!=cursor || part->len>n-cursor) break;
            total+=cost(p+cursor,part->len); cursor+=part->len;
        }
        if (j==a->count && cursor==n && total<best_cost) { best=i; best_cost=total; }
    }
    if (!best) { surface(p,n,count); return; }
    for (j=0;j<analyses.analyses[best].count;j++) {
        const MorphPart *part=&analyses.analyses[best].parts[j]; surface(p+part->offset,part->len,count);
    }
}
static int encode(u32 n) {
    u32 pos=0,start,k,cp,count=0;
    while (pos<n) {
        k=scalar(input+pos,n-pos,&cp);
        if (!k) { token(&count,input[pos++]); continue; }
        if (separator(cp)) { while (k--) token(&count,input[pos++]); continue; }
        start=pos; pos+=k;
        while (pos<n) {
            k=scalar(input+pos,n-pos,&cp); if (!k || separator(cp)) break; pos+=k;
        }
        word(input+start,pos-start,&count);
    }
    memcpy(output,magic,8); put32(output+8,1); put32(output+12,SURFACE_VOCAB_VERSION);
    put64(output+16,count); put64(output+24,n); memcpy(output+32,vocab_hash,32);
    return write_all(1,output,HEADER_BYTES+2u*count);
}
static int decode(u32 n) {
    u64 count,original,total=0,i; u32 id,j,part_len; const u8 *part;
    if (n<HEADER_BYTES || !equal(input,magic,8) || get32(input+8)!=1 ||
        get32(input+12)!=SURFACE_VOCAB_VERSION || !equal(input+32,vocab_hash,32)) return 2;
    count=get64(input+16); original=get64(input+24);
    if (count>MAX_BYTES || original>MAX_BYTES || HEADER_BYTES+2*count!=n) return 2;
    /* Validate the complete stream before emitting any bytes. */
    for (i=0;i<count;i++) {
        id=input[HEADER_BYTES+2*i]|((u32)input[HEADER_BYTES+2*i+1]<<8);
        if (id>=SURFACE_VOCAB_SIZE) return 2;
        part_len=id<256 ? 1 : piece_lengths[id-256];
        if (part_len>original-total) return 2;
        total+=part_len;
    }
    if (total!=original) return 2;
    total=0;
    for (i=0;i<count;i++) {
        id=input[HEADER_BYTES+2*i]|((u32)input[HEADER_BYTES+2*i+1]<<8);
        if (id<256) { output[total++]=(u8)id; continue; }
        part=(const u8 *)pieces[id-256]; part_len=piece_lengths[id-256];
        for (j=0;j<part_len;j++) output[total++]=part[j];
    }
    return write_all(1,output,total);
}
int tokenizer_main(u64 argc, const char **argv) {
    u32 n=0,limit; int mode; long rc;
    if (argc!=2) return 64;
    if (arg(argv[1],"vocab")) return write_all(1,(const u8 *)vocab_json,sizeof vocab_json-1);
    if (arg(argv[1],"encode")) { mode=1; limit=MAX_BYTES; }
    else if (arg(argv[1],"decode")) { mode=0; limit=MAX_ENCODED; }
    else return 64;
    while (n<=limit) {
        rc=call3(0,0,(long)(input+n),(long)(limit+1-n));
        if (rc==-4) continue;
        if (rc<0) return 3;
        if (!rc) return mode ? encode(n) : decode(n);
        n+=(u32)rc;
    }
    return 4;
}
