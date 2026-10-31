/* Integer oracle and timing harness for niyah_i8_gemm.
 * The reference dot is the current one-row AVX2 kernel (niyah_i8_dot).
 * A scalar int32 sum is the independent oracle on the small fixtures.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int32_t niyah_i8_dot(const int8_t *a, const int8_t *b, int k);
extern int niyah_i8_gemm(const int8_t *a, const int8_t *b, int32_t *c, int m, int n, int k);

static int32_t scalar_dot(const int8_t *a, const int8_t *b, int k) {
    int32_t s = 0;
    for (int i = 0; i < k; ++i) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

static void dot_rows(const int8_t *a, const int8_t *b, int32_t *c, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            c[(int64_t)i * n + j] = niyah_i8_dot(a + (int64_t)i * k, b + (int64_t)j * k, k);
}

static int expect_eq(const char *name, const int32_t *got, const int32_t *want, int count) {
    for (int i = 0; i < count; ++i) {
        if (got[i] != want[i]) {
            fprintf(stderr, "FAIL %s at %d got %d want %d\n", name, i, got[i], want[i]);
            return 1;
        }
    }
    return 0;
}

static int check_small(const char *name, const int8_t *a, const int8_t *b, int m, int n, int k) {
    int32_t *g = calloc((size_t)m * (size_t)n, sizeof(int32_t));
    int32_t *d = calloc((size_t)m * (size_t)n, sizeof(int32_t));
    int32_t *s = calloc((size_t)m * (size_t)n, sizeof(int32_t));
    if (!g || !d || !s) return 1;
    int rc = niyah_i8_gemm(a, b, g, m, n, k);
    if (rc != 0) {
        fprintf(stderr, "FAIL %s gemm rc %d\n", name, rc);
        free(g); free(d); free(s);
        return 1;
    }
    dot_rows(a, b, d, m, n, k);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            s[(int64_t)i * n + j] = scalar_dot(a + (int64_t)i * k, b + (int64_t)j * k, k);
    int bad = expect_eq(name, g, d, m * n) | expect_eq(name, g, s, m * n);
    /* Same FP32 scale epilogue as qwen_native_layer0.c linear(). */
    double max_abs = 0.0;
    for (int i = 0; i < m; ++i) {
        float act = (float)(0.01 * (i + 1));
        for (int j = 0; j < n; ++j) {
            float ws = (float)(0.02 * ((j % 7) + 1));
            float fg = (float)((double)g[(int64_t)i * n + j] * (double)act * (double)ws);
            float fd = (float)((double)d[(int64_t)i * n + j] * (double)act * (double)ws);
            if (memcmp(&fg, &fd, sizeof(float)) != 0) {
                fprintf(stderr, "FAIL %s float bits i=%d j=%d\n", name, i, j);
                bad = 1;
            }
            double diff = fabs((double)fg - (double)fd);
            if (diff > max_abs) max_abs = diff;
        }
    }
    if (max_abs != 0.0) bad = 1;
    free(g); free(d); free(s);
    return bad;
}

static uint64_t lcg = 20261029u;
static int8_t rnd_i8(void) {
    lcg = lcg * 6364136223846793005ull + 1ull;
    return (int8_t)(lcg >> 33);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_double(const void *p, const void *q) {
    double a = *(const double *)p, b = *(const double *)q;
    return (a > b) - (a < b);
}

static double median5(double *v) {
    qsort(v, 5, sizeof(double), cmp_double);
    return v[2];
}

static double saved_dot[4], saved_gemm[4];
static int saved_n;

static int time_shape(const char *name, int m, int n, int k) {
    int8_t *a = malloc((size_t)m * (size_t)k);
    int8_t *b = malloc((size_t)n * (size_t)k);
    int32_t *g = malloc((size_t)m * (size_t)n * sizeof(int32_t));
    int32_t *d = malloc((size_t)m * (size_t)n * sizeof(int32_t));
    if (!a || !b || !g || !d) return 1;
    for (size_t i = 0; i < (size_t)m * (size_t)k; ++i) a[i] = rnd_i8();
    for (size_t i = 0; i < (size_t)n * (size_t)k; ++i) b[i] = rnd_i8();
    if (niyah_i8_gemm(a, b, g, m, n, k) != 0) {
        fprintf(stderr, "FAIL gemm %s\n", name);
        return 1;
    }
    dot_rows(a, b, d, m, n, k);
    if (expect_eq(name, g, d, m * n)) return 1;
    /* One warmup each, then five alternating samples. */
    niyah_i8_gemm(a, b, g, m, n, k);
    dot_rows(a, b, d, m, n, k);
    double gs[5], ds[5];
    int gi = 0, di = 0;
    for (int i = 0; i < 10; ++i) {
        double t0 = now_ms();
        if (i & 1) {
            dot_rows(a, b, d, m, n, k);
            ds[di++] = now_ms() - t0;
        } else {
            niyah_i8_gemm(a, b, g, m, n, k);
            gs[gi++] = now_ms() - t0;
        }
    }
    double gm = median5(gs), dm = median5(ds);
    double macs = (double)m * (double)n * (double)k;
    if (saved_n < 4) { saved_dot[saved_n] = dm; saved_gemm[saved_n] = gm; saved_n++; }
    printf("  {\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,"
           "\"integer_match_vs_niyah_i8_dot\":\"PASS\","
           "\"dot_rows_median_ms\":%.6f,\"gemm_median_ms\":%.6f,"
           "\"dot_over_gemm\":%.4f,\"gemm_gmac_per_s\":%.4f}\n",
           name, m, n, k, dm, gm, dm / gm, macs / (gm * 1e6));
    free(a); free(b); free(g); free(d);
    return 0;
}

static int check_reject(void) {
    int8_t a[16] = {0}, b[16] = {0};
    int32_t c[4] = {0};
    if (niyah_i8_gemm(a, b, c, 0, 1, 16) != 64) return 1;
    if (niyah_i8_gemm(a, b, c, 1, 0, 16) != 64) return 1;
    if (niyah_i8_gemm(a, b, c, 1, 1, 15) != 64) return 1;
    if (niyah_i8_gemm(a, b, c, 1, 1, 16384 + 16) != 64) return 1;
    if (niyah_i8_gemm(a, b, c, 100000, 100000, 16) != 64) return 1;
    return 0;
}

static int load_raw(const char *path, int8_t **out, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    int8_t *p = malloc(bytes);
    if (!p) { fclose(f); return 1; }
    if (fread(p, 1, bytes, f) != bytes) { free(p); fclose(f); return 1; }
    fclose(f);
    *out = p;
    return 0;
}

static int check_files(const char *a_path, const char *b_path, int m, int n, int k) {
    int8_t *a = NULL, *b = NULL;
    if (load_raw(a_path, &a, (size_t)m * (size_t)k) || load_raw(b_path, &b, (size_t)n * (size_t)k)) {
        fprintf(stderr, "FAIL read raw fixtures\n");
        free(a); free(b);
        return 1;
    }
    int bad = check_small("raw-fixture", a, b, m, n, k);
    free(a); free(b);
    if (!bad) printf("{\"raw_fixture\":\"PASS\",\"M\":%d,\"N\":%d,\"K\":%d}\n", m, n, k);
    return bad;
}

int main(int argc, char **argv) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "FAIL affinity %s\n", strerror(errno));
        return 1;
    }
    if (argc == 6) return check_files(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    if (check_reject()) {
        fprintf(stderr, "FAIL geometry reject\n");
        return 1;
    }
    int8_t one_a[16], one_b[16];
    for (int i = 0; i < 16; ++i) { one_a[i] = (int8_t)(i - 8); one_b[i] = (int8_t)(3 - i); }
    if (check_small("k16", one_a, one_b, 1, 1, 16)) return 1;
    int8_t ext_a[4 * 32], ext_b[6 * 32];
    memset(ext_a, 0x80, sizeof(ext_a));
    memset(ext_b, 0x80, sizeof(ext_b));
    if (check_small("neg128", ext_a, ext_b, 4, 6, 32)) return 1;
    memset(ext_a, 127, sizeof(ext_a));
    memset(ext_b, -128, sizeof(ext_b));
    if (check_small("127x-128", ext_a, ext_b, 4, 6, 32)) return 1;
    int8_t rnd_a[9 * 80], rnd_b[6 * 80];
    for (size_t i = 0; i < sizeof(rnd_a); ++i) rnd_a[i] = rnd_i8();
    for (size_t i = 0; i < sizeof(rnd_b); ++i) rnd_b[i] = rnd_i8();
    if (check_small("m9n6k80", rnd_a, rnd_b, 9, 6, 80)) return 1;
    if (check_small("m7-prefix", rnd_a, rnd_b, 7, 5, 48)) return 1;
    if (check_small("m3", rnd_a, rnd_b, 3, 5, 32)) return 1;
    if (check_small("m8", rnd_a, rnd_b, 8, 3, 64)) return 1;
    if (check_small("m16k12288-narrow", rnd_a, rnd_b, 2, 2, 16)) return 1;
    /* K=12288 narrow random, scalar oracle included. */
    int8_t *long_a = malloc(5ull * 12288);
    int8_t *long_b = malloc(3ull * 12288);
    if (!long_a || !long_b) return 1;
    for (size_t i = 0; i < 5ull * 12288; ++i) long_a[i] = rnd_i8();
    for (size_t i = 0; i < 3ull * 12288; ++i) long_b[i] = rnd_i8();
    if (check_small("m5n3k12288", long_a, long_b, 5, 3, 12288)) return 1;
    free(long_a); free(long_b);
    printf("{\"fixtures\":\"PASS\",\"scalar_oracle\":\"PASS\",\"dot_oracle\":\"PASS\",\"float_epilogue_max_abs\":0}\n");
    printf("{\"timings\":[\n");
    if (time_shape("kv", 16, 1024, 4096)) return 1;
    printf(",\n");
    if (time_shape("q", 16, 4096, 4096)) return 1;
    printf(",\n");
    if (time_shape("gate", 16, 12288, 4096)) return 1;
    printf(",\n");
    if (time_shape("down", 16, 4096, 12288)) return 1;
    printf("\n]}\n");
    /* Call order is kv, q, gate, down. K and V match kv; O matches q; up matches gate. */
    double dot_sum = 2 * saved_dot[1] + 2 * saved_dot[0] + 2 * saved_dot[2] + saved_dot[3];
    double gemm_sum = 2 * saved_gemm[1] + 2 * saved_gemm[0] + 2 * saved_gemm[2] + saved_gemm[3];
    printf("{\"seven_linear_isolated_sum_ms\":{\"dot_rows\":%.4f,\"gemm\":%.4f,"
           "\"note\":\"sum of isolated medians for q+o+k+v+gate+up+down; not a fused layer-0 run and not RMSNorm, RoPE, attention, SiLU, or residuals\"}}\n",
           dot_sum, gemm_sum);
    return 0;
}
