#include <stdio.h>
#include <stdlib.h>

typedef unsigned long long ull;

static int cmp_ull(const void *a, const void *b) {
    ull x = *(const ull *)a;
    ull y = *(const ull *)b;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

int main(void) {
    ull n;
    if (scanf("%llu", &n) != 1) {
        return 0;
    }

    if (n < 4) {
        printf("%llu\n", n);
        return 0;
    }

    ull limit = 1;
    while ((limit + 1) <= n / (limit + 1)) {
        ++limit;
    }

    ull capacity = 1200000;
    ull *vals = (ull *)malloc(sizeof(ull) * capacity);
    if (vals == NULL) {
        return 0;
    }

    ull count = 0;
    for (ull a = 2; a <= limit; ++a) {
        ull p = a * a;
        while (p <= n) {
            vals[count++] = p;
            if (p > n / a) {
                break;
            }
            p *= a;
        }
    }

    qsort(vals, count, sizeof(ull), cmp_ull);

    ull perfect_powers = 0;
    for (ull i = 0; i < count; ++i) {
        if (i == 0 || vals[i] != vals[i - 1]) {
            ++perfect_powers;
        }
    }

    printf("%llu\n", n - perfect_powers);

    free(vals);
    return 0;
}
