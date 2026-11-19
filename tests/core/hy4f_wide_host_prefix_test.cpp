#include "../../src/hy4f/hy4f_host_prefix.hpp"
#include <cstdio>
#include <cstdlib>

static void check(bool condition) {
    if (!condition) {
        std::fputs("invalid large-chunk host prefix boundary\n", stderr);
        std::exit(1);
    }
}

// every checkpoint of a grain is a chunk boundary below the prompt's end; one block keeps the original positions
static void grains() {
    using strata::hy4f::HostPrefixIndex;
    for (size_t grain : {2048u, 4096u}) {
        check(HostPrefixIndex::grain_supported(grain));
        const HostPrefixIndex keys(2, true, true, grain);
        check(keys.max_delta(262144) == 8192 - grain);
        for (size_t n = 0; n <= 262144; ++n) {
            const size_t a = keys.shallow_checkpoint(n), b = keys.deep_checkpoint(n), c = keys.fine_checkpoint(n);
            check(a % grain == 0 && b % grain == 0 && c % grain == 0);
            check((!a || a < n) && (!b || b < n) && (!c || c < n));
            check(a <= 32768 && (!b || b > 32768) && (!c || (c > 32768 && c - c / 8192 * 8192 <= keys.max_delta(262144))));
            if (grain == 2048) check(a == HostPrefixIndex::checkpoint(n));
            if (n > 4096) check(n - std::max({a, b, c}) <= (n > 32768 ? grain : std::max<size_t>(grain, n - 32768)));
        }
    }
    check(!HostPrefixIndex::grain_supported(1024) && !HostPrefixIndex::grain_supported(57600));
}

int main() {
    using strata::hy4f::WidePrefixGrid;
    grains();
    const WidePrefixGrid profile(57600);
    check(profile.shallow(4096) == 0 && profile.shallow(4097) == 4096);
    check(profile.shallow(32768) == 4096 && profile.shallow(32769) == 32768);
    check(profile.deep(90112) == 0 && profile.deep(90113) == 90112);
    check(profile.deep(262144) == 204800);
    for (size_t chunk : {32768u, 32769u, 57600u, 65536u}) {
        const WidePrefixGrid grid(chunk);
        for (size_t pos = 0; pos <= 262144; ++pos) {
            const size_t lo = grid.floor(pos), hi = grid.next(pos);
            check(lo <= pos && pos < hi && hi - pos <= chunk);
            check(lo % 2048 == 0 && hi % 2048 == 0);
            check(grid.next(lo) == hi && grid.floor(hi) == hi);
            if (pos) check(grid.shallow(pos) < pos && grid.deep(pos) < pos);
        }
    }
}
