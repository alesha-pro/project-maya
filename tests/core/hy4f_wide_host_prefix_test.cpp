#include "../../src/hy4f/hy4f_host_prefix.hpp"
#include <cstdio>
#include <cstdlib>

static void check(bool condition) {
    if (!condition) {
        std::fputs("invalid large-chunk host prefix boundary\n", stderr);
        std::exit(1);
    }
}

int main() {
    using strata::hy4f::WidePrefixGrid;
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
