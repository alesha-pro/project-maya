#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::hy4f {

// Benchmark-only probe. At rejection i, target row i+1 predicts the position
// just beyond the correction on the old prefix. Score those free predictions
// against the actual emitted continuation and the next regular draft.
class JacobiProbe {
    struct Step {
        int offset, width, accepted;
        std::array<int32_t, 4> target{}, proposal{};
    };
    struct Group {
        int count = 0, jacobi_hit = 0, reference_count = 0, reference_hit = 0;
        int paired_jacobi_hit = 0, both = 0, agree_wrong = 0;
    };
    const char* path_;
    std::vector<int32_t> emitted_;
    std::vector<Step> steps_;
public:
    explicit JacobiProbe(bool greedy) : path_(greedy ? std::getenv("HY4F_JACOBI_STATS") : nullptr) {}
    bool enabled() const { return path_ && *path_; }
    void emitted(int32_t id) { if (enabled()) emitted_.push_back(id); }
    void step(int offset, int width, int accepted, const int32_t* target, const int32_t* proposal) {
        if (!enabled()) return;
        Step s{offset, width, accepted};
        for (int i = 0; i <= width; ++i) s.target[i] = target[i];
        for (int i = 0; i < width; ++i) s.proposal[i] = proposal[i];
        steps_.push_back(s);
    }
    void report(unsigned long long request, size_t prompt) const {
        if (!enabled()) return;
        Group groups[4][4][2]{};
        for (size_t i = 0; i + 1 < steps_.size(); ++i) {
            const Step& old = steps_[i];
            if (old.accepted >= old.width) continue;
            const Step& next = steps_[i+1];
            if (next.offset != old.offset + old.accepted + 1) continue;
            bool prefix = true;
            for (int d = 1; d <= old.width - old.accepted; ++d) {
                const size_t position = size_t(next.offset) + d;
                if (position >= emitted_.size()) break;
                const int32_t pred = old.target[old.accepted + d];
                const int32_t truth = emitted_[position];
                const bool hit = pred == truth;
                Group& c = groups[old.accepted][d][int(prefix)];
                ++c.count;
                c.jacobi_hit += hit;
                if (d <= next.width) {
                    const int32_t ref = next.proposal[d-1];
                    const bool ref_hit = ref == truth;
                    ++c.reference_count;
                    c.reference_hit += ref_hit;
                    c.paired_jacobi_hit += hit;
                    c.both += hit && ref_hit;
                    c.agree_wrong += !hit && pred == ref;
                }
                prefix &= hit;
            }
        }
        std::FILE* out = std::fopen(path_, "a");
        if (!out) { std::perror("HY4F_JACOBI_STATS"); return; }
        std::fprintf(out,"request,%llu,%zu,%zu,%zu\n",request,prompt,emitted_.size(),steps_.size());
        for (int a = 0; a < 3; ++a) for (int d = 1; d <= 3; ++d) for (int p = 0; p < 2; ++p) {
            const Group& c = groups[a][d][p];
            if (c.count) std::fprintf(out,"hits,%llu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",request,a,d,p,
                c.count,c.jacobi_hit,c.reference_count,c.reference_hit,c.paired_jacobi_hit,c.both,c.agree_wrong);
        }
        std::fclose(out);
    }
};

}  // namespace strata::hy4f
