// Choose the greedy verify width. Only the target's exact verification emits tokens.
#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <vector>

namespace strata::hy4f {

class DraftWidth {
    struct Bin {
        int seen = 0;
        int kept = 0;
    };
    std::array<std::array<Bin, 4>, 3> bins_{};
    std::array<int, 3> bin_{};
    std::array<double, 3> probabilities_{};
    std::vector<float> logits_;
    double prefix_ = 1.0;

public:
    explicit DraftWidth(int vocab, bool allow) {
        if (allow && std::getenv("HY4F_DYNAMIC_SPEC")) logits_.resize(vocab);
    }
    bool enabled() const { return !logits_.empty(); }
    float* logits() { return enabled() ? logits_.data() : nullptr; }
    void begin() { prefix_ = 1.0; }

    // Given a draft already computed, decide if adding its verify row is worthwhile.
    bool offer(int depth, int token) {
        if (!enabled()) return true;
        // The graph returns the full MTP row even in greedy mode. One max-reduction
        // either side of its argmax supplies a token-independent ambiguity measure.
        const float* row = logits_.data();
        float second = -std::numeric_limits<float>::infinity();
        if (token) second = *std::max_element(row, row + token);
        if ((size_t) token + 1 < logits_.size())
            second = std::max(second, *std::max_element(row + token + 1, row + logits_.size()));
        const float margin = row[token] - second;
        int b = (margin >= 0.5f) + (margin >= 1.5f) + (margin >= 3.0f);
        const Bin& obs = bins_[depth][b];
        constexpr double prior[3][4] = {{0.60, 0.75, 0.87, 0.96},
                                         {0.48, 0.67, 0.84, 0.94},
                                         {0.44, 0.63, 0.80, 0.92}};
        const double p = (obs.kept + 12.0 * prior[depth][b]) / (obs.seen + 12.0);
        bin_[depth] = b;
        probabilities_[depth] = p;
        prefix_ *= p;
        // Delta verify time 2.0 / 2.7 / 3.1 ms per row at ~145 tok/s.
        constexpr double verify_cost[3] = {0.29, 0.39, 0.45};
        return depth == 0 || prefix_ >= verify_cost[depth];
    }
    bool extend(int depth) const {
        if (!enabled()) return true;
        // The next draft costs a forward through the MTP layer as well as a
        // verify row. The current draft is known; the next one's margin is not.
        const double next_p = 0.55 + 0.32 * probabilities_[depth];
        return prefix_ * next_p > (depth == 0 ? 0.50 : 0.57);
    }
    void observe(int offered, int kept) {
        if (!enabled()) return;
        for (int i = 0; i < offered && i <= kept; ++i) {
            Bin& b = bins_[i][bin_[i]];
            ++b.seen;
            b.kept += i < kept;
        }
    }
};

}  // namespace strata::hy4f
