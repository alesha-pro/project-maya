// Request-local prompt/generated-token lookup for greedy draft suffixes.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace strata::hy4f {
class PromptLookup {
    int min_match_ = 0;
    std::vector<int32_t> tokens_;
    std::vector<int> previous_;
    std::unordered_map<uint64_t, int> latest_;
    // Three 17-bit vocabulary IDs. Equality is checked against the token array.
    static uint64_t key(int a, int b, int c) {
        return (uint64_t(a) << 34) | (uint64_t(b) << 17) | uint64_t(c);
    }
public:
    explicit PromptLookup(const std::vector<int32_t>& prompt, bool allow) {
        const char* opt = std::getenv("HY4F_LOOKUP");
        if (!allow || !opt || std::atoi(opt) < 3) return;
        min_match_ = std::atoi(opt);
        latest_.reserve(prompt.size());
        for (auto id : prompt) append(id);
    }
    void append(int32_t id) {
        if (!min_match_) return;
        const int i = (int) tokens_.size();
        tokens_.push_back(id);
        previous_.push_back(-1);
        if (i < 2) return;
        const auto k = key(tokens_[i-2], tokens_[i-1], id);
        auto it = latest_.emplace(k, i);
        if (!it.second) { previous_[i] = it.first->second; it.first->second = i; }
    }
    // The first draft is the neural prediction. Match the virtual suffix ending
    // there and offer only continuation tokens already present in this request.
    int extend(int first, int limit, int* output) const {
        const int n = (int) tokens_.size();
        if (!min_match_ || n < 2 || limit < 1) return 0;
        const auto it = latest_.find(key(tokens_[n-2], tokens_[n-1], first));
        if (it == latest_.end()) return 0;
        int best = -1, best_len = min_match_ - 1;
        for (int j = it->second, tries = 0; j >= 2 && tries < 6; j = previous_[j], ++tries) {
            if (j + 1 >= n) continue;
            int length = 3;
            while (length <= j && length <= n && length < 24 &&
                   tokens_[j - length] == tokens_[n - length]) ++length;
            if (length > best_len) { best = j; best_len = length; }
        }
        if (best < 0) return 0;
        const int count = std::min(limit, n - best - 1);
        std::copy_n(tokens_.begin() + best + 1, count, output);
        return count;
    }
};
}  // namespace strata::hy4f
