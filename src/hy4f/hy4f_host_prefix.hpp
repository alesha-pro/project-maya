// Text prefix keys for the bounded host KV cache used by the single-request serve loop.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::hy4f {
class HostPrefixIndex {
public:
    using Key = std::vector<int32_t>;
    static constexpr size_t block = 2048;
    static constexpr size_t limit = 32768;

    static constexpr unsigned fine_slots = 2;
    explicit HostPrefixIndex(unsigned slots, bool deep = false, bool fine = false)
        : short_slots_(slots), anchor_slots_(deep ? 2 * slots : slots),
          entries_(anchor_slots_ + (deep && fine && slots ? fine_slots : 0)) {}
    bool enabled() const { return !entries_.empty(); }
    static size_t checkpoint(size_t prompt_size) {
        if (prompt_size <= 2 * block) return 0;
        return std::min(limit, prompt_size - 1) / block * block;
    }
    // Alternate long sessions without evicting the 32K anchors used by short branches.
    size_t deep_checkpoint(size_t prompt_size) const {
        if (anchor_slots_ == short_slots_ || prompt_size <= limit) return 0;
        const size_t stride = 4 * block;
        const size_t pos = (prompt_size - 1) / stride * stride;
        return pos > limit ? pos : 0;
    }
    size_t fine_checkpoint(size_t prompt_size) const {
        if (entries_.size() == anchor_slots_ || prompt_size <= limit) return 0;
        const size_t pos = (prompt_size - 1) / block * block;
        return pos > limit && pos % (4 * block) ? pos : 0;
    }
    size_t max_delta(size_t ctx) const {
        if (entries_.size() == anchor_slots_ || ctx <= limit + block) return 0;
        return std::min(3 * block, (ctx - 1) / block * block - limit);
    }
    int fine_parent(const Key& key) const {
        const size_t boundary = key.size() / (4 * block) * (4 * block);
        for (size_t i = 0; i < anchor_slots_; ++i) {
            const auto& parent = entries_[i].tokens;
            if (parent.size() == boundary && boundary >= limit &&
                std::equal(parent.begin(), parent.end(), key.begin())) return int(i);
        }
        return -1;
    }
    // Replay the last token for logits. Avoid paying PCIe transfer for a small gain.
    int lookup(const Key& request, size_t retained) {
        int best = -1;
        for (size_t i = 0; i < entries_.size(); ++i) {
            const Key& key = entries_[i].tokens;
            if (key.empty() || key.size() >= request.size() || key.size() < retained + 256) continue;
            if (best >= 0 && key.size() <= entries_[best].tokens.size()) continue;
            if (std::equal(key.begin(), key.end(), request.begin())) best = static_cast<int>(i);
        }
        if (best >= 0) {
            entries_[best].age = ++tick_;
            if (entries_[best].parent >= 0) entries_[entries_[best].parent].age = tick_;
        }
        return best;
    }
    // Refresh an identical hit, extend an existing branch, otherwise use the least recently used slot.
    int destination(const Key& key, int parent = -1) {
        int victim = -1, ancestor = -1;
        const bool deep = key.size() > limit;
        const size_t first = parent >= 0 ? anchor_slots_ : deep ? short_slots_ : 0;
        const size_t end = parent >= 0 ? entries_.size() : deep ? anchor_slots_ : short_slots_;
        for (size_t i = first; i < end; ++i) {
            auto& entry = entries_[i];
            if (entry.tokens == key && entry.parent == parent) { entry.age = ++tick_; return -1; }
            if (!entry.tokens.empty() && entry.tokens.size() < key.size() &&
                std::equal(entry.tokens.begin(), entry.tokens.end(), key.begin()) &&
                (ancestor < 0 || entry.tokens.size() > entries_[ancestor].tokens.size()))
                ancestor = static_cast<int>(i);
            if (victim < 0 || entry.age < entries_[victim].age) victim = static_cast<int>(i);
        }
        return ancestor >= 0 ? ancestor : victim;
    }
    void clear(int slot) {
        for (size_t i = anchor_slots_; i < entries_.size(); ++i)
            if (entries_[i].parent == slot) entries_[i] = {};
        entries_[slot] = {};
    }
    void publish(int slot, const Key& key, int parent = -1) {
        if (!entries_[slot].tokens.empty() && entries_[slot].tokens != key) clear(slot);
        entries_[slot].parent = parent;
        entries_[slot].tokens = key;
        entries_[slot].age = ++tick_;
    }
    size_t length(int slot) const { return entries_[slot].tokens.size(); }
private:
    struct Entry { Key tokens; uint64_t age = 0; int parent = -1; };
    size_t short_slots_, anchor_slots_;
    std::vector<Entry> entries_;
    uint64_t tick_ = 0;
};
} // namespace strata::hy4f
