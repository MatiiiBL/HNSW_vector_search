#pragma once

// Internal: the scratch memory of graph searches, pooled so searches don't allocate.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "vecdex/neighbor.hpp"

namespace vecdex::detail {

// Marks the nodes one search has visited. Clearing an n-element array before every search
// would cost O(n) per query (4 MB for a million nodes). Instead each search bumps an epoch,
// and a node counts as visited only if its mark equals the current epoch.
class VisitedList {
public:
    // Starts a new search over ids 0 .. n-1. O(1), except when the list grows or, once every
    // 2^32 searches, when the epoch wraps around and old marks must be cleared.
    void reset(std::size_t n) {
        if (marks_.size() < n) {
            marks_.resize(n, 0);  // new entries are 0, which is never a current epoch
        }
        if (++epoch_ == 0) {
            std::ranges::fill(marks_, 0U);
            epoch_ = 1;
        }
    }

    // Marks `id` visited. True if this is the first visit in the current search.
    bool insert(Id id) noexcept {
        if (marks_[id] == epoch_) {
            return false;
        }
        marks_[id] = epoch_;
        return true;
    }

private:
    std::vector<std::uint32_t> marks_;
    std::uint32_t epoch_ = 0;
};

// Everything one layer search needs besides the graph. The vectors keep their capacity from one
// search to the next, so once they have grown to fit, a search allocates nothing.
struct SearchScratch {
    VisitedList visited;
    std::vector<Neighbor> candidates;  // min-heap of nodes still to expand
    std::vector<Neighbor> results;     // max-heap of the best ef found so far
    std::vector<Id> fresh;             // unvisited neighbors of the node being expanded
    std::vector<Id> link_copy;         // a node's links, copied under its lock (parallel add)
};

// Lends SearchScratch to searches, so concurrent searches never share one and each keeps its
// memory between searches. The mutex is held only to hand one out or take it back.
class ScratchPool {
public:
    // Returns its scratch to the pool when destroyed (RAII), even if the search throws.
    class Lease {
    public:
        Lease(ScratchPool& pool, std::unique_ptr<SearchScratch> scratch) noexcept
            : pool_(&pool), scratch_(std::move(scratch)) {}
        ~Lease() { pool_->give_back(std::move(scratch_)); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&&) = delete;
        Lease& operator=(Lease&&) = delete;

        SearchScratch& operator*() const noexcept { return *scratch_; }
        SearchScratch* operator->() const noexcept { return scratch_.get(); }

    private:
        ScratchPool* pool_;
        std::unique_ptr<SearchScratch> scratch_;
    };

    Lease acquire() {
        const std::scoped_lock lock(mutex_);
        if (free_.empty()) {
            // Room for every scratch in existence, so give_back's push_back never allocates
            // (and so never throws inside a destructor).
            free_.reserve(++created_);
            return Lease(*this, std::make_unique<SearchScratch>());
        }
        std::unique_ptr<SearchScratch> scratch = std::move(free_.back());
        free_.pop_back();
        return Lease(*this, std::move(scratch));
    }

private:
    void give_back(std::unique_ptr<SearchScratch> scratch) noexcept {
        const std::scoped_lock lock(mutex_);
        assert(free_.size() < free_.capacity());  // reserved in acquire(): no allocation here
        free_.push_back(std::move(scratch));
    }

    std::mutex mutex_;
    std::vector<std::unique_ptr<SearchScratch>> free_;
    std::size_t created_ = 0;
};

}  // namespace vecdex::detail
