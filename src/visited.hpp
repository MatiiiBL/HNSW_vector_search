#pragma once

// Internal: the "visited" bookkeeping for graph searches.

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

// Lends VisitedLists to searches, so concurrent searches never share one and each list keeps
// its memory between searches. The mutex is held only to hand a list out or take it back.
class VisitedPool {
public:
    // Returns its list to the pool when destroyed (RAII), even if the search throws.
    class Lease {
    public:
        Lease(VisitedPool& pool, std::unique_ptr<VisitedList> list) noexcept
            : pool_(&pool), list_(std::move(list)) {}
        ~Lease() { pool_->give_back(std::move(list_)); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&&) = delete;
        Lease& operator=(Lease&&) = delete;

        VisitedList& operator*() const noexcept { return *list_; }

    private:
        VisitedPool* pool_;
        std::unique_ptr<VisitedList> list_;
    };

    Lease acquire() {
        const std::scoped_lock lock(mutex_);
        if (free_.empty()) {
            // Room for every list in existence, so give_back's push_back never allocates
            // (and so never throws inside a destructor).
            free_.reserve(++created_);
            return Lease(*this, std::make_unique<VisitedList>());
        }
        std::unique_ptr<VisitedList> list = std::move(free_.back());
        free_.pop_back();
        return Lease(*this, std::move(list));
    }

private:
    void give_back(std::unique_ptr<VisitedList> list) noexcept {
        const std::scoped_lock lock(mutex_);
        assert(free_.size() < free_.capacity());  // reserved in acquire(): no allocation here
        free_.push_back(std::move(list));
    }

    std::mutex mutex_;
    std::vector<std::unique_ptr<VisitedList>> free_;
    std::size_t created_ = 0;
};

}  // namespace vecdex::detail
