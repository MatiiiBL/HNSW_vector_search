// Proves the asan preset really instruments the code. Each test runs a deliberate bug inside a
// death test (a forked child process) and expects the sanitizer to kill it. If the sanitizer
// flags ever stop being applied, these tests fail instead of every other test silently passing.

#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <vector>

namespace {

#ifdef VECDEX_SANITIZE

int read_one_past_the_end() {
    std::vector<int> values(4, 1);
    volatile std::size_t index = values.size();  // volatile: the compiler can't see the bug
    return values.data()[index];
}

int overflow_int() {
    volatile int big = INT_MAX;
    return big + 1;  // signed overflow is undefined behavior
}

TEST(SanitizerSmoke, AsanCatchesHeapBufferOverflow) {
    EXPECT_DEATH(read_one_past_the_end(), "heap-buffer-overflow");
}

TEST(SanitizerSmoke, UbsanCatchesSignedOverflow) {
    EXPECT_DEATH(overflow_int(), "signed integer overflow");
}

#else

TEST(SanitizerSmoke, NeedsAsanPreset) {
    GTEST_SKIP() << "sanitizer checks only run in the asan preset";
}

#endif

}  // namespace
