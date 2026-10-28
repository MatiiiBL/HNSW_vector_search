// Proves the asan and tsan presets really instrument the code. Each test runs a deliberate bug
// inside a death test (a child process) and expects the sanitizer to kill it. If the sanitizer
// flags ever stop being applied, these tests fail instead of every other test silently passing.

#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>

// The tsan preset sets its flags directly, so detect ThreadSanitizer the way each compiler
// reports it: GCC defines __SANITIZE_THREAD__, Clang answers __has_feature.
#ifdef __SANITIZE_THREAD__
#define VECDEX_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define VECDEX_TSAN 1
#endif
#endif

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

#elif defined(VECDEX_TSAN)

// Two threads write one int with nothing ordering the writes: a data race.
void race() {
    int shared = 0;
    std::thread writer([&shared] { shared = 1; });
    shared = 2;
    writer.join();
}

TEST(SanitizerSmoke, TsanCatchesDataRace) {
    // The child re-runs the test binary rather than continuing as a fork() of this process:
    // TSan won't start threads in a child forked from a multithreaded process.
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // TSan fails the process at the report (halt_on_error=1, as in the tsan test preset) or else
    // at exit: with status 66 on Linux, by abort() on macOS.
    EXPECT_DEATH(
        {
            race();
            std::exit(0);
        },
        "data race");
}

#else

TEST(SanitizerSmoke, NeedsSanitizerPreset) {
    GTEST_SKIP() << "sanitizer checks only run in the asan and tsan presets";
}

#endif

}  // namespace
