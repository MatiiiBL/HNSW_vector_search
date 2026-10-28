#include "file_sync.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <filesystem>
#include <system_error>

// Standard C++ can't flush to stable storage, and doing it properly differs between the systems
// vecdex supports. A port must decide how for its system, not silently lose durability.
#if !defined(__APPLE__) && !defined(__linux__)
#error "file_sync.cpp knows how to flush to stable storage only on macOS and Linux"
#endif

namespace vecdex::detail {
namespace {

// Runs a POSIX call again while a signal interrupts it (EINTR) before it has done anything.
template <typename Call>
int retry_on_eintr(Call call) noexcept {
    int result = 0;
    do {
        result = call();
    } while (result == -1 && errno == EINTR);
    return result;
}

// Owns a file descriptor and closes it.
class Descriptor {
public:
    explicit Descriptor(int fd) noexcept : fd_(fd) {}
    // Its only use was the flush, which reported any error, so close() has nothing to add.
    ~Descriptor() { (void)::close(fd_); }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    Descriptor(Descriptor&&) = delete;
    Descriptor& operator=(Descriptor&&) = delete;

    int get() const noexcept { return fd_; }

private:
    int fd_;
};

// Flushes what was written to the descriptor's file down to the storage medium.
int flush_to_storage(int fd) noexcept {
#ifdef __APPLE__
    // On macOS, fsync() only hands the data to the drive, which may keep it in a volatile cache
    // (see fsync(2)). F_FULLFSYNC also has the drive write its cache out. Some file systems, SMB
    // and WebDAV among them, don't implement it and fail with ENOTSUP or (undocumented) ENOTTY:
    // there fsync() is the best on offer. Any other error is a real failure.
    const int result = retry_on_eintr([fd] { return ::fcntl(fd, F_FULLFSYNC); });
    if (result != -1 || (errno != ENOTSUP && errno != ENOTTY)) {
        return result;
    }
#endif
    // On Linux, fsync() flushes the drive's cache as well. Not fdatasync(): the saved file is new,
    // so its size and block allocation must be flushed either way, and it would skip only the
    // timestamps.
    return retry_on_eintr([fd] { return ::fsync(fd); });
}

std::error_code sync_path(const std::filesystem::path& path, int flags) noexcept {
    // O_CLOEXEC: a process forked meanwhile by another thread doesn't inherit the descriptor.
    const int fd = retry_on_eintr([&] { return ::open(path.c_str(), flags | O_CLOEXEC); });
    if (fd == -1) {
        return {errno, std::generic_category()};
    }
    const Descriptor descriptor(fd);
    if (flush_to_storage(descriptor.get()) == -1) {
        return {errno, std::generic_category()};
    }
    return {};
}

}  // namespace

std::error_code sync_file(const std::filesystem::path& file) noexcept {
    // Read-only is enough: the flush covers the file's data, whichever descriptor wrote it.
    return sync_path(file, O_RDONLY);
}

std::error_code sync_directory(const std::filesystem::path& dir) noexcept {
    return sync_path(dir, O_RDONLY | O_DIRECTORY);
}

}  // namespace vecdex::detail
