// libFuzzer target for HnswIndex::load. Each input is a whole index file. Its checksum is
// recomputed first, so that mutations reach the validation instead of all failing the checksum.
// Every file must then either throw std::runtime_error or give an index that inspect() finds
// sound and that can be searched and grown. Anything else (a crash, a sanitizer report, another
// exception, an unsound graph) is a finding.
//
// Apple clang has no libFuzzer, so build with Homebrew's clang in a build tree of its own:
//   cmake -S . -B build/fuzz -G Ninja -DCMAKE_BUILD_TYPE=Debug \
//         -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
//         -DVECDEX_SANITIZE=ON -DVECDEX_BUILD_FUZZERS=ON
//   cmake --build build/fuzz --target vecdex_load_fuzzer
//   mkdir -p build/fuzz/corpus
//   build/fuzz/tests/vecdex_load_fuzzer build/fuzz/corpus -max_total_time=300
// An empty corpus directory is seeded with small valid indexes, one per metric.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "vecdex/hnsw.hpp"
#include "xxhash64.hpp"

namespace {

namespace fs = std::filesystem;

constexpr std::size_t kChecksumBytes = 8;

// The file each input is written to: one per fuzzer process, so parallel jobs (-jobs=N) don't
// share it, and removed when the process exits normally.
class ScratchFile {
public:
    ScratchFile()
        : path_(fs::temp_directory_path() /
                ("vecdex_load_fuzzer_" + std::to_string(std::random_device{}()) + ".vdx")) {}
    ~ScratchFile() {
        std::error_code ignored;
        fs::remove(path_, ignored);
    }
    ScratchFile(const ScratchFile&) = delete;
    ScratchFile& operator=(const ScratchFile&) = delete;
    ScratchFile(ScratchFile&&) = delete;
    ScratchFile& operator=(ScratchFile&&) = delete;

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

const fs::path& scratch_file() {
    static const ScratchFile file;
    return file.path();
}

void write_file(const fs::path& path, std::span<const std::byte> bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        std::abort();  // an environment problem, not a finding
    }
}

// Small valid indexes, so the fuzzer starts from files that get past the header.
void write_seeds(const fs::path& dir) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    int count = 0;
    for (const vecdex::Metric metric :
         {vecdex::Metric::L2, vecdex::Metric::InnerProduct, vecdex::Metric::Cosine}) {
        vecdex::HnswIndex index(3, metric, {.M = 2, .ef_construction = 8, .seed = 42});
        vecdex::Matrix<float> vectors(16, 3);
        for (float& v : vectors.data()) {
            v = uniform(rng);
        }
        index.add(vectors);
        index.save(dir / ("seed_" + std::to_string(count++) + ".vdx"));
    }
    vecdex::HnswIndex(2).save(dir / "seed_empty.vdx");
}

void check_usable(vecdex::HnswIndex& index) {
    if (!index.inspect().violations.empty()) {
        std::abort();  // load() promises inspect()'s invariants
    }
    if (index.dim() > 4096) {  // a valid empty index can have any dimension
        return;
    }
    const std::vector<float> query(index.dim(), 0.5F);
    const std::size_t size = index.size();
    (void)index.search(query, 5, 16);
    // add() allocates 1 + 2M ids for the new node. A valid empty index can carry any M the
    // constructor accepts, so that allocation is as large as the parameters ask for: a property
    // of the parameters, which HnswIndex(dim, metric, params) shares, not of load().
    if (index.params().M > 1024) {
        return;
    }
    if (index.add(query) != size || !index.inspect().violations.empty()) {
        std::abort();
    }
}

}  // namespace

// Seeds the first corpus directory on the command line if it is empty.
extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
    for (int i = 1; i < *argc; ++i) {
        const std::string_view arg = (*argv)[i];
        std::error_code ec;
        if (!arg.starts_with('-') && fs::is_directory(arg, ec)) {
            if (fs::is_empty(arg, ec)) {
                write_seeds(fs::path(arg));
            }
            break;
        }
    }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    std::vector<std::byte> file(size);
    if (size > 0) {
        std::memcpy(file.data(), data, size);
    }
    if (size >= kChecksumBytes) {
        const std::span<const std::byte> covered(file.data(), size - kChecksumBytes);
        const std::uint64_t checksum = vecdex::detail::xxh64(covered);
        std::memcpy(file.data() + covered.size(), &checksum, sizeof checksum);
    }
    write_file(scratch_file(), file);
    try {
        vecdex::HnswIndex index = vecdex::HnswIndex::load(scratch_file());
        check_usable(index);
    } catch (const std::runtime_error&) {
        // rejected: the expected outcome for most inputs
    }
    return 0;
}
