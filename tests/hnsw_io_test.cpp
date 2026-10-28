// HnswIndex::save and HnswIndex::load. The corruption tests find fields through the offsets that
// docs/FORMAT.md documents, not through the implementation, so they also check that the code
// writes what the spec says.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "test_helpers.hpp"
#include "vecdex/hnsw.hpp"
#include "xxhash64.hpp"

namespace {

namespace fs = std::filesystem;
using vecdex::HnswIndex;
using vecdex::HnswParams;
using vecdex::Id;
using vecdex::Matrix;
using vecdex::Metric;
using Bytes = std::vector<std::byte>;

// Header fields (FORMAT.md, "Header"). Numbers are little-endian, like every host we test on.
constexpr std::size_t kVersionAt = 8;
constexpr std::size_t kMetricAt = 12;
constexpr std::size_t kDimAt = 16;
constexpr std::size_t kMAt = 24;
constexpr std::size_t kEfConstructionAt = 32;
constexpr std::size_t kSeedAt = 40;
constexpr std::size_t kNodesAt = 48;
constexpr std::size_t kMaxLevelAt = 56;
constexpr std::size_t kEntryPointAt = 64;
constexpr std::size_t kUpperBlocksAt = 72;
constexpr std::size_t kHeaderBytes = 80;
constexpr std::size_t kChecksumBytes = 8;

Matrix<float> random_matrix(std::size_t rows, std::size_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    Matrix<float> m(rows, dim);
    for (float& v : m.data()) {
        v = uniform(rng);
    }
    return m;
}

HnswParams params(std::size_t m, std::size_t ef_construction) {
    return {.M = m, .ef_construction = ef_construction, .seed = 42};
}

HnswIndex build(std::size_t rows, std::size_t dim, Metric metric, const HnswParams& p) {
    HnswIndex index(dim, metric, p);
    index.add(random_matrix(rows, dim, 1));
    return index;
}

// For the corruption tests: small enough to load every variant of its ~600-byte file. M = 2
// makes upper layers common (P(level >= l) = 2^-l), so every section has content. Levels come
// from the index's own portable generator, so they are the same on every platform.
HnswIndex tiny_index() {
    return build(12, 2, Metric::L2, params(2, 8));
}

// A directory of its own for each test, removed with its contents at the end.
class TempDir {
public:
    TempDir() : path_(fs::temp_directory_path() / unique_name()) { fs::create_directories(path_); }
    ~TempDir() {
        std::error_code ignored;
        fs::permissions(path_, fs::perms::owner_all, fs::perm_options::add, ignored);
        fs::remove_all(path_, ignored);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    TempDir(TempDir&&) = delete;
    TempDir& operator=(TempDir&&) = delete;

    const fs::path& path() const { return path_; }
    fs::path operator/(std::string_view name) const { return path_ / name; }

    std::vector<std::string> entries() const {
        std::vector<std::string> names;
        for (const fs::directory_entry& entry : fs::directory_iterator(path_)) {
            names.push_back(entry.path().filename().string());
        }
        std::ranges::sort(names);
        return names;
    }

private:
    // Random, so the same test running in two build trees at once can't collide.
    static std::string unique_name() {
        const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
        std::random_device random;
        return std::string("vecdex_") + test->test_suite_name() + "_" + test->name() + "_" +
               std::to_string(random());
    }

    fs::path path_;
};

Bytes read_file(const fs::path& path) {
    Bytes bytes(fs::file_size(path));
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) {
        throw std::runtime_error("read_file: cannot read " + path.string());
    }
    return bytes;
}

void write_file(const fs::path& path, std::span<const std::byte> bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error("write_file: cannot write " + path.string());
    }
}

Bytes saved_bytes(const HnswIndex& index, const TempDir& dir) {
    const fs::path path = dir / "saved.vdx";
    index.save(path);
    return read_file(path);
}

template <typename T>
T get(std::span<const std::byte> file, std::size_t offset) {
    T value{};
    std::memcpy(&value, file.data() + offset, sizeof value);
    return value;
}

template <typename T>
void put(std::span<std::byte> file, std::size_t offset, T value) {
    std::memcpy(file.data() + offset, &value, sizeof value);
}

// Recomputes the trailing checksum, as someone crafting a malicious file would, so that the
// corruption reaches the validation behind it.
void reseal(Bytes& file) {
    const std::span<const std::byte> covered(file.data(), file.size() - kChecksumBytes);
    put(file, covered.size(), vecdex::detail::xxh64(covered));
}

// Where the sections of a file start (FORMAT.md, "Sections").
struct Sections {
    std::size_t nodes;
    std::size_t dim;
    std::size_t M;
    std::size_t max_level;
    std::size_t entry_point;
    std::vector<std::uint8_t> levels;
    std::size_t layer0_at;
    std::size_t upper_at;
    std::size_t vectors_at;
    std::size_t checksum_at;

    // The offset of the count that starts `node`'s link block on `layer`; its ids follow.
    std::size_t block_at(std::size_t node, std::size_t layer) const {
        if (layer == 0) {
            return layer0_at + node * (1 + 2 * M) * sizeof(Id);
        }
        std::size_t blocks_before = 0;
        for (std::size_t other = 0; other < node; ++other) {
            blocks_before += levels[other];
        }
        return upper_at + (blocks_before + layer - 1) * (1 + M) * sizeof(Id);
    }
    std::size_t vector_at(std::size_t node) const {
        return vectors_at + node * dim * sizeof(float);
    }
};

Sections sections_of(std::span<const std::byte> file) {
    Sections s{};
    s.nodes = get<std::uint64_t>(file, kNodesAt);
    s.dim = get<std::uint64_t>(file, kDimAt);
    s.M = get<std::uint64_t>(file, kMAt);
    s.max_level = get<std::uint64_t>(file, kMaxLevelAt);
    s.entry_point = get<std::uint64_t>(file, kEntryPointAt);
    const auto upper_blocks = get<std::uint64_t>(file, kUpperBlocksAt);
    for (std::size_t node = 0; node < s.nodes; ++node) {
        s.levels.push_back(get<std::uint8_t>(file, kHeaderBytes + node));
    }
    s.layer0_at = kHeaderBytes + s.nodes;
    s.upper_at = s.layer0_at + s.nodes * (1 + 2 * s.M) * sizeof(Id);
    s.vectors_at = s.upper_at + upper_blocks * (1 + s.M) * sizeof(Id);
    s.checksum_at = s.vectors_at + s.nodes * s.dim * sizeof(float);
    return s;
}

// Success if loading `file` throws std::runtime_error naming the file and containing `reason`.
// Any other exception (bad_alloc from trusting a count, invalid_argument from the constructor)
// escapes and fails the test.
::testing::AssertionResult rejected(const fs::path& path, std::span<const std::byte> file,
                                    std::string_view reason) {
    write_file(path, file);
    try {
        (void)HnswIndex::load(path);
    } catch (const std::runtime_error& e) {
        const std::string_view message = e.what();
        if (message.find(path.string()) == std::string_view::npos) {
            return ::testing::AssertionFailure()
                   << "the message doesn't name the file: " << message;
        }
        if (message.find(reason) == std::string_view::npos) {
            return ::testing::AssertionFailure()
                   << "expected \"" << reason << "\" in the message: " << message;
        }
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "the file loaded";
}

// Success if save() throws std::runtime_error naming `path`.
::testing::AssertionResult save_fails(const HnswIndex& index, const fs::path& path) {
    try {
        index.save(path);
    } catch (const std::runtime_error& e) {
        if (std::string_view(e.what()).find(path.string()) == std::string_view::npos) {
            return ::testing::AssertionFailure()
                   << "the message doesn't name the file: " << e.what();
        }
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "save succeeded";
}

// ---------------------------------------------------------------------------------------------
// Round trips

// Exact equality of ids and distances, not a tolerance (compare D-007): both indexes compute every
// distance with the same kernel, on bit-identical stored vectors and the same queries.
TEST(HnswSaveLoad, RoundTripGivesIdenticalResultsForEveryMetric) {
    const TempDir dir;
    const Matrix<float> queries = random_matrix(100, 16, 2);
    for (const Metric metric : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
        SCOPED_TRACE(::testing::Message() << "metric " << static_cast<int>(metric));
        const HnswIndex original = build(1000, 16, metric, params(8, 40));
        const fs::path path = dir / "index.vdx";
        original.save(path);
        const HnswIndex loaded = HnswIndex::load(path);

        EXPECT_EQ(loaded.size(), original.size());
        EXPECT_EQ(loaded.dim(), original.dim());
        EXPECT_EQ(loaded.metric(), metric);
        EXPECT_EQ(loaded.params().M, original.params().M);
        EXPECT_EQ(loaded.params().ef_construction, original.params().ef_construction);
        EXPECT_EQ(loaded.params().seed, original.params().seed);
        EXPECT_EQ(loaded.memory_bytes(), original.memory_bytes());
        constexpr std::array<std::size_t, 4> kEfs{1, 10, 40, 200};
        for (const std::size_t ef : kEfs) {
            for (std::size_t q = 0; q < queries.rows(); ++q) {
                ASSERT_EQ(loaded.search(queries.row(q), 10, ef),
                          original.search(queries.row(q), 10, ef))
                    << "ef " << ef << ", query " << q;
            }
        }
        const vecdex::GraphReport before = original.inspect();
        const vecdex::GraphReport after = loaded.inspect();
        EXPECT_TRUE(after.violations.empty());
        EXPECT_EQ(after.reachable, before.reachable);
        EXPECT_EQ(after.max_level, before.max_level);
        EXPECT_EQ(after.mean_degree0, before.mean_degree0);
    }
}

TEST(HnswSaveLoad, EmptyIndexRoundTrips) {
    const TempDir dir;
    const HnswIndex original(5, Metric::Cosine, params(6, 30));
    original.save(dir / "empty.vdx");
    HnswIndex loaded = HnswIndex::load(dir / "empty.vdx");
    EXPECT_EQ(loaded.size(), 0U);
    EXPECT_EQ(loaded.dim(), 5U);
    EXPECT_EQ(loaded.metric(), Metric::Cosine);
    EXPECT_EQ(loaded.params().M, 6U);
    EXPECT_EQ(loaded.params().ef_construction, 30U);
    const std::vector<float> v{1.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    EXPECT_TRUE(loaded.search(v, 3, 10).empty());
    EXPECT_EQ(loaded.add(v), 0U);
    EXPECT_EQ(loaded.search(v, 1, 10).front().id, 0U);
}

TEST(HnswSaveLoad, SingleNodeRoundTrips) {
    const TempDir dir;
    HnswIndex original(3);
    original.add(std::vector<float>{1.0F, 2.0F, 3.0F});
    original.save(dir / "one.vdx");
    HnswIndex loaded = HnswIndex::load(dir / "one.vdx");
    const std::vector<vecdex::Neighbor> expected{{0.0F, 0}};
    EXPECT_EQ(loaded.search(std::vector<float>{1.0F, 2.0F, 3.0F}, 1, 1), expected);
    EXPECT_EQ(loaded.add(std::vector<float>{0.0F, 0.0F, 0.0F}), 1U);
    EXPECT_EQ(loaded.search(std::vector<float>{0.0F, 0.0F, 0.0F}, 1, 10).front().id, 1U);
    EXPECT_TRUE(loaded.inspect().violations.empty());
}

// Determinism: save -> load -> add builds exactly the graph that add on the original builds. The
// new nodes' levels depend on the generator state that load() restores; without it, the files
// differ in the levels section.
TEST(HnswSaveLoad, AddAfterLoadBuildsTheSameGraphAsWithoutTheRoundTrip) {
    const TempDir dir;
    const Matrix<float> first = random_matrix(500, 8, 1);
    const Matrix<float> second = random_matrix(300, 8, 2);
    HnswIndex direct(8, Metric::L2, params(8, 40));
    direct.add(first);
    direct.save(dir / "half.vdx");
    direct.add(second);
    HnswIndex resumed = HnswIndex::load(dir / "half.vdx");
    resumed.add(second);

    direct.save(dir / "direct.vdx");
    resumed.save(dir / "resumed.vdx");
    EXPECT_EQ(read_file(dir / "direct.vdx"), read_file(dir / "resumed.vdx"));
}

// The file is a pure function of the index's state, so a loaded index saves to the same bytes.
TEST(HnswSaveLoad, SavingALoadedIndexReproducesTheFile) {
    const TempDir dir;
    build(300, 8, Metric::InnerProduct, params(8, 40)).save(dir / "first.vdx");
    HnswIndex::load(dir / "first.vdx").save(dir / "second.vdx");
    EXPECT_EQ(read_file(dir / "first.vdx"), read_file(dir / "second.vdx"));
}

TEST(HnswSaveLoad, MovedFromIndexSavesAsEmpty) {
    const TempDir dir;
    HnswIndex a = build(50, 4, Metric::L2, params(4, 20));
    const HnswIndex b = std::move(a);
    // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move): the point of the test
    a.save(dir / "moved.vdx");
    // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    const HnswIndex loaded = HnswIndex::load(dir / "moved.vdx");
    EXPECT_EQ(loaded.size(), 0U);
    EXPECT_EQ(loaded.dim(), 4U);
    EXPECT_EQ(b.size(), 50U);
}

// The bytes, field by field, against docs/FORMAT.md.
TEST(HnswSaveLoad, FileFollowsTheDocumentedFormat) {
    const TempDir dir;
    constexpr std::size_t kRows = 40;
    constexpr std::size_t kDim = 3;
    const Matrix<float> vectors = random_matrix(kRows, kDim, 1);
    HnswIndex index(kDim, Metric::L2, {.M = 5, .ef_construction = 17, .seed = 1234});
    index.add(vectors);
    const Bytes file = saved_bytes(index, dir);

    constexpr std::array<unsigned char, 8> kMagic{0x89, 'V', 'D', 'X', 'H', 'N', 'S', 'W'};
    EXPECT_EQ(std::memcmp(file.data(), kMagic.data(), kMagic.size()), 0);
    EXPECT_EQ(get<std::uint32_t>(file, kVersionAt), 1U);
    EXPECT_EQ(get<std::uint32_t>(file, kMetricAt), 0U);
    EXPECT_EQ(get<std::uint64_t>(file, kDimAt), kDim);
    EXPECT_EQ(get<std::uint64_t>(file, kMAt), 5U);
    EXPECT_EQ(get<std::uint64_t>(file, kEfConstructionAt), 17U);
    EXPECT_EQ(get<std::uint64_t>(file, kSeedAt), 1234U);
    EXPECT_EQ(get<std::uint64_t>(file, kNodesAt), kRows);
    EXPECT_EQ(get<std::uint64_t>(file, kMaxLevelAt), index.inspect().max_level);

    const Sections s = sections_of(file);
    EXPECT_EQ(s.levels[s.entry_point], s.max_level);
    std::size_t level_sum = 0;
    for (const std::uint8_t level : s.levels) {
        level_sum += level;
    }
    EXPECT_EQ(get<std::uint64_t>(file, kUpperBlocksAt), level_sum);
    ASSERT_EQ(s.checksum_at + kChecksumBytes, file.size());
    EXPECT_EQ(get<std::uint64_t>(file, s.checksum_at),
              vecdex::detail::xxh64(std::span(file).first(s.checksum_at)));
    for (std::size_t r = 0; r < kRows; ++r) {
        for (std::size_t c = 0; c < kDim; ++c) {
            EXPECT_EQ(get<float>(file, s.vector_at(r) + c * sizeof(float)), vectors.row(r)[c]);
        }
    }

    // Metric codes are fixed by the spec, whatever the enum's values.
    for (const auto& [metric, code] :
         {std::pair{Metric::L2, 0U}, std::pair{Metric::InnerProduct, 1U},
          std::pair{Metric::Cosine, 2U}}) {
        EXPECT_EQ(get<std::uint32_t>(saved_bytes(HnswIndex(2, metric), dir), kMetricAt), code);
    }
}

// ---------------------------------------------------------------------------------------------
// Rejected files

TEST(HnswSaveLoad, EveryTruncationIsRejected) {
    const TempDir dir;
    const Bytes file = saved_bytes(tiny_index(), dir);
    for (std::size_t length = 0; length < file.size(); ++length) {
        const std::string_view reason =
            length < kHeaderBytes + kChecksumBytes ? "too small" : "truncated or corrupted";
        EXPECT_TRUE(rejected(dir / "cut.vdx", std::span(file).first(length), reason))
            << "length " << length << " of " << file.size();
    }
}

// Without resealing, the checksum (or a check before it) catches every single flipped bit.
TEST(HnswSaveLoad, EveryFlippedBitIsRejected) {
    const TempDir dir;
    const Bytes original = saved_bytes(tiny_index(), dir);
    for (std::size_t offset = 0; offset < original.size(); ++offset) {
        for (unsigned bit = 0; bit < 8; ++bit) {
            Bytes file = original;
            file[offset] ^= std::byte{1} << bit;
            EXPECT_TRUE(rejected(dir / "flipped.vdx", file, ""))
                << "offset " << offset << " bit " << bit;
        }
    }
}

// A small fuzzer over every bit: flip it, reseal the checksum so the damage reaches the
// validation, and load. Each attempt must throw std::runtime_error or yield an index that
// inspect() finds sound and that searches and grows normally. Under ASan/UBSan and Valgrind this
// shows that no such file makes load(), search() or add() touch memory they shouldn't.
TEST(HnswSaveLoad, ResealedCorruptionIsRejectedOrHarmless) {
    const TempDir dir;
    const Bytes original = saved_bytes(tiny_index(), dir);
    const fs::path path = dir / "fuzzed.vdx";
    const std::vector<float> query{0.25F, -0.5F};
    std::size_t rejections = 0;
    std::size_t acceptances = 0;
    for (std::size_t offset = 0; offset + kChecksumBytes < original.size(); ++offset) {
        for (unsigned bit = 0; bit < 8; ++bit) {
            Bytes file = original;
            file[offset] ^= std::byte{1} << bit;
            reseal(file);
            write_file(path, file);
            std::optional<HnswIndex> index;
            try {
                index.emplace(HnswIndex::load(path));
            } catch (const std::runtime_error&) {
                ++rejections;
                continue;
            }
            ++acceptances;
            SCOPED_TRACE(::testing::Message() << "offset " << offset << " bit " << bit);
            const vecdex::GraphReport report = index->inspect();
            EXPECT_TRUE(report.violations.empty()) << report.violations.front();
            EXPECT_FALSE(index->search(query, 5, 20).empty());
            EXPECT_EQ(index->add(query), 12U);
            EXPECT_TRUE(index->inspect().violations.empty());
        }
    }
    RecordProperty("rejected", std::to_string(rejections));
    RecordProperty("accepted", std::to_string(acceptances));
    // Both outcomes occur: e.g. a flipped level is always caught, a flipped seed never is.
    EXPECT_GT(rejections, 0U);
    EXPECT_GT(acceptances, 0U);
}

TEST(HnswSaveLoad, HostileHeaderFieldsAreRejected) {
    const TempDir dir;
    const Bytes original = saved_bytes(tiny_index(), dir);
    const auto nodes = get<std::uint64_t>(original, kNodesAt);
    struct Case {
        std::size_t offset;
        std::uint64_t value;
        bool wide;  // a 64-bit field, else 32-bit
        std::string_view reason;
    };
    const std::array<Case, 15> kCases{{
        {kVersionAt, 0, false, "format version 0 is not supported"},
        {kVersionAt, 2, false, "format version 2 is not supported"},
        {kMetricAt, 3, false, "unknown metric code 3"},
        {kDimAt, 0, true, "dimension is 0"},
        {kMAt, 1, true, "below the minimum of 2"},
        {kMAt, std::uint64_t{1} << 40U, true, "too large for 32-bit link counts"},
        {kEfConstructionAt, 0, true, "ef_construction is 0"},
        {kNodesAt, std::uint64_t{1} << 32U, true, "more than 32-bit ids can address"},
        {kNodesAt, nodes + 1, true, "truncated or corrupted"},
        {kNodesAt, nodes - 1, true, "truncated or corrupted"},
        {kMaxLevelAt, 256, true, "above 255"},
        {kEntryPointAt, nodes, true, "is not one of the"},
        {kUpperBlocksAt, 0, true, "truncated or corrupted"},
        // Sizes whose products wrap around 64 bits must not pass for small ones.
        {kDimAt, std::uint64_t{1} << 62U, true, "overflow 64 bits"},
        {kUpperBlocksAt, std::uint64_t{1} << 62U, true, "overflow 64 bits"},
    }};
    for (const Case& c : kCases) {
        Bytes file = original;
        if (c.wide) {
            put(file, c.offset, c.value);
        } else {
            put(file, c.offset, static_cast<std::uint32_t>(c.value));
        }
        reseal(file);
        EXPECT_TRUE(rejected(dir / "bad.vdx", file, c.reason)) << "offset " << c.offset;
    }

    Bytes empty = saved_bytes(HnswIndex(2), dir);
    put(empty, kMaxLevelAt, std::uint64_t{1});
    reseal(empty);
    EXPECT_TRUE(rejected(dir / "bad.vdx", empty, "an empty index must have"));
}

// A count in the header must not size an allocation before it is checked against the file size.
// These imply buffers of about 50 TB. A load() that trusted them would fail at once (ASan aborts
// on the request, and elsewhere std::bad_alloc escapes rejected()), rather than zero-fill
// gigabytes and hang: that is what the largest valid node count, 2^32 - 1, would do.
TEST(HnswSaveLoad, HugeCountsFailWithoutAllocating) {
    const TempDir dir;
    const Bytes original = saved_bytes(tiny_index(), dir);
    for (const auto& [offset, value] : {std::pair{kUpperBlocksAt, std::uint64_t{1} << 42U},
                                        std::pair{kDimAt, std::uint64_t{1} << 40U}}) {
        Bytes file = original;
        put(file, offset, value);
        EXPECT_TRUE(rejected(dir / "huge.vdx", file, "truncated or corrupted"))
            << "offset " << offset;
    }
}

TEST(HnswSaveLoad, HostileGraphsAreRejected) {
    const TempDir dir;
    const Bytes original = saved_bytes(tiny_index(), dir);
    const Sections s = sections_of(original);
    ASSERT_GE(s.max_level, 1U) << "the test needs upper layers";

    // Nodes to corrupt: one with at least two links on layer 0, one with links on layer 1, one
    // that lives only on layer 0, and one below the top layer.
    std::optional<std::size_t> linked_node;
    std::optional<std::size_t> upper_node;
    std::optional<std::size_t> ground_node;
    std::optional<std::size_t> low_node;
    for (std::size_t node = 0; node < s.nodes; ++node) {
        if (get<Id>(original, s.block_at(node, 0)) >= 2) {
            linked_node = linked_node.value_or(node);
        }
        if (s.levels[node] >= 1 && get<Id>(original, s.block_at(node, 1)) > 0) {
            upper_node = upper_node.value_or(node);
        }
        if (s.levels[node] == 0) {
            ground_node = ground_node.value_or(node);
        }
        if (s.levels[node] < s.max_level) {
            low_node = low_node.value_or(node);
        }
    }
    ASSERT_TRUE(linked_node && upper_node && ground_node && low_node);
    const std::size_t node = *linked_node;
    const std::size_t block = s.block_at(node, 0);

    const auto expect_rejected = [&](auto&& corrupt, std::string_view reason) {
        Bytes file = original;
        corrupt(file);
        reseal(file);
        EXPECT_TRUE(rejected(dir / "bad.vdx", file, reason));
    };
    const auto id_at = [](std::size_t block_offset, std::size_t i) {
        return block_offset + (1 + i) * sizeof(Id);
    };

    expect_rejected([&](Bytes& f) { put(f, kHeaderBytes + *ground_node, std::uint8_t{255}); },
                    "above the max level");
    expect_rejected([&](Bytes& f) { put(f, kHeaderBytes + *ground_node, std::uint8_t{1}); },
                    "the levels add up to");
    expect_rejected([&](Bytes& f) { put(f, kEntryPointAt, std::uint64_t{*low_node}); },
                    "not the max level");
    expect_rejected([&](Bytes& f) { put(f, block, static_cast<Id>(2 * s.M + 1)); },
                    "links on layer 0 (max");
    expect_rejected([&](Bytes& f) { put(f, id_at(block, 0), static_cast<Id>(s.nodes)); },
                    "but there are only");
    expect_rejected([&](Bytes& f) { put(f, id_at(block, 0), static_cast<Id>(node)); },
                    "links to itself");
    expect_rejected([&](Bytes& f) { put(f, id_at(block, 1), get<Id>(original, id_at(block, 0))); },
                    "twice");
    expect_rejected(
        [&](Bytes& f) {
            put(f, id_at(s.block_at(*upper_node, 1), 0), static_cast<Id>(*ground_node));
        },
        "above its level");
    expect_rejected(
        [&](Bytes& f) { put(f, s.vector_at(3), std::numeric_limits<float>::quiet_NaN()); },
        "NaN or infinity");
    expect_rejected(
        [&](Bytes& f) { put(f, s.vector_at(5) + 4, std::numeric_limits<float>::infinity()); },
        "NaN or infinity");
}

TEST(HnswSaveLoad, OtherFilesAreRejected) {
    const TempDir dir;
    Bytes file = saved_bytes(tiny_index(), dir);
    file[0] = std::byte{'v'};
    reseal(file);
    EXPECT_TRUE(rejected(dir / "bad.vdx", file, "wrong magic"));

    std::string text;
    for (int line = 0; line < 10; ++line) {
        text += "Not an index, but longer than a header.\n";
    }
    EXPECT_TRUE(rejected(dir / "text.vdx", std::as_bytes(std::span(text)), "wrong magic"));
    EXPECT_TRUE(rejected(dir / "zeros.vdx", Bytes(1000), "wrong magic"));
    EXPECT_TRUE(rejected(dir / "empty.vdx", Bytes(), "too small"));
}

TEST(HnswSaveLoad, MissingFileOrDirectoryIsRejected) {
    const TempDir dir;
    for (const fs::path& path : {dir / "no_such_file.vdx", dir.path()}) {
        try {
            (void)HnswIndex::load(path);
            ADD_FAILURE() << "loaded " << path;
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string_view(e.what()).find(path.string()), std::string_view::npos)
                << e.what();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Saving

TEST(HnswSaveLoad, SaveReplacesAnExistingFileAndLeavesNoTemporaryFile) {
    const TempDir dir;
    const fs::path path = dir / "index.vdx";
    build(20, 4, Metric::L2, params(4, 20)).save(path);
    build(30, 4, Metric::L2, params(4, 20)).save(path);
    EXPECT_EQ(HnswIndex::load(path).size(), 30U);
    EXPECT_EQ(dir.entries(), std::vector<std::string>{"index.vdx"});
}

TEST(HnswSaveLoad, SaveIntoAMissingDirectoryThrows) {
    const TempDir dir;
    EXPECT_TRUE(save_fails(tiny_index(), dir / "no_such_dir" / "index.vdx"));
    EXPECT_TRUE(dir.entries().empty());
}

// The temporary file gets written in full here; only the final rename fails, because the
// destination is a directory. The temporary file must not be left behind.
TEST(HnswSaveLoad, FailedRenameRemovesTheTemporaryFile) {
    const TempDir dir;
    fs::create_directory(dir / "target");
    EXPECT_TRUE(save_fails(tiny_index(), dir / "target"));
    EXPECT_EQ(dir.entries(), std::vector<std::string>{"target"});
    EXPECT_TRUE(fs::is_empty(dir / "target"));
}

TEST(HnswSaveLoad, SaveIntoAReadOnlyDirectoryThrowsAndKeepsTheOldFile) {
    const TempDir dir;
    const fs::path path = dir / "index.vdx";
    tiny_index().save(path);
    const Bytes before = read_file(path);
    fs::permissions(dir.path(), fs::perms::owner_read | fs::perms::owner_exec);
    if (std::ofstream(dir / "probe")) {  // root, for one, ignores the permissions
        fs::permissions(dir.path(), fs::perms::owner_all);
        GTEST_SKIP() << "this user can write to a read-only directory";
    }
    EXPECT_TRUE(save_fails(build(30, 2, Metric::L2, params(2, 8)), path));
    fs::permissions(dir.path(), fs::perms::owner_all);
    EXPECT_EQ(read_file(path), before);
    EXPECT_EQ(dir.entries(), std::vector<std::string>{"index.vdx"});
}

}  // namespace
