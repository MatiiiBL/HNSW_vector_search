// HnswIndex::save and HnswIndex::load: the vecdex HNSW file format, version 1. docs/FORMAT.md
// specifies it byte by byte; change the two together.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <istream>
#include <limits>
#include <memory>
#include <ostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "file_sync.hpp"
#include "search_scratch.hpp"
#include "vecdex/hnsw.hpp"
#include "vector_ops.hpp"
#include "xxhash64.hpp"

namespace vecdex {
namespace {

namespace fs = std::filesystem;

// The file is little-endian, like every target vecdex builds for, so numbers are written and
// read exactly as they lie in memory: the arrays move in large blocks, at the speed of the disk.
// A big-endian port would byte-swap in Writer and Reader; until then it fails to compile here.
static_assert(std::endian::native == std::endian::little,
              "the index format is little-endian; a big-endian host would need byte swapping");
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t),
              "the format's 64-bit counts are used as size_t");
static_assert(2 * HnswParams::kMaxM <= std::numeric_limits<Id>::max(),
              "a link block's count, up to 2M, is stored as an Id");

// "\x89VDXHNSW". A non-ASCII first byte, as in PNG, so that no text file can start like this.
constexpr std::array<std::byte, 8> kMagic{std::byte{0x89}, std::byte{'V'}, std::byte{'D'},
                                          std::byte{'X'},  std::byte{'H'}, std::byte{'N'},
                                          std::byte{'S'},  std::byte{'W'}};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint64_t kHeaderBytes = 80;
constexpr std::uint64_t kChecksumBytes = 8;

// The file's metric codes are the positions in this table (FORMAT.md), so reordering the enum
// can't silently change the format.
constexpr std::array kMetrics{Metric::L2, Metric::InnerProduct, Metric::Cosine};

// Reads and writes go in chunks of this size, each hashed while it is still in cache.
constexpr std::size_t kChunkBytes = std::size_t{1} << 20U;

[[noreturn]] void fail(const fs::path& path, const std::string& what) {
    throw std::runtime_error(std::format("{}: {}", path.string(), what));
}

// Writes to a stream and hashes what it writes, so the checksum covers exactly the bytes in the
// file. Stream errors are sticky: save() checks the stream once, after closing it.
class Writer {
public:
    explicit Writer(std::ostream& out) noexcept : out_(out) {}

    void write(std::span<const std::byte> bytes) {
        for (std::size_t done = 0; done < bytes.size(); done += kChunkBytes) {
            const auto chunk = bytes.subspan(done, std::min(kChunkBytes, bytes.size() - done));
            hash_.update(chunk);
            // Any object's bytes may be accessed through char*, so this cast is well-defined.
            out_.write(reinterpret_cast<const char*>(chunk.data()),
                       static_cast<std::streamsize>(chunk.size()));
        }
    }
    void write_u32(std::uint32_t value) { write(std::as_bytes(std::span(&value, 1))); }
    void write_u64(std::uint64_t value) { write(std::as_bytes(std::span(&value, 1))); }

    std::uint64_t digest() const noexcept { return hash_.digest(); }

private:
    std::ostream& out_;
    detail::Xxh64 hash_;
};

// Reads from a stream and hashes what it reads. Fails if the stream ends early, which can still
// happen after the file-size check if the file shrinks while it is being read.
class Reader {
public:
    Reader(std::istream& in, const fs::path& path) noexcept : in_(in), path_(path) {}

    void read(std::span<std::byte> bytes) {
        for (std::size_t done = 0; done < bytes.size(); done += kChunkBytes) {
            const auto chunk = bytes.subspan(done, std::min(kChunkBytes, bytes.size() - done));
            const auto size = static_cast<std::streamsize>(chunk.size());
            in_.read(reinterpret_cast<char*>(chunk.data()), size);  // as in Writer::write
            if (in_.gcount() != size) {
                fail(path_, "unexpected end of file");
            }
            hash_.update(chunk);
        }
    }
    std::uint32_t read_u32() {
        std::uint32_t value = 0;
        read(std::as_writable_bytes(std::span(&value, 1)));
        return value;
    }
    std::uint64_t read_u64() {
        std::uint64_t value = 0;
        read(std::as_writable_bytes(std::span(&value, 1)));
        return value;
    }

    std::uint64_t digest() const noexcept { return hash_.digest(); }

private:
    std::istream& in_;
    const fs::path& path_;
    detail::Xxh64 hash_;
};

// The fixed-size header (FORMAT.md, "Header"). Always written and read one field at a time,
// never as a struct: a struct's padding is up to the compiler, and padding is uninitialized.
struct Header {
    std::array<std::byte, 8> magic{};
    std::uint32_t version = 0;
    std::uint32_t metric = 0;
    std::uint64_t dim = 0;
    std::uint64_t M = 0;
    std::uint64_t ef_construction = 0;
    std::uint64_t seed = 0;
    std::uint64_t nodes = 0;
    std::uint64_t max_level = 0;
    std::uint64_t entry_point = 0;
    std::uint64_t upper_blocks = 0;  // link blocks above layer 0: the sum of all levels
};
static_assert(kHeaderBytes == 8 + 2 * 4 + 8 * 8);

void write_header(Writer& out, const Header& header) {
    out.write(header.magic);
    out.write_u32(header.version);
    out.write_u32(header.metric);
    for (const std::uint64_t field :
         {header.dim, header.M, header.ef_construction, header.seed, header.nodes, header.max_level,
          header.entry_point, header.upper_blocks}) {
        out.write_u64(field);
    }
}

Header read_header(Reader& in) {
    Header header;
    in.read(header.magic);
    header.version = in.read_u32();
    header.metric = in.read_u32();
    for (std::uint64_t* field :
         {&header.dim, &header.M, &header.ef_construction, &header.seed, &header.nodes,
          &header.max_level, &header.entry_point, &header.upper_blocks}) {
        *field = in.read_u64();
    }
    return header;
}

// The header, validated and converted to the index's types.
struct Layout {
    Metric metric = Metric::L2;
    std::size_t dim = 0;
    HnswParams params;
    std::size_t nodes = 0;
    std::size_t max_level = 0;
    Id entry_point = 0;
    std::size_t upper_blocks = 0;
    std::size_t layer0_block = 0;  // ids per layer-0 block: 1 + 2M
    std::size_t upper_block = 0;   // ids per upper-layer block: 1 + M
};

// Overflow-checked arithmetic on header fields. A product that wrapped around could pass the
// file-size check and then size a buffer too small for the bytes read into it.
std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const fs::path& path) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        fail(path, "the header's sizes overflow 64 bits");
    }
    return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const fs::path& path) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b) {
        fail(path, "the header's sizes overflow 64 bits");
    }
    return a + b;
}

// Checks every header field on its own, then that the sizes it implies add up to exactly the
// file's size. Only then may they size an allocation: no buffer can outgrow the file.
Layout check_header(const Header& h, std::uintmax_t file_size, const fs::path& path) {
    if (h.magic != kMagic) {
        fail(path, "not a vecdex HNSW index (wrong magic bytes)");
    }
    if (h.version != kVersion) {
        fail(path, std::format("format version {} is not supported (this build reads version {})",
                               h.version, kVersion));
    }
    if (h.metric >= kMetrics.size()) {
        fail(path, std::format("unknown metric code {}", h.metric));
    }
    // The constructor's rules, reported as a bad file rather than as bad arguments.
    if (h.dim == 0) {
        fail(path, "dimension is 0");
    }
    if (h.M < 2) {
        fail(path, std::format("M is {}, below the minimum of 2", h.M));
    }
    if (h.M > HnswParams::kMaxM) {
        fail(path, std::format("M is {}, above the maximum of {}", h.M, HnswParams::kMaxM));
    }
    if (h.ef_construction == 0) {
        fail(path, "ef_construction is 0");
    }
    if (h.nodes > std::numeric_limits<Id>::max()) {
        fail(path, std::format("{} nodes is more than 32-bit ids can address", h.nodes));
    }
    if (h.max_level > std::numeric_limits<std::uint8_t>::max()) {
        fail(path, std::format("max level {} is above 255", h.max_level));
    }
    if (h.nodes == 0 && (h.max_level != 0 || h.entry_point != 0 || h.upper_blocks != 0)) {
        fail(path, "an empty index must have max level, entry point and upper blocks 0");
    }
    if (h.nodes > 0 && h.entry_point >= h.nodes) {
        fail(path,
             std::format("entry point {} is not one of the {} nodes", h.entry_point, h.nodes));
    }

    // FORMAT.md, "Sections": levels, layer-0 blocks, upper-layer blocks, vectors.
    const std::uint64_t layer0_block = 1 + 2 * h.M;  // no overflow: M <= kMaxM
    const std::uint64_t upper_block = 1 + h.M;
    const auto bytes = [&path](std::uint64_t count, std::uint64_t per_item, std::uint64_t size) {
        return checked_mul(checked_mul(count, per_item, path), size, path);
    };
    std::uint64_t expected = kHeaderBytes + kChecksumBytes;
    expected = checked_add(expected, bytes(h.nodes, 1, sizeof(std::uint8_t)), path);
    expected = checked_add(expected, bytes(h.nodes, layer0_block, sizeof(Id)), path);
    expected = checked_add(expected, bytes(h.upper_blocks, upper_block, sizeof(Id)), path);
    expected = checked_add(expected, bytes(h.nodes, h.dim, sizeof(float)), path);
    if (expected != file_size) {
        fail(path, std::format("the header describes {} bytes but the file has {}; it is "
                               "truncated or corrupted",
                               expected, file_size));
    }
    return {.metric = kMetrics[h.metric],
            .dim = h.dim,
            .params = {.M = h.M, .ef_construction = h.ef_construction, .seed = h.seed},
            .nodes = h.nodes,
            .max_level = h.max_level,
            .entry_point = static_cast<Id>(h.entry_point),
            .upper_blocks = h.upper_blocks,
            .layer0_block = layer0_block,
            .upper_block = upper_block};
}

// Every level within the max level, the entry point on the top layer, and the levels adding up
// to the upper-layer block count the header announced, which sized that section.
void check_levels(std::span<const std::uint8_t> levels, const Layout& layout,
                  const fs::path& path) {
    std::size_t blocks = 0;
    for (std::size_t node = 0; node < levels.size(); ++node) {
        if (levels[node] > layout.max_level) {
            fail(path, std::format("node {} has level {}, above the max level {}", node,
                                   levels[node], layout.max_level));
        }
        blocks += levels[node];
    }
    if (blocks != layout.upper_blocks) {
        fail(path, std::format("the levels add up to {} upper-layer blocks, but the header says {}",
                               blocks, layout.upper_blocks));
    }
    if (!levels.empty() && levels[layout.entry_point] != layout.max_level) {
        fail(path, std::format("entry point {} has level {}, not the max level {}",
                               layout.entry_point, levels[layout.entry_point], layout.max_level));
    }
}

// Cuts the upper-layer section into each node's blocks for its layers 1..level, in the layout of
// HnswIndex::upper_. Precondition (check_levels): the levels add up to upper.size() / block.
std::vector<std::vector<Id>> split_upper(std::span<const Id> upper,
                                         std::span<const std::uint8_t> levels, std::size_t block) {
    std::vector<std::vector<Id>> per_node;
    per_node.reserve(levels.size());
    for (const std::uint8_t level : levels) {
        const std::span<const Id> blocks = upper.first(std::size_t{level} * block);
        per_node.emplace_back(blocks.begin(), blocks.end());
        upper = upper.subspan(blocks.size());
    }
    return per_node;
}

// A fresh name next to `path`: "<path>.tmp-<16 random hex digits>". In the same directory, so
// the final rename stays within one file system, where it is atomic. Unique, so concurrent saves
// to one path never write into each other's temporary file.
fs::path temporary_path(const fs::path& path) {
    std::random_device random;
    const std::uint64_t suffix = (std::uint64_t{random()} << 32U) | random();
    fs::path temp = path;
    temp += std::format(".tmp-{:016x}", suffix);
    return temp;
}

// The directory whose entry for `path` a rename to `path` changes.
fs::path parent_directory(const fs::path& path) {
    const fs::path parent = path.parent_path();
    return parent.empty() ? fs::path(".") : parent;
}

// Removes a file when it goes out of scope, unless release()d. save() releases its temporary
// file only after renaming it into place, so every failure, exceptions included, cleans it up.
class RemoveOnExit {
public:
    explicit RemoveOnExit(fs::path path) noexcept : path_(std::move(path)) {}
    ~RemoveOnExit() {
        if (!path_.empty()) {
            std::error_code ignored;  // nothing better to do about it in a destructor
            fs::remove(path_, ignored);
        }
    }
    RemoveOnExit(const RemoveOnExit&) = delete;
    RemoveOnExit& operator=(const RemoveOnExit&) = delete;
    RemoveOnExit(RemoveOnExit&&) = delete;
    RemoveOnExit& operator=(RemoveOnExit&&) = delete;

    void release() noexcept { path_.clear(); }

private:
    fs::path path_;
};

}  // namespace

void HnswIndex::save(const std::filesystem::path& path) const {
    std::size_t upper_blocks = 0;
    for (const std::uint8_t level : levels_) {
        upper_blocks += level;
    }
    const bool empty = size() == 0;
    const Header header{
        .magic = kMagic,
        .version = kVersion,
        .metric =
            static_cast<std::uint32_t>(std::ranges::find(kMetrics, metric_) - kMetrics.begin()),
        .dim = dim(),
        .M = params_.M,
        .ef_construction = params_.ef_construction,
        .seed = params_.seed,
        .nodes = size(),
        // A moved-from index keeps its old entry point and max level; empty means zeros.
        .max_level = empty ? 0 : max_level_,
        .entry_point = empty ? 0 : entry_point_,
        .upper_blocks = upper_blocks,
    };

    const fs::path temp = temporary_path(path);
    RemoveOnExit cleanup(temp);
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) {
            fail(path, std::format("cannot create the temporary file {}", temp.string()));
        }
        Writer writer(out);
        write_header(writer, header);
        writer.write(std::as_bytes(std::span(levels_)));
        writer.write(std::as_bytes(std::span(layer0_)));
        for (const std::vector<Id>& blocks : upper_) {
            writer.write(std::as_bytes(std::span(blocks)));
        }
        writer.write(std::as_bytes(vectors_.data()));
        writer.write_u64(writer.digest());  // the checksum covers every byte before it
        out.close();                        // flushes; a failed write or flush sets failbit
        if (!out) {
            fail(path, std::format("cannot write the temporary file {}", temp.string()));
        }
    }
    // Durable, not only atomic: the data must be on stable storage before the rename is, or a
    // power loss could keep the rename but not the data, replacing the old index with a file
    // that fails its checksum.
    std::error_code ec = detail::sync_file(temp);
    if (ec) {
        fail(path, std::format("cannot flush the temporary file {} to storage: {}", temp.string(),
                               ec.message()));
    }
    fs::rename(temp, path, ec);  // atomically replaces any existing file
    if (ec) {
        fail(path, std::format("cannot replace it with {}: {}", temp.string(), ec.message()));
    }
    cleanup.release();

    // Makes the rename itself survive a power loss. A failure here comes too late to undo:
    // `path` already holds the new, complete index and the old one is gone, so it is reported,
    // and the file is left in place.
    const fs::path dir = parent_directory(path);
    ec = detail::sync_directory(dir);
    if (ec) {
        fail(path, std::format("saved, but cannot flush the directory {} to storage: {}",
                               dir.string(), ec.message()));
    }
}

HnswIndex HnswIndex::load(const std::filesystem::path& path) {
    std::error_code ec;
    const std::uintmax_t file_size = fs::file_size(path, ec);
    if (ec) {
        fail(path, ec.message());
    }
    if (file_size < kHeaderBytes + kChecksumBytes) {
        fail(path, std::format("{} bytes is too small for a vecdex index", file_size));
    }
    // A large stream buffer: with its default 4 KB one, libc++'s filebuf made reading the file
    // 2-8 times slower than fread (measured on 65 MB, 2026-10-27). Implementation-defined, so at
    // worst ignored. Left uninitialized, as the stream only reads what it has filled in; zeroing
    // a megabyte per load is slow in unoptimized builds. Declared first: it must outlive `in`.
    const auto stream_buffer = std::make_unique_for_overwrite<char[]>(kChunkBytes);
    std::ifstream in;
    in.rdbuf()->pubsetbuf(stream_buffer.get(), static_cast<std::streamsize>(kChunkBytes));
    in.open(path, std::ios::binary);
    if (!in) {
        fail(path, "cannot open file");
    }
    Reader reader(in, path);
    const Layout layout = check_header(read_header(reader), file_size, path);

    // Read everything, then check the checksum before interpreting any of it, so an accidentally
    // damaged file is reported as such. Every size here was matched against the file size.
    std::vector<std::uint8_t> levels(layout.nodes);
    std::vector<Id> layer0(layout.nodes * layout.layer0_block);
    std::vector<Id> upper(layout.upper_blocks * layout.upper_block);
    Matrix<float> vectors(layout.nodes, layout.dim);
    reader.read(std::as_writable_bytes(std::span(levels)));
    reader.read(std::as_writable_bytes(std::span(layer0)));
    reader.read(std::as_writable_bytes(std::span(upper)));
    reader.read(std::as_writable_bytes(vectors.data()));
    const std::uint64_t computed = reader.digest();  // before the checksum's own bytes
    if (reader.read_u64() != computed) {
        fail(path, "checksum mismatch: the file is corrupted");
    }

    // Anyone can write a file with a valid checksum, so validate it as hostile before anything
    // follows a link or ranks a distance. NaN would break the strict weak ordering that the heaps
    // and sorts rely on, which is why add() rejects it too.
    check_levels(levels, layout, path);
    if (!detail::all_finite(vectors.data())) {
        fail(path, "a stored vector contains NaN or infinity");
    }
    HnswIndex index(layout.dim, layout.metric, layout.params);
    index.vectors_ = std::move(vectors);
    index.upper_ = split_upper(upper, levels, layout.upper_block);
    index.levels_ = std::move(levels);
    index.layer0_ = std::move(layer0);
    index.entry_point_ = layout.entry_point;
    index.max_level_ = layout.max_level;
    index.check_links(path);

    // Each stored node drew its level from the generator exactly once, so this puts the
    // generator where the saved index's was, and add() continues the same sequence.
    index.rng_.discard(layout.nodes);
    return index;
}

// Called by load() before the graph is used. These are the invariants inspect() reports, but
// enforced, and in an order that never reads outside a block: each count is checked before the
// links it covers, and each id before it is used as an index.
void HnswIndex::check_links(const std::filesystem::path& path) const {
    const std::size_t n = size();
    detail::VisitedList seen;  // duplicate detection, reset per block in O(1)
    for (Id node = 0; node < n; ++node) {
        const std::size_t level = levels_[node];
        for (std::size_t layer = 0; layer <= level; ++layer) {
            const std::size_t count = link_block(node, layer)[0];
            if (count > max_links(layer)) {
                fail(path, std::format("node {} has {} links on layer {} (max {})", node, count,
                                       layer, max_links(layer)));
            }
            seen.reset(n);
            for (const Id id : links(node, layer)) {
                if (id >= n) {
                    fail(path, std::format("node {} links to node {}, but there are only {} nodes",
                                           node, id, n));
                }
                if (id == node) {
                    fail(path, std::format("node {} links to itself on layer {}", node, layer));
                }
                if (levels_[id] < layer) {
                    fail(path,
                         std::format("node {} links to node {} on layer {}, above its level {}",
                                     node, id, layer, levels_[id]));
                }
                if (!seen.insert(id)) {
                    fail(path, std::format("node {} links to node {} twice on layer {}", node, id,
                                           layer));
                }
            }
        }
    }
}

}  // namespace vecdex
