# vecdex HNSW index file format, version 1

`HnswIndex::save` writes this format and `HnswIndex::load` reads it ([src/hnsw_io.cpp](../src/hnsw_io.cpp)). This document is the specification: a reader written from it alone must accept every file `save` writes and reject what `load` rejects.

## Conventions

- Every integer is unsigned and **little-endian**: `u8`, `u32`, `u64`.
- Every float is an IEEE 754 binary32 (`f32`), little-endian.
- Fields and sections follow one another with **no padding and no alignment**. The format is meant for streaming into memory the index owns, not for `mmap`.
- `n` is the node count, `dim` the dimension, `M` the link parameter, and `U` the number of upper-layer link blocks, all from the header.
- A node's id is its position: 0 to n − 1, in insertion order.

## Layout

```
offset 0                          header                     80 bytes
offset 80                         levels                     n × u8
offset 80 + n                     layer-0 link blocks        n × (1 + 2M) × u32
                                  upper-layer link blocks    U × (1 + M) × u32
                                  vectors                    n × dim × f32
offset size − 8                   checksum                   u64
```

The file size is exactly

```
88 + n + 4·n·(1 + 2M) + 4·U·(1 + M) + 4·n·dim
```

bytes. For SIFT1M (n = 10⁶, dim = 128, M = 16, U ≈ n / (M − 1)) that is about 650 MB. Nearly 80% of it is the vectors.

## Header

| Offset | Type | Field | Valid values |
|---:|---|---|---|
| 0 | 8 bytes | magic | `89 56 44 58 48 4E 53 57` (`\x89VDXHNSW`) |
| 8 | u32 | format version | `1` |
| 12 | u32 | metric | `0` = L2 (squared Euclidean), `1` = InnerProduct (1 − ⟨a, b⟩), `2` = Cosine |
| 16 | u64 | dim | ≥ 1 |
| 24 | u64 | M | 2 to 2³¹ − 1 (a link count is a u32 and can reach 2M) |
| 32 | u64 | ef_construction | ≥ 1 |
| 40 | u64 | seed | any |
| 48 | u64 | n, the node count | ≤ 2³² − 1 (ids are u32) |
| 56 | u64 | max level | ≤ 255; 0 when n = 0 |
| 64 | u64 | entry point | < n; 0 when n = 0 |
| 72 | u64 | U, the upper-layer block count | the sum of all levels; 0 when n = 0 |

The magic's first byte is not ASCII (the same trick PNG uses), so no text file can pass for an index.

The metric codes are part of the format. They are not the values of the C++ `Metric` enum, which the code maps explicitly.

### Example

The header of a 12-node index (dim 2, M = 2, ef_construction 8, seed 42, L2), the one the corruption tests use:

```
00000000: 8956 4458 484e 5357 0100 0000 0000 0000  .VDXHNSW........
00000010: 0200 0000 0000 0000 0200 0000 0000 0000  ................
00000020: 0800 0000 0000 0000 2a00 0000 0000 0000  ........*.......
00000030: 0c00 0000 0000 0000 0300 0000 0000 0000  ................
00000040: 0400 0000 0000 0000 0a00 0000 0000 0000  ................
```

That is magic, version 1, metric 0, dim 2, M 2, ef_construction 8, seed 42, n = 12, max level 3, entry point 4, U = 10. The file is 88 + 12 + 4·12·5 + 4·10·3 + 4·12·2 = 556 bytes. In Python:

```python
import struct
magic, version, metric, dim, M, ef_construction, seed, n, max_level, entry_point, U = \
    struct.unpack("<8sII8Q", open("index.vdx", "rb").read(80))
```

## Sections

### Levels

`n` bytes. Byte `i` is node `i`'s level, the highest layer it lives on. Each level is ≤ the max level, and the entry point's level equals it.

### Layer-0 link blocks

One block per node, in id order. Each block is `1 + 2M` u32 values:

```
count, id_1, id_2, ..., id_2M
```

Node `i`'s block starts at byte `80 + n + 4·i·(1 + 2M)`. Its first `count` ids are its neighbors on layer 0. The `2M − count` slots after them have unspecified contents: `save` writes them as they are in memory, and `load` never reads them as links. The checksum still covers them.

### Upper-layer link blocks

For every node in id order, one block for each of its layers 1 to level. A node at level 0 has none. Each block is `1 + M` u32 values, laid out like a layer-0 block: `count`, then `M` id slots.

Node `i`'s block for layer `l` is block number `(level_0 + … + level_{i−1}) + (l − 1)` of this section.

### Vectors

`n × dim` floats, row-major: node `i`'s vector is floats `i·dim` to `(i + 1)·dim − 1`. These are the vectors as the index stores them. Under Cosine they are the unit-length vectors `add` computed, not the caller's originals.

### Checksum

XXH64 (xxHash, 64-bit, seed 0) of every byte before it, `[0, size − 8)`, stored as a little-endian u64. The algorithm is specified in the xxHash repository ([doc/xxhash_spec.md](https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md)). Note that `xxhsum -H1` prints the canonical big-endian hex, so its output reads byte-reversed compared with the file.

The checksum detects accidental damage: truncated copies, flipped bits, a file overwritten partway. It is not a defense against a crafted file, since anyone can recompute it. The validation below is that defense.

## What load checks

`load` treats the file as hostile. It checks the following, in this order, and stops at the first failure with a `std::runtime_error` whose message starts with the path:

1. The file exists and is a regular file of at least 88 bytes.
2. The magic matches and the version is 1.
3. Every header field is in the range the table gives.
4. The file size equals the formula above exactly. The arithmetic is overflow-checked, so wrapped-around products can't pass for small ones. **Nothing is allocated before this check**, and after it every buffer is bounded by the file's size.
5. The checksum matches.
6. Every level is ≤ the max level, the levels add up to U, and the entry point's level is the max level.
7. Every vector component is finite. NaN would break the strict weak ordering the search heaps and sorts depend on.
8. In every link block, before any link is followed: count ≤ the block's capacity (2M on layer 0, M above); every id is < n, is not the block's own node, and names a node that lives on that layer (level ≥ the layer); no id appears twice in one block.

A loaded index therefore satisfies every invariant `HnswIndex::inspect()` reports on.

Not checked: whether the graph is good for search (its connectivity, which `inspect()` measures but doesn't require), whether Cosine vectors have unit length, and whether `ef_construction` and `seed` are the values a caller would choose. None of these can make search or insertion unsafe.

## State that is not in the file

- **The level generator.** The index draws each node's level from `std::mt19937_64(seed)`, once per node, in id order. `load` re-seeds the generator and discards `n` draws, which recreates the saved index's generator state exactly. So `add` after `load` builds the same graph as `add` on the index that was saved.
- **The distance kernel.** It is chosen at run time from the CPU (scalar, NEON or AVX2). A file loads on any little-endian machine. With the same kernel, search results are identical to the saved index's. A different kernel sums in a different order, so distances may differ in their last bits.
- **Search scratch memory.** It is recreated empty.

## Saving

`save` writes to a fresh temporary file next to the destination, `<path>.tmp-<16 random hex digits>`, then closes it and renames it over `<path>`. A rename within one file system is atomic, so a reader sees either the old file or the new one, never a partial one. If anything fails, the temporary file is removed and `<path>` is untouched.

The replacement is also durable. Before the rename, `save` flushes the temporary file to stable storage; after it, it flushes the directory, so the rename itself survives a power loss. Without the first flush, a file system could persist the rename before the data, and a power loss would replace the old index with one that fails its checksum. On macOS the flushes use `fcntl(F_FULLFSYNC)`, because `fsync` there leaves the data in the drive's volatile cache; on file systems that don't support `F_FULLFSYNC` (some network file systems), `save` falls back to `fsync`. On Linux they use `fsync`. Once `save` returns, the new index survives a crash or a power loss, as far as the drive honors flush requests.

If only the directory flush fails, `save` throws although `<path>` already holds the new, complete index: the old file is gone, so the rename isn't undone, but it may not survive a power loss. A process killed partway through `save` can leave its temporary file behind.

## Versioning

Any change to the layout or to the meaning of a field gets a new version number. `load` rejects every version but 1 with a message naming both versions. A future reader can accept older versions by dispatching on the version field, which stays at offset 8 for that reason.
