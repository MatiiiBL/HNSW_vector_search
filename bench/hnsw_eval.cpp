// Builds an HNSW index on a TEXMEX dataset, then sweeps efSearch and prints one CSV row per
// ef: recall@k against the dataset's ground truth, and single-thread queries per second.
//
//   vecdex_eval <dir> <name> [--M 16] [--ef-construction 200] [--k 10]
//               [--ef 10,16,32,64,128,256,512] [--repeats 5]
//
// Reads <dir>/<name>_base.fvecs, <name>_query.fvecs and <name>_groundtruth.ivecs (for example
// data/sift sift). Run a Release build. QPS is the median of `repeats` timed passes over all
// queries, after one untimed warmup pass.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vecdex/hnsw.hpp"
#include "vecdex/io.hpp"
#include "vecdex/kernels.hpp"
#include "vecdex/recall.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path dir;
    std::string name;
    vecdex::HnswParams params;
    std::size_t k = 10;
    std::vector<std::size_t> efs{10, 16, 32, 64, 128, 256, 512};
    std::size_t repeats = 5;
};

std::optional<std::size_t> parse_size(std::string_view text) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::vector<std::size_t>> parse_list(std::string_view text) {
    std::vector<std::size_t> values;
    while (!text.empty()) {
        const std::size_t comma = std::min(text.find(','), text.size());
        const std::optional<std::size_t> value = parse_size(text.substr(0, comma));
        if (!value) {
            return std::nullopt;
        }
        values.push_back(*value);
        text.remove_prefix(std::min(comma + 1, text.size()));
    }
    return values;
}

std::optional<Options> parse(int argc, char** argv) {
    const std::vector<std::string_view> args(argv + 1, argv + argc);
    if (args.size() < 2) {
        return std::nullopt;
    }
    Options options;
    options.dir = args[0];
    options.name = args[1];
    for (std::size_t i = 2; i + 1 < args.size(); i += 2) {
        const std::string_view flag = args[i];
        const std::string_view value = args[i + 1];
        if (flag == "--ef") {
            const auto list = parse_list(value);
            if (!list || list->empty()) {
                return std::nullopt;
            }
            options.efs = *list;
            continue;
        }
        std::size_t* target = nullptr;
        if (flag == "--M") {
            target = &options.params.M;
        } else if (flag == "--ef-construction") {
            target = &options.params.ef_construction;
        } else if (flag == "--k") {
            target = &options.k;
        } else if (flag == "--repeats") {
            target = &options.repeats;
        }
        const std::optional<std::size_t> number = parse_size(value);
        if (target == nullptr || !number || *number == 0) {
            return std::nullopt;
        }
        *target = *number;
    }
    if (args.size() % 2 != 0) {  // a flag without a value
        return std::nullopt;
    }
    return options;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

}  // namespace

namespace {

int run(int argc, char** argv) {
    const std::optional<Options> options = parse(argc, argv);
    if (!options) {
        std::cerr << "usage: vecdex_eval <dir> <name> [--M 16] [--ef-construction 200] "
                     "[--k 10] [--ef 10,16,32,64,128,256,512] [--repeats 5]\n";
        return 2;
    }
    const std::filesystem::path prefix = options->dir / options->name;
    const auto base = vecdex::read_fvecs(prefix.string() + "_base.fvecs");
    const auto queries = vecdex::read_fvecs(prefix.string() + "_query.fvecs");
    const auto truth = vecdex::read_ivecs(prefix.string() + "_groundtruth.ivecs");
    std::cerr << "loaded " << base.rows() << " x " << base.dim() << " base vectors, "
              << queries.rows()
              << " queries; kernels: " << vecdex::kernels::name(vecdex::kernels::active().isa)
              << "\n";

    const Clock::time_point build_start = Clock::now();
    vecdex::HnswIndex index(base.dim(), vecdex::Metric::L2, options->params);
    index.add(base);
    const double build_seconds = seconds_since(build_start);
    const vecdex::GraphReport report = index.inspect();
    std::cerr << "built in " << build_seconds << " s; max level " << report.max_level
              << ", mean layer-0 degree " << report.mean_degree0 << ", reachable "
              << report.reachable << "/" << index.size() << ", invariant violations "
              << report.violations.size() << "\n";

    std::cout << "dataset,n,dim,M,ef_construction,build_s,memory_mb,k,ef,recall,qps\n";
    const std::size_t k = options->k;
    for (const std::size_t ef : options->efs) {
        std::vector<std::vector<vecdex::Neighbor>> results(queries.rows());
        const auto run_all = [&] {
            for (std::size_t q = 0; q < queries.rows(); ++q) {
                results[q] = index.search(queries.row(q), k, ef);
            }
        };
        run_all();  // warmup: caches, branch predictors, the visited-list pool
        std::vector<double> times;
        for (std::size_t r = 0; r < options->repeats; ++r) {
            const Clock::time_point start = Clock::now();
            run_all();
            times.push_back(seconds_since(start));
        }
        std::ranges::sort(times);
        const double median = times[times.size() / 2];

        double recall = 0.0;
        for (std::size_t q = 0; q < queries.rows(); ++q) {
            recall += vecdex::recall_at_k(results[q], truth.row(q), k);
        }
        recall /= static_cast<double>(queries.rows());
        std::printf("%s,%zu,%zu,%zu,%zu,%.1f,%.0f,%zu,%zu,%.4f,%.0f\n", options->name.c_str(),
                    index.size(), index.dim(), options->params.M, options->params.ef_construction,
                    build_seconds, static_cast<double>(index.memory_bytes()) / 1e6, k, ef, recall,
                    static_cast<double>(queries.rows()) / median);
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {  // unreadable dataset, bad parameters, ...
        std::cerr << "vecdex_eval: " << e.what() << "\n";
        return 1;
    }
}
