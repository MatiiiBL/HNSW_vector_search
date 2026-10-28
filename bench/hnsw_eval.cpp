// Builds an approximate nearest-neighbor index on a TEXMEX dataset, then sweeps efSearch and
// prints one CSV row per ef: recall@k against the dataset's ground truth, and single-thread
// queries per second. Every library goes through the same loading, timing and recall code, so
// their rows are directly comparable.
//
//   vecdex_eval <dir> <name> [--lib vecdex] [--M 16] [--ef-construction 200] [--k 10]
//               [--ef 10,16,32,64,128,256,512] [--repeats 5]
//
// --lib is one of:
//   vecdex                 this library
//   hnswlib                hnswlib 0.8.0 as shipped
//   hnswlib-vecdex-kernel  hnswlib calling vecdex's distance kernel, which separates the graph
//                          code from the distance code in a comparison
//
// Reads <dir>/<name>_base.fvecs, <name>_query.fvecs and <name>_groundtruth.ivecs (for example
// data/sift sift). L2 only. Run a Release build. QPS is the median of `repeats` timed passes over
// all queries, after one untimed warmup pass.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "vecdex/distance.hpp"
#include "vecdex/hnsw.hpp"
#include "vecdex/io.hpp"
#include "vecdex/kernels.hpp"
#include "vecdex/recall.hpp"

#ifdef __APPLE__
#include <pthread.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using vecdex::Matrix;
using vecdex::Neighbor;

// ---------------------------------------------------------------------------------------------
// The indexes under test

// One index under test. search() isn't const because hnswlib keeps ef as index state.
class Engine {
public:
    Engine() = default;
    virtual ~Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    virtual void build(const Matrix<float>& base) = 0;
    // Up to k neighbors, closest first, with squared L2 distances.
    virtual std::vector<Neighbor> search(std::span<const float> query, std::size_t k,
                                         std::size_t ef) = 0;
    virtual std::size_t memory_bytes() const = 0;
    // One line about the built index, for stderr.
    virtual std::string describe() const = 0;
};

class VecdexEngine final : public Engine {
public:
    VecdexEngine(std::size_t dim, const vecdex::HnswParams& params)
        : index_(dim, vecdex::Metric::L2, params) {}

    void build(const Matrix<float>& base) override { index_.add(base); }

    std::vector<Neighbor> search(std::span<const float> query, std::size_t k,
                                 std::size_t ef) override {
        return index_.search(query, k, ef);
    }

    std::size_t memory_bytes() const override { return index_.memory_bytes(); }

    std::string describe() const override {
        const vecdex::GraphReport report = index_.inspect();
        return std::format(
            "vecdex: max level {}, mean layer-0 degree {:.2f}, reachable {}/{}, invariant "
            "violations {}",
            report.max_level, report.mean_degree0, report.reachable, index_.size(),
            report.violations.size());
    }

private:
    vecdex::HnswIndex index_;
};

// hnswlib calls distances through a plain function pointer plus an opaque parameter. This
// parameter carries vecdex's kernel and the dimension, so hnswlib pays one extra indirect call
// per distance compared with vecdex.
struct SharedKernel {
    vecdex::kernels::DistanceFn l2_squared;
    std::size_t dim;
};

float shared_kernel_l2(const void* a, const void* b, const void* param) {
    const auto& kernel = *static_cast<const SharedKernel*>(param);
    return kernel.l2_squared(static_cast<const float*>(a), static_cast<const float*>(b),
                             kernel.dim);
}

class SharedKernelSpace final : public hnswlib::SpaceInterface<float> {
public:
    explicit SharedKernelSpace(std::size_t dim)
        : param_{.l2_squared = vecdex::kernels::active().l2_squared, .dim = dim} {}

    std::size_t get_data_size() override { return param_.dim * sizeof(float); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return shared_kernel_l2; }
    void* get_dist_func_param() override { return &param_; }

private:
    SharedKernel param_;
};

class HnswlibEngine final : public Engine {
public:
    HnswlibEngine(std::size_t dim, const vecdex::HnswParams& params, bool vecdex_kernel)
        : params_(params), vecdex_kernel_(vecdex_kernel) {
        if (vecdex_kernel) {
            space_ = std::make_unique<SharedKernelSpace>(dim);
        } else {
            space_ = std::make_unique<hnswlib::L2Space>(dim);
        }
    }

    void build(const Matrix<float>& base) override {
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), base.rows(), params_.M, params_.ef_construction, params_.seed);
        for (std::size_t r = 0; r < base.rows(); ++r) {
            index_->addPoint(base.row(r).data(), r);  // label = row, so ids match the truth
        }
    }

    std::vector<Neighbor> search(std::span<const float> query, std::size_t k,
                                 std::size_t ef) override {
        index_->setEf(ef);
        auto queue = index_->searchKnn(query.data(), k);  // a max-heap: farthest on top
        std::vector<Neighbor> found(queue.size());
        for (std::size_t i = found.size(); i-- > 0;) {
            found[i] = {.distance = queue.top().first,
                        .id = static_cast<vecdex::Id>(queue.top().second)};
            queue.pop();
        }
        return found;
    }

    std::size_t memory_bytes() const override { return index_->indexFileSize(); }

    std::string describe() const override {
        return std::format("hnswlib 0.8.0 with {} distances",
                           vecdex_kernel_ ? "vecdex's kernel for" : "its own");
    }

private:
    vecdex::HnswParams params_;
    bool vecdex_kernel_;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;  // must outlive index_
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

// ---------------------------------------------------------------------------------------------
// Command line

struct Options {
    std::filesystem::path dir;
    std::string name;
    std::string lib = "vecdex";
    vecdex::HnswParams params;
    std::size_t k = 10;
    std::vector<std::size_t> efs{10, 16, 32, 64, 128, 256, 512};
    std::size_t repeats = 5;
};

constexpr std::string_view kUsage =
    "usage: vecdex_eval <dir> <name> [--lib vecdex|hnswlib|hnswlib-vecdex-kernel] [--M 16] "
    "[--ef-construction 200] [--k 10] [--ef 10,16,32,64,128,256,512] [--repeats 5]\n";

std::unique_ptr<Engine> make_engine(const Options& options, std::size_t dim) {
    if (options.lib == "vecdex") {
        return std::make_unique<VecdexEngine>(dim, options.params);
    }
    return std::make_unique<HnswlibEngine>(dim, options.params,
                                           options.lib == "hnswlib-vecdex-kernel");
}

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
    if (args.size() < 2 || args.size() % 2 != 0) {  // odd: a flag without a value
        return std::nullopt;
    }
    Options options;
    options.dir = args[0];
    options.name = args[1];
    for (std::size_t i = 2; i + 1 < args.size(); i += 2) {
        const std::string_view flag = args[i];
        const std::string_view value = args[i + 1];
        if (flag == "--lib") {
            if (value != "vecdex" && value != "hnswlib" && value != "hnswlib-vecdex-kernel") {
                return std::nullopt;
            }
            options.lib = value;
            continue;
        }
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
    return options;
}

// ---------------------------------------------------------------------------------------------
// The sweep

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// macOS places threads on performance or efficiency cores by their QoS class. The M4's
// efficiency cores are slower, so an unlucky placement would skew a whole sweep.
void prefer_performance_cores() {
#ifdef __APPLE__
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0) {
        std::cerr << "warning: could not raise the thread's QoS class\n";
    }
#endif
}

// The ground truth must cover every query with at least k ids, all naming base vectors.
void check_truth(const Matrix<std::int32_t>& truth, std::size_t queries, std::size_t base_rows,
                 std::size_t k) {
    if (truth.rows() < queries || truth.dim() < k) {
        throw std::runtime_error(std::format("ground truth is {} x {}, need at least {} x {}",
                                             truth.rows(), truth.dim(), queries, k));
    }
    for (std::size_t q = 0; q < queries; ++q) {
        for (const std::int32_t id : truth.row(q).first(k)) {
            if (id < 0 || std::cmp_greater_equal(id, base_rows)) {
                throw std::runtime_error(std::format("ground truth names missing vector {}", id));
            }
        }
    }
}

int run(int argc, char** argv) {
    const std::optional<Options> options = parse(argc, argv);
    if (!options) {
        std::cerr << kUsage;
        return 2;
    }
    prefer_performance_cores();
    const std::filesystem::path prefix = options->dir / options->name;
    const auto base = vecdex::read_fvecs(prefix.string() + "_base.fvecs");
    const auto queries = vecdex::read_fvecs(prefix.string() + "_query.fvecs");
    const auto truth = vecdex::read_ivecs(prefix.string() + "_groundtruth.ivecs");
    const std::size_t k = options->k;
    if (queries.dim() != base.dim()) {
        throw std::runtime_error("queries and base vectors have different dimensions");
    }
    check_truth(truth, queries.rows(), base.rows(), k);
    std::cerr << "loaded " << base.rows() << " x " << base.dim() << " base vectors, "
              << queries.rows() << " queries; vecdex kernels: "
              << vecdex::kernels::name(vecdex::kernels::active().isa) << "\n";

    // The k-th true neighbor's distance, for distance-threshold recall. SIFT's components are
    // small integers, so these squared distances are exact in float whatever the summation order.
    std::vector<float> kth_distance(queries.rows());
    for (std::size_t q = 0; q < queries.rows(); ++q) {
        const auto kth = static_cast<std::size_t>(truth.row(q)[k - 1]);
        kth_distance[q] = vecdex::l2_squared(queries.row(q), base.row(kth));
    }

    const std::unique_ptr<Engine> engine = make_engine(*options, base.dim());
    const Clock::time_point build_start = Clock::now();
    engine->build(base);
    const double build_seconds = seconds_since(build_start);
    std::cerr << "built in " << build_seconds << " s; " << engine->describe() << "\n";

    std::cout << "lib,dataset,n,dim,M,ef_construction,build_s,memory_mb,k,ef,recall,"
                 "recall_dist,qps\n";
    for (const std::size_t ef : options->efs) {
        std::vector<std::vector<Neighbor>> results(queries.rows());
        const auto run_all = [&] {
            for (std::size_t q = 0; q < queries.rows(); ++q) {
                results[q] = engine->search(queries.row(q), k, ef);
            }
        };
        run_all();  // warmup: caches, branch predictors, scratch-memory pools
        std::vector<double> times;
        for (std::size_t r = 0; r < options->repeats; ++r) {
            const Clock::time_point start = Clock::now();
            run_all();
            times.push_back(seconds_since(start));
        }
        std::ranges::sort(times);
        const double median = times[times.size() / 2];

        double recall = 0.0;
        double recall_dist = 0.0;
        for (std::size_t q = 0; q < queries.rows(); ++q) {
            recall += vecdex::recall_at_k(results[q], truth.row(q), k);
            recall_dist += vecdex::threshold_recall_at_k(results[q], kth_distance[q], k);
        }
        const auto n_queries = static_cast<double>(queries.rows());
        std::printf("%s,%s,%zu,%zu,%zu,%zu,%.1f,%.0f,%zu,%zu,%.4f,%.4f,%.0f\n",
                    options->lib.c_str(), options->name.c_str(), base.rows(), base.dim(),
                    options->params.M, options->params.ef_construction, build_seconds,
                    static_cast<double>(engine->memory_bytes()) / 1e6, k, ef, recall / n_queries,
                    recall_dist / n_queries, n_queries / median);
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
