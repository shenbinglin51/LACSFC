#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ac_brq_nd.hpp"
#include "ac_sfc_nd.hpp"

using namespace ac_sfc_nd;
using namespace ac_brq_nd;

struct BenchRow {
    std::string curve;
    int dim = 0;
    int grid = 0;
    std::string workload;
    size_t queries = 0;
    double avgVolume = 0.0;
    double avgSegments = 0.0;
    double indexRawKiB = 0.0;
    long long initUs = 0;
    long long brqUs = 0;
    long long hybridUs = 0;
    long long bruteUs = 0;
    uint64_t hybridThreshold = 0;
    double brqSpeedup = 0.0;
    double hybridSpeedup = 0.0;
    int mismatches = 0;
};

template<int Dim>
static uint64_t volumeOf(const QueryBoxND<Dim>& q) {
    uint64_t volume = 1;
    for (int i = 0; i < Dim; ++i) {
        volume *= static_cast<uint64_t>(q.high[i] - q.low[i] + 1);
    }
    return volume;
}

template<int Dim>
static QueryBoxND<Dim> fixedSizeBox(int gridSize, int side, std::mt19937& rng) {
    QueryBoxND<Dim> q;
    side = std::max(1, std::min(side, gridSize));
    for (int i = 0; i < Dim; ++i) {
        std::uniform_int_distribution<int> dist(0, gridSize - side);
        q.low[i] = dist(rng);
        q.high[i] = q.low[i] + side - 1;
    }
    return q;
}

template<int Dim>
static QueryBoxND<Dim> slabBox(int gridSize, int thickness, std::mt19937& rng) {
    QueryBoxND<Dim> q;
    for (int i = 0; i < Dim; ++i) {
        q.low[i] = 0;
        q.high[i] = gridSize - 1;
    }
    std::uniform_int_distribution<int> axisDist(0, Dim - 1);
    int axis = axisDist(rng);
    thickness = std::max(1, std::min(thickness, gridSize));
    std::uniform_int_distribution<int> dist(0, gridSize - thickness);
    q.low[axis] = dist(rng);
    q.high[axis] = q.low[axis] + thickness - 1;
    return q;
}

template<int Dim>
static QueryBoxND<Dim> randomBox(int gridSize, std::mt19937& rng) {
    QueryBoxND<Dim> q;
    std::uniform_int_distribution<int> dist(0, gridSize - 1);
    for (int i = 0; i < Dim; ++i) {
        int a = dist(rng);
        int b = dist(rng);
        q.low[i] = std::min(a, b);
        q.high[i] = std::max(a, b);
    }
    return q;
}

template<int Dim>
static std::vector<QueryBoxND<Dim>> makeWorkload(int gridSize,
                                                 const std::string& workload,
                                                 size_t queries) {
    std::vector<QueryBoxND<Dim>> boxes;
    boxes.reserve(queries);
    std::mt19937 rng(20260429 + Dim * 100 + gridSize * 17 + static_cast<int>(workload.size()));

    int smallSide = std::max(1, gridSize / 8);
    int mediumSide = std::max(1, gridSize / 4);
    int largeSide = std::max(1, gridSize / 2);
    int thinSide = std::max(1, gridSize / 16);

    for (size_t i = 0; i < queries; ++i) {
        if (workload == "small") boxes.push_back(fixedSizeBox<Dim>(gridSize, smallSide, rng));
        else if (workload == "medium") boxes.push_back(fixedSizeBox<Dim>(gridSize, mediumSide, rng));
        else if (workload == "large") boxes.push_back(fixedSizeBox<Dim>(gridSize, largeSide, rng));
        else if (workload == "slab") boxes.push_back(slabBox<Dim>(gridSize, thinSide, rng));
        else boxes.push_back(randomBox<Dim>(gridSize, rng));
    }
    return boxes;
}

template<int Dim>
static BenchRow runBenchmark(const ACSpaceFillingCurveND<Dim>& sfc,
                             const std::string& workload,
                             size_t queries,
                             uint64_t hybridThreshold) {
    BenchRow row;
    row.curve = sfc.name();
    row.dim = Dim;
    row.grid = sfc.gridSize();
    row.workload = workload;
    row.queries = queries;
    row.hybridThreshold = hybridThreshold;

    if (!verifyAC<Dim>(sfc) || !verifyBijection<Dim>(sfc)) {
        row.mismatches = static_cast<int>(queries);
        return row;
    }

    auto boxes = makeWorkload<Dim>(sfc.gridSize(), workload, queries);

    auto init0 = std::chrono::steady_clock::now();
    ACBrqProcessorND<Dim> processor(sfc);
    processor.initialize();
    auto init1 = std::chrono::steady_clock::now();

    row.indexRawKiB = static_cast<double>(processor.indexRawBits()) / 8.0 / 1024.0;
    row.initUs = std::chrono::duration_cast<std::chrono::microseconds>(init1 - init0).count();

    uint64_t volumeSum = 0;
    size_t segmentSum = 0;

    auto brq0 = std::chrono::steady_clock::now();
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> brqResults;
    brqResults.reserve(queries);
    for (const auto& q : boxes) {
        auto segments = processor.querySegments(q);
        segmentSum += segments.size();
        volumeSum += volumeOf<Dim>(q);
        brqResults.push_back(std::move(segments));
    }
    auto brq1 = std::chrono::steady_clock::now();

    auto hybrid0 = std::chrono::steady_clock::now();
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> hybridResults;
    hybridResults.reserve(queries);
    for (const auto& q : boxes) {
        hybridResults.push_back(processor.querySegmentsHybrid(q, hybridThreshold));
    }
    auto hybrid1 = std::chrono::steady_clock::now();

    auto brute0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < boxes.size(); ++i) {
        auto truth = bruteForceSegments<Dim>(sfc, boxes[i]);
        if (truth != brqResults[i]) ++row.mismatches;
        if (truth != hybridResults[i]) ++row.mismatches;
    }
    auto brute1 = std::chrono::steady_clock::now();

    row.brqUs = std::chrono::duration_cast<std::chrono::microseconds>(brq1 - brq0).count();
    row.hybridUs = std::chrono::duration_cast<std::chrono::microseconds>(hybrid1 - hybrid0).count();
    row.bruteUs = std::chrono::duration_cast<std::chrono::microseconds>(brute1 - brute0).count();
    row.avgVolume = static_cast<double>(volumeSum) / static_cast<double>(queries);
    row.avgSegments = static_cast<double>(segmentSum) / static_cast<double>(queries);
    row.brqSpeedup = row.brqUs > 0 ? static_cast<double>(row.bruteUs) / static_cast<double>(row.brqUs) : 0.0;
    row.hybridSpeedup = row.hybridUs > 0 ? static_cast<double>(row.bruteUs) / static_cast<double>(row.hybridUs) : 0.0;
    return row;
}

static void printHeader() {
    std::cout << std::left
              << std::setw(14) << "curve"
              << std::setw(5) << "dim"
              << std::setw(7) << "grid"
              << std::setw(10) << "workload"
              << std::setw(9) << "queries"
              << std::setw(12) << "avgVol"
              << std::setw(12) << "avgSeg"
              << std::setw(12) << "rawKiB"
              << std::setw(10) << "hyThres"
              << std::setw(12) << "initUs"
              << std::setw(12) << "brqUs"
              << std::setw(12) << "hybridUs"
              << std::setw(12) << "bruteUs"
              << std::setw(10) << "brqX"
              << std::setw(10) << "hybX"
              << std::setw(10) << "mismatch"
              << "\n";
}

static void printRow(const BenchRow& r) {
    std::cout << std::left
              << std::setw(14) << r.curve
              << std::setw(5) << r.dim
              << std::setw(7) << r.grid
              << std::setw(10) << r.workload
              << std::setw(9) << r.queries
              << std::setw(12) << std::fixed << std::setprecision(1) << r.avgVolume
              << std::setw(12) << std::fixed << std::setprecision(2) << r.avgSegments
              << std::setw(12) << std::fixed << std::setprecision(2) << r.indexRawKiB
              << std::setw(10) << r.hybridThreshold
              << std::setw(12) << r.initUs
              << std::setw(12) << r.brqUs
              << std::setw(12) << r.hybridUs
              << std::setw(12) << r.bruteUs
              << std::setw(10) << std::fixed << std::setprecision(2) << r.brqSpeedup
              << std::setw(10) << std::fixed << std::setprecision(2) << r.hybridSpeedup
              << std::setw(10) << r.mismatches
              << "\n";
}

template<int Dim>
static int runCurveSuite(const ACSpaceFillingCurveND<Dim>& sfc, size_t queries, uint64_t hybridThreshold) {
    std::vector<std::string> workloads = {"small", "medium", "large", "slab", "random"};
    int mismatches = 0;
    for (const auto& workload : workloads) {
        BenchRow row = runBenchmark<Dim>(sfc, workload, queries, hybridThreshold);
        mismatches += row.mismatches;
        printRow(row);
    }
    return mismatches;
}

int main() {
    printHeader();

    int mismatches = 0;

    SnakeCurveND<2> snake2(7);
    HilbertCurveND<2> hilbert2(7);
    SnakeCurveND<3> snake3(5);
    HilbertCurveND<3> hilbert3(5);
    SnakeCurveND<4> snake4(3);
    HilbertCurveND<4> hilbert4(3);

    mismatches += runCurveSuite<2>(snake2, 2000, 128);
    mismatches += runCurveSuite<2>(hilbert2, 2000, 128);
    mismatches += runCurveSuite<3>(snake3, 1000, 128);
    mismatches += runCurveSuite<3>(hilbert3, 1000, 128);
    mismatches += runCurveSuite<4>(snake4, 600, 128);
    mismatches += runCurveSuite<4>(hilbert4, 600, 128);

    std::cout << "\n=== AC-BRQ v2 hybrid benchmark mismatches = " << mismatches << " ===\n";
    return mismatches == 0 ? 0 : 1;
}
