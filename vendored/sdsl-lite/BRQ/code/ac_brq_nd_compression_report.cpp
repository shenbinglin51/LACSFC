#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <sdsl/int_vector.hpp>
#include <sdsl/io.hpp>
#include <sdsl/rank_support_v.hpp>
#include <sdsl/sd_vector.hpp>
#include <sdsl/select_support_mcl.hpp>

struct VariantRow {
    std::string variant;
    uint64_t length = 0;
    uint64_t ones = 0;
    double density = 0.0;
    uint64_t bytes = 0;
    double bitsPerOne = 0.0;
    double bitsPerBit = 0.0;
    long long buildUs = 0;
    long long rankUs = 0;
    long long selectUs = 0;
    uint64_t checksum = 0;
};

static sdsl::bit_vector makeVector(uint64_t length, double density) {
    sdsl::bit_vector bv(length, 0);
    uint64_t ones = static_cast<uint64_t>(static_cast<long double>(length) * density);
    ones = std::max<uint64_t>(1, std::min<uint64_t>(ones, length));

    std::vector<uint64_t> positions(length);
    std::iota(positions.begin(), positions.end(), 0);
    std::mt19937_64 rng(20260429ULL + length + static_cast<uint64_t>(density * 1000000.0));
    std::shuffle(positions.begin(), positions.end(), rng);
    positions.resize(ones);
    std::sort(positions.begin(), positions.end());

    for (uint64_t p : positions) bv[p] = 1;
    return bv;
}

static std::vector<uint64_t> makeRankQueries(uint64_t length, size_t count) {
    std::vector<uint64_t> qs;
    qs.reserve(count);
    std::mt19937_64 rng(777 + length + count);
    std::uniform_int_distribution<uint64_t> dist(0, length);
    for (size_t i = 0; i < count; ++i) qs.push_back(dist(rng));
    return qs;
}

static std::vector<uint64_t> makeSelectQueries(uint64_t ones, size_t count) {
    std::vector<uint64_t> qs;
    qs.reserve(count);
    std::mt19937_64 rng(999 + ones + count);
    std::uniform_int_distribution<uint64_t> dist(1, ones);
    for (size_t i = 0; i < count; ++i) qs.push_back(dist(rng));
    return qs;
}

static VariantRow benchPlain(const sdsl::bit_vector& bv,
                             const std::vector<uint64_t>& rankQueries,
                             const std::vector<uint64_t>& selectQueries) {
    VariantRow row;
    row.variant = "plain";
    row.length = bv.size();
    row.ones = sdsl::util::cnt_one_bits(bv);
    row.density = static_cast<double>(row.ones) / static_cast<double>(row.length);

    auto build0 = std::chrono::steady_clock::now();
    sdsl::rank_support_v<1> rank(&bv);
    sdsl::select_support_mcl<1> select(&bv);
    auto build1 = std::chrono::steady_clock::now();

    row.bytes = sdsl::size_in_bytes(bv) + sdsl::size_in_bytes(rank) + sdsl::size_in_bytes(select);
    row.bitsPerOne = row.ones > 0 ? static_cast<double>(row.bytes * 8) / static_cast<double>(row.ones) : 0.0;
    row.bitsPerBit = static_cast<double>(row.bytes * 8) / static_cast<double>(row.length);
    row.buildUs = std::chrono::duration_cast<std::chrono::microseconds>(build1 - build0).count();

    auto rank0 = std::chrono::steady_clock::now();
    for (uint64_t q : rankQueries) row.checksum += rank.rank(q);
    auto rank1 = std::chrono::steady_clock::now();

    auto select0 = std::chrono::steady_clock::now();
    for (uint64_t q : selectQueries) row.checksum += select.select(q);
    auto select1 = std::chrono::steady_clock::now();

    row.rankUs = std::chrono::duration_cast<std::chrono::microseconds>(rank1 - rank0).count();
    row.selectUs = std::chrono::duration_cast<std::chrono::microseconds>(select1 - select0).count();
    return row;
}

static VariantRow benchSd(const sdsl::bit_vector& bv,
                          const std::vector<uint64_t>& rankQueries,
                          const std::vector<uint64_t>& selectQueries) {
    VariantRow row;
    row.variant = "sd_vector";
    row.length = bv.size();
    row.ones = sdsl::util::cnt_one_bits(bv);
    row.density = static_cast<double>(row.ones) / static_cast<double>(row.length);

    auto build0 = std::chrono::steady_clock::now();
    sdsl::sd_vector<> sv(bv);
    sdsl::sd_vector<>::rank_1_type rank(&sv);
    sdsl::sd_vector<>::select_1_type select(&sv);
    auto build1 = std::chrono::steady_clock::now();

    row.bytes = sdsl::size_in_bytes(sv) + sdsl::size_in_bytes(rank) + sdsl::size_in_bytes(select);
    row.bitsPerOne = row.ones > 0 ? static_cast<double>(row.bytes * 8) / static_cast<double>(row.ones) : 0.0;
    row.bitsPerBit = static_cast<double>(row.bytes * 8) / static_cast<double>(row.length);
    row.buildUs = std::chrono::duration_cast<std::chrono::microseconds>(build1 - build0).count();

    auto rank0 = std::chrono::steady_clock::now();
    for (uint64_t q : rankQueries) row.checksum += rank.rank(q);
    auto rank1 = std::chrono::steady_clock::now();

    auto select0 = std::chrono::steady_clock::now();
    for (uint64_t q : selectQueries) row.checksum += select.select(q);
    auto select1 = std::chrono::steady_clock::now();

    row.rankUs = std::chrono::duration_cast<std::chrono::microseconds>(rank1 - rank0).count();
    row.selectUs = std::chrono::duration_cast<std::chrono::microseconds>(select1 - select0).count();
    return row;
}

static void printVariantHeader() {
    std::cout << std::left
              << std::setw(11) << "variant"
              << std::setw(10) << "len"
              << std::setw(9) << "ones"
              << std::setw(10) << "density"
              << std::setw(10) << "KiB"
              << std::setw(9) << "b/bit"
              << std::setw(10) << "b/one"
              << std::setw(10) << "buildUs"
              << std::setw(10) << "rankUs"
              << std::setw(10) << "selUs"
              << std::setw(12) << "checksum"
              << "\n";
}

static void printVariantRow(const VariantRow& r) {
    std::cout << std::left
              << std::setw(11) << r.variant
              << std::setw(10) << r.length
              << std::setw(9) << r.ones
              << std::setw(10) << std::fixed << std::setprecision(3) << r.density
              << std::setw(10) << std::fixed << std::setprecision(2) << (static_cast<double>(r.bytes) / 1024.0)
              << std::setw(9) << std::fixed << std::setprecision(2) << r.bitsPerBit
              << std::setw(10) << std::fixed << std::setprecision(2) << r.bitsPerOne
              << std::setw(10) << r.buildUs
              << std::setw(10) << r.rankUs
              << std::setw(10) << r.selectUs
              << std::setw(12) << r.checksum
              << "\n";
}

static void printFallbackComparison() {
    std::cout << "\n## Low-space face enumeration fallback theory\n";
    std::cout << std::left
              << std::setw(5) << "dim"
              << std::setw(14) << "hyper_bpc"
              << std::setw(14) << "fallback_bpc"
              << std::setw(12) << "saving"
              << "\n";
    for (int d = 2; d <= 8; ++d) {
        double hyper = 2.0 * d * d;
        double fallback = 2.0 * d;
        double saving = hyper / fallback;
        std::cout << std::left
                  << std::setw(5) << d
                  << std::setw(14) << std::fixed << std::setprecision(1) << hyper
                  << std::setw(14) << std::fixed << std::setprecision(1) << fallback
                  << std::setw(12) << std::fixed << std::setprecision(1) << saving
                  << "\n";
    }
}

int main() {
    const uint64_t length = 1ULL << 20;
    const size_t queryCount = 200000;
    const std::vector<double> densities = {0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50};

    std::cout << "AC-BRQ v2 bitvector compression report\n";
    std::cout << "length=" << length << " rank/select_queries=" << queryCount << "\n";
    std::cout << "rrr_vector<127> is used instead of rrr_vector<15> because the local rrr<15> specialization appears modified.\n";

    for (double density : densities) {
        std::cout << "\n## density " << density << "\n";
        sdsl::bit_vector bv = makeVector(length, density);
        uint64_t ones = sdsl::util::cnt_one_bits(bv);
        auto rankQueries = makeRankQueries(length, queryCount);
        auto selectQueries = makeSelectQueries(ones, queryCount);

        printVariantHeader();
        printVariantRow(benchPlain(bv, rankQueries, selectQueries));
        printVariantRow(benchSd(bv, rankQueries, selectQueries));
        std::cout << "rrr127     unavailable in current local SDSL tree; rrr_vector.hpp includes modified/broken rrr_vector_15.hpp\n";
    }

    printFallbackComparison();
    return 0;
}
