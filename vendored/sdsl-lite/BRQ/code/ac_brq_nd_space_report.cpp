#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct SpaceRow {
    int dim = 0;
    int grid = 0;
    long double cells = 0.0L;
    long double bitvectors = 0.0L;
    long double planeSize = 0.0L;
    long double rawBits = 0.0L;
    long double bitsPerCell = 0.0L;
    long double rawMiB = 0.0L;
    long double rankEstMiB = 0.0L;
    long double selectEstMiB = 0.0L;
    long double totalEstMiB = 0.0L;
};

static long double powInt(long double base, int exp) {
    long double result = 1.0L;
    for (int i = 0; i < exp; ++i) result *= base;
    return result;
}

static SpaceRow makeRow(int dim, int grid) {
    SpaceRow row;
    row.dim = dim;
    row.grid = grid;
    row.cells = powInt(static_cast<long double>(grid), dim);
    row.planeSize = powInt(static_cast<long double>(grid), dim - 1);
    row.bitvectors = static_cast<long double>(2 * dim * dim * grid);
    row.rawBits = row.bitvectors * row.planeSize;
    row.bitsPerCell = row.rawBits / row.cells;
    row.rawMiB = row.rawBits / 8.0L / 1024.0L / 1024.0L;

    row.rankEstMiB = row.rawMiB * 0.25L;
    row.selectEstMiB = row.rawMiB * 0.50L;
    row.totalEstMiB = row.rawMiB + row.rankEstMiB + row.selectEstMiB;
    return row;
}

static std::string compact(long double value) {
    const long double oneK = 1000.0L;
    const long double oneM = oneK * oneK;
    const long double oneB = oneM * oneK;
    const long double oneT = oneB * oneK;

    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (value >= oneT) out << static_cast<double>(value / oneT) << "T";
    else if (value >= oneB) out << static_cast<double>(value / oneB) << "B";
    else if (value >= oneM) out << static_cast<double>(value / oneM) << "M";
    else if (value >= oneK) out << static_cast<double>(value / oneK) << "K";
    else out << static_cast<double>(value);
    return out.str();
}

static void printHeader() {
    std::cout << std::left
              << std::setw(5) << "dim"
              << std::setw(8) << "grid"
              << std::setw(12) << "cells"
              << std::setw(12) << "bv_count"
              << std::setw(12) << "plane"
              << std::setw(12) << "bpc"
              << std::setw(12) << "rawMiB"
              << std::setw(13) << "rankMiB*"
              << std::setw(14) << "selectMiB*"
              << std::setw(14) << "totalMiB*"
              << "\n";
}

static void printRow(const SpaceRow& row) {
    std::cout << std::left
              << std::setw(5) << row.dim
              << std::setw(8) << row.grid
              << std::setw(12) << compact(row.cells)
              << std::setw(12) << compact(row.bitvectors)
              << std::setw(12) << compact(row.planeSize)
              << std::setw(12) << std::fixed << std::setprecision(1) << static_cast<double>(row.bitsPerCell)
              << std::setw(12) << std::fixed << std::setprecision(3) << static_cast<double>(row.rawMiB)
              << std::setw(13) << std::fixed << std::setprecision(3) << static_cast<double>(row.rankEstMiB)
              << std::setw(14) << std::fixed << std::setprecision(3) << static_cast<double>(row.selectEstMiB)
              << std::setw(14) << std::fixed << std::setprecision(3) << static_cast<double>(row.totalEstMiB)
              << "\n";
}

static void printGroup(const std::string& title, const std::vector<std::pair<int, int>>& cases) {
    std::cout << "\n## " << title << "\n";
    printHeader();
    for (const auto& c : cases) printRow(makeRow(c.first, c.second));
}

int main() {
    std::cout << "AC-BRQ v2 hyperplane-local index space report\n";
    std::cout << "Exact raw formula: raw_bits = 2 * d^2 * M, where M = grid^d\n";
    std::cout << "Exact bitvectors: 2 * d * d * grid, each of length grid^(d-1)\n";
    std::cout << "Columns marked with * are rough support-structure estimates, not measured SDSL allocations.\n";
    std::cout << "Estimate used here: rank ~= 25% of raw bits, select ~= 50% of raw bits.\n";

    printGroup("Benchmark-size configurations", {
        {2, 128},
        {3, 32},
        {4, 8},
        {5, 4}
    });

    printGroup("Same-cell-count comparison around one million cells", {
        {2, 1024},
        {3, 100},
        {4, 32},
        {5, 16},
        {6, 10}
    });

    printGroup("Scaling examples", {
        {2, 256},
        {2, 512},
        {2, 1024},
        {3, 64},
        {3, 128},
        {4, 16},
        {4, 32},
        {5, 8},
        {5, 16},
        {6, 8}
    });

    std::cout << "\nKey implication: raw bits per cell are exactly 2*d^2.\n";
    std::cout << "For d=2,3,4,5,6 this is 8,18,32,50,72 raw bits/cell before rank/select support.\n";
    return 0;
}
