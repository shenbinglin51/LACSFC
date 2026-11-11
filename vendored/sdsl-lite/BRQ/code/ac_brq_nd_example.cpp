#include <iostream>
#include <random>
#include <vector>

#include "ac_brq_nd.hpp"
#include "ac_sfc_nd.hpp"

using namespace ac_sfc_nd;
using namespace ac_brq_nd;

template<int Dim>
static void printPoint(const PointND<Dim>& p) {
    std::cout << "(";
    for (int i = 0; i < Dim; ++i) std::cout << p[i] << (i + 1 < Dim ? "," : "");
    std::cout << ")";
}

template<int Dim>
static QueryBoxND<Dim> singlePointBox(const PointND<Dim>& p) {
    QueryBoxND<Dim> q;
    q.low = p;
    q.high = p;
    return q;
}

template<int Dim>
static QueryBoxND<Dim> fullBox(int gridSize) {
    QueryBoxND<Dim> q;
    for (int i = 0; i < Dim; ++i) {
        q.low[i] = 0;
        q.high[i] = gridSize - 1;
    }
    return q;
}

template<int Dim>
static QueryBoxND<Dim> endpointNeighborhood(const PointND<Dim>& p, int gridSize) {
    QueryBoxND<Dim> q;
    for (int i = 0; i < Dim; ++i) {
        q.low[i] = std::max(0, p[i] - 1);
        q.high[i] = std::min(gridSize - 1, p[i] + 1);
    }
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
static std::vector<QueryBoxND<Dim>> makeBoxes(const ACSpaceFillingCurveND<Dim>& sfc) {
    const int G = sfc.gridSize();
    std::vector<QueryBoxND<Dim>> boxes;
    boxes.push_back(fullBox<Dim>(G));
    boxes.push_back(singlePointBox<Dim>(sfc.decode(0)));
    boxes.push_back(singlePointBox<Dim>(sfc.decode(sfc.totalCodes() - 1)));
    boxes.push_back(endpointNeighborhood<Dim>(sfc.decode(0), G));
    boxes.push_back(endpointNeighborhood<Dim>(sfc.decode(sfc.totalCodes() - 1), G));

    QueryBoxND<Dim> lowerSlab = fullBox<Dim>(G);
    lowerSlab.high[0] = std::min(G - 1, 1);
    boxes.push_back(lowerSlab);

    QueryBoxND<Dim> upperSlab = fullBox<Dim>(G);
    upperSlab.low[0] = std::max(0, G - 2);
    boxes.push_back(upperSlab);

    QueryBoxND<Dim> center;
    for (int i = 0; i < Dim; ++i) {
        center.low[i] = G / 4;
        center.high[i] = std::max(center.low[i], (3 * G) / 4 - 1);
    }
    boxes.push_back(center);

    std::mt19937 rng(12345 + Dim * 100 + G);
    for (int i = 0; i < 24; ++i) boxes.push_back(randomBox<Dim>(G, rng));
    return boxes;
}

template<int Dim>
static int runCurve(const ACSpaceFillingCurveND<Dim>& sfc) {
    std::cout << "\n========================================\n";
    std::cout << "SFC = " << sfc.name() << ", grid = " << sfc.gridSize() << "^" << Dim
              << " (= " << sfc.totalCodes() << " cells)\n";
    std::cout << "========================================\n";

    const bool acOK = verifyAC<Dim>(sfc);
    const bool bjOK = verifyBijection<Dim>(sfc);
    std::cout << "verifyAC       : " << (acOK ? "PASS" : "FAIL") << "\n";
    std::cout << "verifyBijection: " << (bjOK ? "PASS" : "FAIL") << "\n";
    if (!acOK || !bjOK) return 1;

    ACBrqProcessorND<Dim> processor(sfc);
    processor.initialize();

    int fails = 0;
    const auto boxes = makeBoxes<Dim>(sfc);
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& q = boxes[i];
        auto got = processor.querySegments(q);
        auto truth = bruteForceSegments<Dim>(sfc, q);
        const bool ok = got == truth;

        std::cout << "Query #" << i << " low=";
        printPoint<Dim>(q.low);
        std::cout << " high=";
        printPoint<Dim>(q.high);
        std::cout << " -> " << (ok ? "PASS" : "FAIL") << "\n";

        if (!ok) {
            ++fails;
            std::cout << "  Got  : ";
            for (const auto& s : got) std::cout << "[" << s.first << "," << s.second << "] ";
            std::cout << "\n  Truth: ";
            for (const auto& s : truth) std::cout << "[" << s.first << "," << s.second << "] ";
            std::cout << "\n";
        }
    }

    std::cout << "Fails: " << fails << "\n";
    return fails;
}

int main() {
    int fails = 0;

    SnakeCurveND<2> snake2(3);
    HilbertCurveND<2> hilbert2(3);
    SnakeCurveND<3> snake3(2);
    HilbertCurveND<3> hilbert3(2);
    SnakeCurveND<4> snake4(2);
    HilbertCurveND<4> hilbert4(2);

    fails += runCurve<2>(snake2);
    fails += runCurve<2>(hilbert2);
    fails += runCurve<3>(snake3);
    fails += runCurve<3>(hilbert3);
    fails += runCurve<4>(snake4);
    fails += runCurve<4>(hilbert4);

    std::cout << "\n=== AC-BRQ v2 summary: total fails = " << fails << " ===\n";
    return fails == 0 ? 0 : 1;
}
