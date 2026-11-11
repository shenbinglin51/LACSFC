#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "ac_brq_nd.hpp"
#include "ac_sfc_nd.hpp"

using namespace ac_sfc_nd;
using namespace ac_brq_nd;

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
static QueryBoxND<Dim> singlePointBox(const PointND<Dim>& p) {
    QueryBoxND<Dim> q;
    q.low = p;
    q.high = p;
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
static void addBoundaryBoxes(std::vector<QueryBoxND<Dim>>& boxes, int gridSize) {
    for (int axis = 0; axis < Dim; ++axis) {
        QueryBoxND<Dim> lower = fullBox<Dim>(gridSize);
        lower.high[axis] = 0;
        boxes.push_back(lower);

        QueryBoxND<Dim> upper = fullBox<Dim>(gridSize);
        upper.low[axis] = gridSize - 1;
        boxes.push_back(upper);

        QueryBoxND<Dim> lowerSlab = fullBox<Dim>(gridSize);
        lowerSlab.high[axis] = std::min(gridSize - 1, 1);
        boxes.push_back(lowerSlab);

        QueryBoxND<Dim> upperSlab = fullBox<Dim>(gridSize);
        upperSlab.low[axis] = std::max(0, gridSize - 2);
        boxes.push_back(upperSlab);
    }
}

template<int Dim>
static void addDegenerateBoxes(std::vector<QueryBoxND<Dim>>& boxes, int gridSize) {
    for (int freeAxis = 0; freeAxis < Dim; ++freeAxis) {
        QueryBoxND<Dim> q;
        for (int i = 0; i < Dim; ++i) {
            if (i == freeAxis) {
                q.low[i] = 0;
                q.high[i] = gridSize - 1;
            } else {
                q.low[i] = gridSize / 2;
                q.high[i] = gridSize / 2;
            }
        }
        boxes.push_back(q);
    }

    if constexpr (Dim >= 2) {
        for (int fixedAxis = 0; fixedAxis < Dim; ++fixedAxis) {
            QueryBoxND<Dim> q = fullBox<Dim>(gridSize);
            q.low[fixedAxis] = gridSize / 2;
            q.high[fixedAxis] = gridSize / 2;
            boxes.push_back(q);
        }
    }
}

template<int Dim>
static std::vector<QueryBoxND<Dim>> makeStressBoxes(const ACSpaceFillingCurveND<Dim>& sfc, int randomCount) {
    const int G = sfc.gridSize();
    std::vector<QueryBoxND<Dim>> boxes;
    boxes.push_back(fullBox<Dim>(G));
    boxes.push_back(singlePointBox<Dim>(sfc.decode(0)));
    boxes.push_back(singlePointBox<Dim>(sfc.decode(sfc.totalCodes() - 1)));
    boxes.push_back(endpointNeighborhood<Dim>(sfc.decode(0), G));
    boxes.push_back(endpointNeighborhood<Dim>(sfc.decode(sfc.totalCodes() - 1), G));
    addBoundaryBoxes<Dim>(boxes, G);
    addDegenerateBoxes<Dim>(boxes, G);

    std::mt19937 rng(20260429 + Dim * 1000 + G);
    for (int i = 0; i < randomCount; ++i) boxes.push_back(randomBox<Dim>(G, rng));
    return boxes;
}

template<int Dim>
static int runStress(const ACSpaceFillingCurveND<Dim>& sfc, int randomCount) {
    std::cout << "\n[stress] " << sfc.name() << " grid=" << sfc.gridSize() << "^" << Dim
              << " cells=" << sfc.totalCodes() << "\n";

    if (!verifyAC<Dim>(sfc) || !verifyBijection<Dim>(sfc)) {
        std::cout << "  verification failed before stress\n";
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    ACBrqProcessorND<Dim> processor(sfc);
    processor.initialize();
    auto t1 = std::chrono::steady_clock::now();
    const double rawKiB = static_cast<double>(processor.indexRawBits()) / 8.0 / 1024.0;

    const auto boxes = makeStressBoxes<Dim>(sfc, randomCount);
    int fails = 0;
    size_t segmentCount = 0;

    auto q0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto got = processor.querySegments(boxes[i]);
        const auto truth = bruteForceSegments<Dim>(sfc, boxes[i]);
        segmentCount += got.size();
        if (got != truth) {
            ++fails;
            if (fails <= 3) {
                std::cout << "  FAIL at box #" << i << " got=" << got.size()
                          << " truth=" << truth.size() << "\n";
            }
        }
    }
    auto q1 = std::chrono::steady_clock::now();

    const auto initMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    const auto queryMs = std::chrono::duration_cast<std::chrono::milliseconds>(q1 - q0).count();

    std::cout << "  boxes=" << boxes.size()
              << " segments=" << segmentCount
              << " index_bv=" << processor.indexBitvectorCount()
              << " raw_kib=" << rawKiB
              << " init_ms=" << initMs
              << " query_check_ms=" << queryMs
              << " fails=" << fails << "\n";
    return fails;
}

int main() {
    int fails = 0;

    SnakeCurveND<2> snake2(4);
    HilbertCurveND<2> hilbert2(4);
    SnakeCurveND<3> snake3(3);
    HilbertCurveND<3> hilbert3(3);
    SnakeCurveND<4> snake4(2);
    HilbertCurveND<4> hilbert4(2);
    SnakeCurveND<5> snake5(2);
    HilbertCurveND<5> hilbert5(2);

    fails += runStress<2>(snake2, 1000);
    fails += runStress<2>(hilbert2, 1000);
    fails += runStress<3>(snake3, 600);
    fails += runStress<3>(hilbert3, 600);
    fails += runStress<4>(snake4, 400);
    fails += runStress<4>(hilbert4, 400);
    fails += runStress<5>(snake5, 250);
    fails += runStress<5>(hilbert5, 250);

    std::cout << "\n=== AC-BRQ v2 stress summary: total fails = " << fails << " ===\n";
    return fails == 0 ? 0 : 1;
}
