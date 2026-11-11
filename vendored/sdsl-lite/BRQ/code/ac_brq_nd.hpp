#ifndef AC_BRQ_ND_HPP
#define AC_BRQ_ND_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <sdsl/int_vector.hpp>
#include <sdsl/rank_support_v.hpp>
#include <sdsl/select_support_mcl.hpp>

#include "ac_sfc_nd.hpp"

namespace ac_brq_nd {

using ac_sfc_nd::ACSpaceFillingCurveND;
using ac_sfc_nd::PointND;
using ac_sfc_nd::directionId;
using ac_sfc_nd::directionFromDelta;

template<int Dim>
struct QueryBoxND {
    PointND<Dim> low{};
    PointND<Dim> high{};

    bool valid(int gridSize) const {
        for (int i = 0; i < Dim; ++i) {
            if (low[i] < 0 || high[i] >= gridSize || low[i] > high[i]) return false;
        }
        return true;
    }

    bool contains(const PointND<Dim>& p) const {
        for (int i = 0; i < Dim; ++i) {
            if (p[i] < low[i] || p[i] > high[i]) return false;
        }
        return true;
    }
};

template<int Dim>
std::vector<std::pair<uint64_t, uint64_t>> bruteForceSegments(
    const ACSpaceFillingCurveND<Dim>& sfc,
    const QueryBoxND<Dim>& box);

template<int Dim>
class ACBrqProcessorND {
public:
    using Sfc = ACSpaceFillingCurveND<Dim>;
    using bv_t = sdsl::bit_vector;
    using rank1_t = sdsl::rank_support_v<1>;
    using select1_t = sdsl::select_support_mcl<1>;

    explicit ACBrqProcessorND(const Sfc& sfc) : sfc_(sfc), G_(sfc.gridSize()) {}

    void initialize() {
        const uint64_t cells = totalCells();
        const uint64_t planeCells = planeSize();
        codeGrid_.assign(cells, 0);

        forEachPoint([&](const PointND<Dim>& p) {
            codeGrid_[pointToIndex(p)] = sfc_.encode(p);
        });

        planeBV_.assign(2 * Dim, std::vector<std::vector<bv_t>>(Dim));
        planeRank_.assign(2 * Dim, std::vector<std::vector<rank1_t>>(Dim));
        planeSelect_.assign(2 * Dim, std::vector<std::vector<select1_t>>(Dim));

        for (int dir = 0; dir < 2 * Dim; ++dir) {
            for (int axis = 0; axis < Dim; ++axis) {
                planeBV_[dir][axis].assign(G_, bv_t(planeCells, 0));
                planeRank_[dir][axis].resize(G_);
                planeSelect_[dir][axis].resize(G_);
            }
        }

        forEachPoint([&](const PointND<Dim>& p) {
            const uint64_t code = codeGrid_[pointToIndex(p)];
            if (code == 0) return;
            PointND<Dim> pred = sfc_.decode(code - 1);
            PointND<Dim> delta{};
            for (int k = 0; k < Dim; ++k) delta[k] = pred[k] - p[k];
            const int dir = directionFromDelta<Dim>(delta);
            if (dir < 0) {
                throw std::runtime_error("SFC violates adjacency-continuity");
            }
            for (int axis = 0; axis < Dim; ++axis) {
                planeBV_[dir][axis][p[axis]][localIndexWithoutAxis(p, axis)] = 1;
            }
        });

        for (int dir = 0; dir < 2 * Dim; ++dir) {
            for (int axis = 0; axis < Dim; ++axis) {
                for (int slice = 0; slice < G_; ++slice) {
                    sdsl::util::init_support(planeRank_[dir][axis][slice], &planeBV_[dir][axis][slice]);
                    sdsl::util::init_support(planeSelect_[dir][axis][slice], &planeBV_[dir][axis][slice]);
                }
            }
        }
    }

    void query(const QueryBoxND<Dim>& box,
               std::vector<uint64_t>& entrances,
               std::vector<uint64_t>& exits) const {
        entrances.clear();
        exits.clear();
        if (!box.valid(G_)) {
            throw std::invalid_argument("invalid query box");
        }

        const uint64_t lastCode = totalCells() - 1;
        if (box.contains(sfc_.decode(0))) entrances.push_back(0);
        if (box.contains(sfc_.decode(lastCode))) exits.push_back(lastCode);

        for (int axis = 0; axis < Dim; ++axis) {
            if (box.low[axis] > 0) {
                collectPlaneCodes(box, axis, box.low[axis], directionId(axis, -1), entrances, false, 0);
                collectPlaneCodes(box, axis, box.low[axis] - 1, directionId(axis, +1), exits, true, +1);
            }
            if (box.high[axis] + 1 < G_) {
                collectPlaneCodes(box, axis, box.high[axis], directionId(axis, +1), entrances, false, 0);
                collectPlaneCodes(box, axis, box.high[axis] + 1, directionId(axis, -1), exits, true, -1);
            }
        }

        normalize(entrances);
        normalize(exits);
    }

    std::vector<std::pair<uint64_t, uint64_t>> querySegments(const QueryBoxND<Dim>& box) const {
        std::vector<uint64_t> entrances;
        std::vector<uint64_t> exits;
        query(box, entrances, exits);
        return pairSegments(entrances, exits);
    }

    uint64_t queryVolume(const QueryBoxND<Dim>& box) const {
        uint64_t volume = 1;
        for (int i = 0; i < Dim; ++i) {
            volume *= static_cast<uint64_t>(box.high[i] - box.low[i] + 1);
        }
        return volume;
    }

    std::vector<std::pair<uint64_t, uint64_t>> querySegmentsHybrid(
        const QueryBoxND<Dim>& box,
        uint64_t bruteForceVolumeThreshold) const {
        if (!box.valid(G_)) {
            throw std::invalid_argument("invalid query box");
        }
        if (queryVolume(box) <= bruteForceVolumeThreshold) {
            return bruteForceSegments<Dim>(sfc_, box);
        }
        return querySegments(box);
    }

    static std::vector<std::pair<uint64_t, uint64_t>> pairSegments(
        const std::vector<uint64_t>& entrances,
        const std::vector<uint64_t>& exits) {
        std::vector<std::pair<uint64_t, uint64_t>> segments;
        const size_t count = std::min(entrances.size(), exits.size());
        segments.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            segments.emplace_back(entrances[i], exits[i]);
        }
        return segments;
    }

    uint64_t totalCells() const {
        uint64_t total = 1;
        for (int i = 0; i < Dim; ++i) total *= static_cast<uint64_t>(G_);
        return total;
    }

    uint64_t indexBitvectorCount() const {
        return static_cast<uint64_t>(2 * Dim) * static_cast<uint64_t>(Dim) * static_cast<uint64_t>(G_);
    }

    uint64_t indexRawBits() const {
        return indexBitvectorCount() * planeSize();
    }

private:
    const Sfc& sfc_;
    int G_;
    std::vector<uint64_t> codeGrid_;
    std::vector<std::vector<std::vector<bv_t>>> planeBV_;
    std::vector<std::vector<std::vector<rank1_t>>> planeRank_;
    std::vector<std::vector<std::vector<select1_t>>> planeSelect_;

    uint64_t planeSize() const {
        uint64_t total = 1;
        for (int i = 1; i < Dim; ++i) total *= static_cast<uint64_t>(G_);
        return total;
    }

    uint64_t pointToIndex(const PointND<Dim>& p) const {
        uint64_t idx = 0;
        for (int i = 0; i < Dim; ++i) {
            idx = idx * static_cast<uint64_t>(G_) + static_cast<uint64_t>(p[i]);
        }
        return idx;
    }

    uint64_t localIndexWithoutAxis(const PointND<Dim>& p, int axis) const {
        uint64_t idx = 0;
        for (int i = 0; i < Dim; ++i) {
            if (i == axis) continue;
            idx = idx * static_cast<uint64_t>(G_) + static_cast<uint64_t>(p[i]);
        }
        return idx;
    }

    PointND<Dim> localIndexToPoint(int axis, int slice, uint64_t local) const {
        PointND<Dim> p{};
        p[axis] = slice;
        for (int i = Dim - 1; i >= 0; --i) {
            if (i == axis) continue;
            p[i] = static_cast<int>(local % static_cast<uint64_t>(G_));
            local /= static_cast<uint64_t>(G_);
        }
        return p;
    }

    template<typename Fn>
    void forEachPoint(Fn fn) const {
        PointND<Dim> p{};
        std::function<void(int)> rec = [&](int depth) {
            if (depth == Dim) {
                fn(p);
                return;
            }
            for (int v = 0; v < G_; ++v) {
                p[depth] = v;
                rec(depth + 1);
            }
        };
        rec(0);
    }

    void collectLocalRanges(const QueryBoxND<Dim>& box,
                            int axis,
                            int depth,
                            uint64_t prefix,
                            std::vector<std::pair<uint64_t, uint64_t>>& ranges) const {
        if (depth == Dim) {
            ranges.emplace_back(prefix, prefix);
            return;
        }
        if (depth == axis) {
            collectLocalRanges(box, axis, depth + 1, prefix, ranges);
            return;
        }

        const bool lastLocalDim = nextLocalDim(depth, axis) == Dim;
        if (lastLocalDim) {
            ranges.emplace_back(prefix * static_cast<uint64_t>(G_) + static_cast<uint64_t>(box.low[depth]),
                                prefix * static_cast<uint64_t>(G_) + static_cast<uint64_t>(box.high[depth]));
            return;
        }

        for (int v = box.low[depth]; v <= box.high[depth]; ++v) {
            collectLocalRanges(box, axis, depth + 1,
                               prefix * static_cast<uint64_t>(G_) + static_cast<uint64_t>(v),
                               ranges);
        }
    }

    int nextLocalDim(int depth, int axis) const {
        for (int i = depth + 1; i < Dim; ++i) {
            if (i != axis) return i;
        }
        return Dim;
    }

    std::vector<size_t> findOnesInRange(const rank1_t& rank,
                                        const select1_t& select,
                                        uint64_t lo,
                                        uint64_t hi) const {
        std::vector<size_t> out;
        if (lo > hi) return out;
        const size_t before = rank.rank(lo);
        const size_t until = rank.rank(hi + 1);
        out.reserve(until - before);
        for (size_t k = before + 1; k <= until; ++k) {
            out.push_back(select.select(k));
        }
        return out;
    }

    void collectPlaneCodes(const QueryBoxND<Dim>& box,
                           int axis,
                           int slice,
                           int dir,
                           std::vector<uint64_t>& out,
                           bool mapToInner,
                           int innerStep) const {
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        collectLocalRanges(box, axis, 0, 0, ranges);
        const auto& rank = planeRank_[dir][axis][slice];
        const auto& select = planeSelect_[dir][axis][slice];
        for (const auto& range : ranges) {
            for (size_t local : findOnesInRange(rank, select, range.first, range.second)) {
                PointND<Dim> p = localIndexToPoint(axis, slice, static_cast<uint64_t>(local));
                if (mapToInner) p[axis] += innerStep;
                out.push_back(codeGrid_[pointToIndex(p)]);
            }
        }
    }

    static void normalize(std::vector<uint64_t>& values) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
    }
};

template<int Dim>
std::vector<std::pair<uint64_t, uint64_t>> bruteForceSegments(
    const ACSpaceFillingCurveND<Dim>& sfc,
    const QueryBoxND<Dim>& box) {
    std::set<uint64_t> codes;
    PointND<Dim> p{};
    std::function<void(int)> rec = [&](int depth) {
        if (depth == Dim) {
            codes.insert(sfc.encode(p));
            return;
        }
        for (int v = box.low[depth]; v <= box.high[depth]; ++v) {
            p[depth] = v;
            rec(depth + 1);
        }
    };
    rec(0);

    std::vector<std::pair<uint64_t, uint64_t>> segments;
    if (codes.empty()) return segments;
    auto it = codes.begin();
    uint64_t lo = *it;
    uint64_t hi = *it;
    for (++it; it != codes.end(); ++it) {
        if (*it == hi + 1) {
            hi = *it;
        } else {
            segments.emplace_back(lo, hi);
            lo = hi = *it;
        }
    }
    segments.emplace_back(lo, hi);
    return segments;
}

} // namespace ac_brq_nd

#endif // AC_BRQ_ND_HPP
