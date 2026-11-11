#ifndef AC_SFC_ND_HPP
#define AC_SFC_ND_HPP

// ============================================================================
// N 维 Adjacency-Continuous Space-Filling Curves
// ----------------------------------------------------------------------------
// 抽象 N 维 AC SFC 接口 + 两个具体实现：
//   * SnakeCurveND<N>    —— 递归 boustrophedon，恒为 AC，O(1) encode/decode
//   * HilbertCurveND<N>  —— Skilling (2004) N 维 Hilbert 编码算法
//
// 方向编码（与 2D 版无关，直接用统一的 2N 空间）：
//   direction id = 2 * axis + sign_bit
//     sign_bit = 0  ⇒  前驱在 axis 负方向 (Δaxis = -1)
//     sign_bit = 1  ⇒  前驱在 axis 正方向 (Δaxis = +1)
// 总共 2N 个方向。AC 性质保证 (dx_0, ..., dx_{N-1}) 中恰好一项为 ±1，其余为 0。
// ============================================================================

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>

namespace ac_sfc_nd {

template<int N>
using PointND = std::array<int, N>;

constexpr int directionId(int axis, int sign) {
    // sign in {-1, +1} -> sign_bit in {0, 1}
    return 2 * axis + (sign > 0 ? 1 : 0);
}

template<int N>
inline int directionFromDelta(const PointND<N>& delta) {
    // 前提：恰好一项为 ±1，其余为 0；否则返回 -1
    int axis = -1, sign = 0, nonzero = 0;
    for (int i = 0; i < N; ++i) {
        if (delta[i] == 0) continue;
        if (delta[i] != 1 && delta[i] != -1) return -1;
        ++nonzero;
        axis = i;
        sign = delta[i];
    }
    if (nonzero != 1) return -1;
    return directionId(axis, sign);
}

// ----------------------------------------------------------------------------
// 抽象 N 维 AC SFC
// ----------------------------------------------------------------------------
template<int N>
class ACSpaceFillingCurveND {
public:
    explicit ACSpaceFillingCurveND(int order)
        : order_(order), gridSize_(1 << order) {}
    virtual ~ACSpaceFillingCurveND() = default;

    int order()    const { return order_; }
    int gridSize() const { return gridSize_; }
    uint64_t totalCodes() const {
        uint64_t n = 1;
        for (int i = 0; i < N; ++i) n *= static_cast<uint64_t>(gridSize_);
        return n;
    }

    virtual std::string  name()                 const = 0;
    virtual uint64_t     encode(const PointND<N>& p) const = 0;
    virtual PointND<N>   decode(uint64_t code)  const = 0;

protected:
    int order_;
    int gridSize_;
};

// ----------------------------------------------------------------------------
// 实现 1：Snake N-D (boustrophedon)
//   定义（递归）：
//     N==1: code = x_0
//     N>=2: rest = (x_1,...,x_{N-1}); rest_code = snake_{N-1}(rest)
//           along = (rest_code 偶数) ? x_0 : (G - 1 - x_0)
//           code  = rest_code * G + along
//   AC 证明：code+1 要么 along 改变 ±1（同一 rest），要么 along 已到行末，
//   rest_code 进位 1（rest 的某个分量改变 ±1，AC 由 N-1 归纳成立），且
//   along 在新行的起点 = 上一行末端的同一 x_0，邻接。
// ----------------------------------------------------------------------------
template<int N>
class SnakeCurveND : public ACSpaceFillingCurveND<N> {
public:
    using Base = ACSpaceFillingCurveND<N>;
    using Base::order_;
    using Base::gridSize_;

    explicit SnakeCurveND(int order) : Base(order) {}

    std::string name() const override {
        return std::string("SnakeND<") + std::to_string(N) + ">";
    }

    uint64_t encode(const PointND<N>& p) const override {
        const int G = gridSize_;
        uint64_t rest_code = 0;
        // 从最高维 (N-1) 向 1 维递推，0 维做 along
        for (int d = N - 1; d >= 1; --d) {
            // 当前层的 along 是 p[d]（在 N-1 ... 1 这些维度上递推累计）
            // rest_code 表示截至到 d 的累积“行号”
            uint64_t along = (rest_code % 2 == 0)
                           ? static_cast<uint64_t>(p[d])
                           : static_cast<uint64_t>(G - 1 - p[d]);
            rest_code = rest_code * G + along;
        }
        // 最内层 axis 0
        uint64_t along0 = (rest_code % 2 == 0)
                        ? static_cast<uint64_t>(p[0])
                        : static_cast<uint64_t>(G - 1 - p[0]);
        return rest_code * G + along0;
    }

    PointND<N> decode(uint64_t code) const override {
        const int G = gridSize_;
        PointND<N> p{};
        // 反向：先取 along0，再递归剥离
        uint64_t rest_code = code / G;
        uint64_t along0    = code % G;
        p[0] = (rest_code % 2 == 0) ? static_cast<int>(along0)
                                    : (G - 1 - static_cast<int>(along0));
        // 现在 rest_code 表示 N-1 维 snake 编码
        // 逐维剥离 axis = 1, 2, ..., N-1
        // 等价地：再次递推，但参考 rest_code
        // 我们需要还原每个 p[d]，需要知道每一层的"父 rest_code"
        // 用栈式分解：先收集每一层的 rest_code
        std::array<uint64_t, N> rests;
        rests[0] = code / G;
        for (int d = 1; d < N - 1; ++d) {
            rests[d] = rests[d - 1] / G;
        }
        // 最高层 rest = 0
        // p[d] 的 along 来自 rests[d-1] % G，奇偶取决于 rests[d]（其上层）
        for (int d = 1; d <= N - 1; ++d) {
            uint64_t parent = (d == N - 1) ? 0 : rests[d];
            uint64_t along  = rests[d - 1] % G;
            p[d] = (parent % 2 == 0) ? static_cast<int>(along)
                                     : (G - 1 - static_cast<int>(along));
        }
        return p;
    }
};

// 特化：N=1 直接退化为 identity
template<>
class SnakeCurveND<1> : public ACSpaceFillingCurveND<1> {
public:
    explicit SnakeCurveND(int order) : ACSpaceFillingCurveND<1>(order) {}
    std::string name() const override { return "SnakeND<1>"; }
    uint64_t encode(const PointND<1>& p) const override { return p[0]; }
    PointND<1> decode(uint64_t code) const override { return {static_cast<int>(code)}; }
};

// ----------------------------------------------------------------------------
// 实现 2：N-D Hilbert via Skilling (2004) "Programming the Hilbert curve"
// ----------------------------------------------------------------------------
template<int N>
class HilbertCurveND : public ACSpaceFillingCurveND<N> {
public:
    using Base = ACSpaceFillingCurveND<N>;
    using Base::order_;
    using Base::gridSize_;

    explicit HilbertCurveND(int order) : Base(order) {
        assert(order >= 1 && order <= 20);
    }

    std::string name() const override {
        return std::string("HilbertND<") + std::to_string(N) + ">";
    }

    uint64_t encode(const PointND<N>& p) const override {
        const int b = order_;
        std::array<uint32_t, N> X{};
        for (int i = 0; i < N; ++i) X[i] = static_cast<uint32_t>(p[i]);

        // ---- AxesToTranspose (Skilling) ----
        uint32_t M = 1u << (b - 1);
        for (uint32_t Q = M; Q > 1; Q >>= 1) {
            uint32_t P = Q - 1;
            for (int i = 0; i < N; ++i) {
                if (X[i] & Q) {
                    X[0] ^= P;
                } else {
                    uint32_t t = (X[0] ^ X[i]) & P;
                    X[0] ^= t;
                    X[i] ^= t;
                }
            }
        }
        // Gray encode
        for (int i = 1; i < N; ++i) X[i] ^= X[i - 1];
        uint32_t t = 0;
        for (uint32_t Q = M; Q > 1; Q >>= 1) {
            if (X[N - 1] & Q) t ^= Q - 1;
        }
        for (int i = 0; i < N; ++i) X[i] ^= t;

        // ---- Interleave bits to scalar code ----
        uint64_t code = 0;
        for (int bit = b - 1; bit >= 0; --bit) {
            for (int axis = 0; axis < N; ++axis) {
                code = (code << 1) | ((X[axis] >> bit) & 1u);
            }
        }
        return code;
    }

    PointND<N> decode(uint64_t code) const override {
        const int b = order_;
        std::array<uint32_t, N> X{};
        // ---- Deinterleave ----
        for (int bit = b - 1; bit >= 0; --bit) {
            for (int axis = 0; axis < N; ++axis) {
                int shift = bit * N + (N - 1 - axis);
                X[axis] = (X[axis] << 1) | static_cast<uint32_t>((code >> shift) & 1ULL);
            }
        }
        // ---- TransposeToAxes (Skilling inverse) ----
        uint32_t Nlim = 2u << (b - 1);
        // Gray decode
        uint32_t t = X[N - 1] >> 1;
        for (int i = N - 1; i > 0; --i) X[i] ^= X[i - 1];
        X[0] ^= t;
        // Undo excess work
        for (uint32_t Q = 2; Q != Nlim; Q <<= 1) {
            uint32_t P = Q - 1;
            for (int i = N - 1; i >= 0; --i) {
                if (X[i] & Q) {
                    X[0] ^= P;
                } else {
                    uint32_t tt = (X[0] ^ X[i]) & P;
                    X[0] ^= tt;
                    X[i] ^= tt;
                }
            }
        }
        PointND<N> p{};
        for (int i = 0; i < N; ++i) p[i] = static_cast<int>(X[i]);
        return p;
    }
};

// ----------------------------------------------------------------------------
// 自检：验证 SFC 满足 AC，以及 encode∘decode = id
// ----------------------------------------------------------------------------
template<int N>
inline bool verifyAC(const ACSpaceFillingCurveND<N>& sfc) {
    uint64_t M = sfc.totalCodes();
    PointND<N> prev = sfc.decode(0);
    for (uint64_t z = 1; z < M; ++z) {
        PointND<N> cur = sfc.decode(z);
        int sum = 0;
        for (int i = 0; i < N; ++i) sum += std::abs(cur[i] - prev[i]);
        if (sum != 1) return false;
        prev = cur;
    }
    return true;
}

template<int N>
inline bool verifyBijection(const ACSpaceFillingCurveND<N>& sfc) {
    uint64_t M = sfc.totalCodes();
    std::vector<uint8_t> seen(M, 0);
    for (uint64_t z = 0; z < M; ++z) {
        PointND<N> p = sfc.decode(z);
        for (int i = 0; i < N; ++i) {
            if (p[i] < 0 || p[i] >= sfc.gridSize()) return false;
        }
        uint64_t z2 = sfc.encode(p);
        if (z2 != z) return false;
        if (seen[z]) return false;
        seen[z] = 1;
    }
    return true;
}

} // namespace ac_sfc_nd

#endif // AC_SFC_ND_HPP
