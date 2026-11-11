#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ac_brq_nd.hpp"
#include "ac_sfc_nd.hpp"

namespace py = pybind11;

namespace {

int order_from_grid_size(int grid_size) {
    if (grid_size <= 0) {
        throw std::invalid_argument("grid_size must be positive");
    }
    int order = 0;
    int value = 1;
    while (value < grid_size) {
        value <<= 1;
        ++order;
    }
    if (value != grid_size) {
        throw std::invalid_argument("grid_size must be a power of two");
    }
    return order;
}

template<int Dim>
class ExternalPathSFC final : public ac_sfc_nd::ACSpaceFillingCurveND<Dim> {
public:
    using Base = ac_sfc_nd::ACSpaceFillingCurveND<Dim>;
    using Point = ac_sfc_nd::PointND<Dim>;

    ExternalPathSFC(py::array_t<int64_t, py::array::c_style | py::array::forcecast> path,
                    int grid_size)
        : Base(order_from_grid_size(grid_size)) {
        auto buf = path.request();
        if (buf.ndim != 2) {
            throw std::invalid_argument("path must be a 2-D array with shape (num_cells, dim)");
        }
        if (buf.shape[1] != Dim) {
            throw std::invalid_argument("path second dimension does not match template dimension");
        }

        const uint64_t expected = this->totalCodes();
        if (static_cast<uint64_t>(buf.shape[0]) != expected) {
            throw std::invalid_argument("path row count must equal grid_size ** dim");
        }

        path_.resize(expected);
        code_grid_.assign(expected, UINT64_MAX);

        const int64_t* data = static_cast<const int64_t*>(buf.ptr);
        for (uint64_t code = 0; code < expected; ++code) {
            Point p{};
            for (int axis = 0; axis < Dim; ++axis) {
                const int64_t coord = data[code * Dim + axis];
                if (coord < 0 || coord >= this->gridSize()) {
                    throw std::invalid_argument("path contains coordinate outside grid");
                }
                p[axis] = static_cast<int>(coord);
            }
            const uint64_t idx = point_to_index(p);
            if (code_grid_[idx] != UINT64_MAX) {
                throw std::invalid_argument("path is not bijective: duplicate cell");
            }
            path_[code] = p;
            code_grid_[idx] = code;
        }

        for (uint64_t idx = 0; idx < expected; ++idx) {
            if (code_grid_[idx] == UINT64_MAX) {
                throw std::invalid_argument("path is not bijective: missing cell");
            }
        }
    }

    std::string name() const override {
        return std::string("ExternalPathSFC<") + std::to_string(Dim) + ">";
    }

    uint64_t encode(const Point& p) const override {
        return code_grid_[point_to_index(p)];
    }

    Point decode(uint64_t code) const override {
        if (code >= path_.size()) {
            throw std::out_of_range("code out of range");
        }
        return path_[code];
    }

private:
    std::vector<Point> path_;
    std::vector<uint64_t> code_grid_;

    uint64_t point_to_index(const Point& p) const {
        uint64_t idx = 0;
        for (int axis = 0; axis < Dim; ++axis) {
            idx = idx * static_cast<uint64_t>(this->gridSize()) + static_cast<uint64_t>(p[axis]);
        }
        return idx;
    }
};

template<int Dim>
class BRQIndex {
public:
    BRQIndex(py::array_t<int64_t, py::array::c_style | py::array::forcecast> path,
             int grid_size)
        : sfc_(std::make_shared<ExternalPathSFC<Dim>>(path, grid_size)),
          processor_(std::make_unique<ac_brq_nd::ACBrqProcessorND<Dim>>(*sfc_)) {
        if (!ac_sfc_nd::verifyAC<Dim>(*sfc_)) {
            throw std::invalid_argument("path violates adjacency-continuity");
        }
        processor_->initialize();
    }

    std::vector<std::pair<uint64_t, uint64_t>> query_segments(
        const std::vector<int>& low,
        const std::vector<int>& high) const {
        return processor_->querySegments(make_box(low, high));
    }

    std::vector<std::pair<uint64_t, uint64_t>> query_segments_hybrid(
        const std::vector<int>& low,
        const std::vector<int>& high,
        uint64_t brute_force_volume_threshold) const {
        return processor_->querySegmentsHybrid(make_box(low, high), brute_force_volume_threshold);
    }

    uint64_t index_bitvector_count() const {
        return processor_->indexBitvectorCount();
    }

    uint64_t index_raw_bits() const {
        return processor_->indexRawBits();
    }

private:
    std::shared_ptr<ExternalPathSFC<Dim>> sfc_;
    std::unique_ptr<ac_brq_nd::ACBrqProcessorND<Dim>> processor_;

    ac_brq_nd::QueryBoxND<Dim> make_box(const std::vector<int>& low,
                                        const std::vector<int>& high) const {
        if (low.size() != Dim || high.size() != Dim) {
            throw std::invalid_argument("low/high length does not match dimension");
        }
        ac_brq_nd::QueryBoxND<Dim> box;
        for (int axis = 0; axis < Dim; ++axis) {
            box.low[axis] = low[axis];
            box.high[axis] = high[axis];
        }
        if (!box.valid(sfc_->gridSize())) {
            throw std::invalid_argument("invalid query box");
        }
        return box;
    }
};

template<int Dim>
void bind_brq_index(py::module_& m, const char* name) {
    py::class_<BRQIndex<Dim>>(m, name)
        .def(py::init<py::array_t<int64_t, py::array::c_style | py::array::forcecast>, int>(),
             py::arg("path"), py::arg("grid_size"))
        .def("query_segments", &BRQIndex<Dim>::query_segments,
             py::arg("low"), py::arg("high"))
        .def("query_segments_hybrid", &BRQIndex<Dim>::query_segments_hybrid,
             py::arg("low"), py::arg("high"), py::arg("brute_force_volume_threshold"))
        .def("index_bitvector_count", &BRQIndex<Dim>::index_bitvector_count)
        .def("index_raw_bits", &BRQIndex<Dim>::index_raw_bits);
}

}  // namespace

PYBIND11_MODULE(lacsfc_brq, m) {
    m.doc() = "AC-BRQ v2 SDSL rank/select wrapper for external LACSFC paths";
    bind_brq_index<2>(m, "BRQIndex2");
    bind_brq_index<3>(m, "BRQIndex3");
    bind_brq_index<4>(m, "BRQIndex4");
    bind_brq_index<5>(m, "BRQIndex5");
    bind_brq_index<6>(m, "BRQIndex6");
}
