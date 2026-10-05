#include "skyisle/island/layered_aquifer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace skyisle::island {
namespace {
double potential(double head, double bottom, double top) {
    if (head <= bottom) return 0;
    if (head >= top) return (top-bottom)*(head-(bottom+top)/2);
    return .5*(head-bottom)*(head-bottom);
}
double thickness(double head, double bottom, double top) {
    return std::clamp(head-bottom, 0.0, top-bottom);
}
double stored(double head, double bottom, double top, const HydraulicMaterial& m) {
    return m.specific_yield*thickness(head, bottom, top)+
        m.specific_storage_per_m*(top-bottom)*std::max(0.0, head-top);
}
}

void LayeredAquifer::connect(Edge edge) {
    if (!(edge.conductance > 0)) return;
    nodes_[edge.a].edges.push_back(edges_.size());
    nodes_[edge.b].edges.push_back(edges_.size());
    edges_.push_back(edge);
}

LayeredAquifer::LayeredAquifer(const GridD& surface, const GridD& structural, const GridD& skeleton,
    const Mask& land, const StratRec& strata,
    const std::array<HydraulicMaterial, LI_COUNT>& materials, double dx,
    double shallow_depth, const std::array<HydraulicMaterial, LI_COUNT>& shallow_materials)
    : height_(surface.H), width_(surface.W), area_(dx*dx), land_(land) {
    if (!(std::isfinite(dx) && dx > 0) || land.H != height_ || land.W != width_ ||
        structural.H != height_ || structural.W != width_ || skeleton.H != height_ || skeleton.W != width_)
        throw std::invalid_argument("invalid layered aquifer grid");
    if (!(std::isfinite(shallow_depth) && shallow_depth >= 0))
        throw std::invalid_argument("invalid shallow hydraulic horizon depth");
    const auto validate = [](const HydraulicMaterial& m) {
        if (!(std::isfinite(m.horizontal_ms) && m.horizontal_ms >= 0 &&
              std::isfinite(m.vertical_ms) && m.vertical_ms >= 0 &&
              std::isfinite(m.specific_yield) && m.specific_yield > 0 && m.specific_yield <= 1 &&
              std::isfinite(m.specific_storage_per_m) && m.specific_storage_per_m >= 0))
            throw std::invalid_argument("invalid explicit hydraulic material");
    };
    for (int li = 1; li < LI_COUNT; ++li) {
        validate(materials[li]);
        if (shallow_depth > 0) validate(shallow_materials[li]);
    }
    double maximum = -INF;
    for (size_t k = 0; k < surface.size(); ++k) if (land.v[k]) maximum = std::max(maximum, surface.v[k]);
    offsets_.resize(surface.size()+1);
    for (size_t k = 0; k < surface.size(); ++k) {
        offsets_[k] = nodes_.size();
        if (!land.v[k]) continue;
        auto column = strat_column(surface.v[k], structural.v[k], skeleton.v[k], strata);
        // Split surviving rock at a depth below today's surface, retaining all
        // geological contacts. The depth and both tables are explicit inputs.
        const double contact = surface.v[k]-shallow_depth;
        if (shallow_depth > 0) {
            std::vector<StratLayer> split;
            for (const auto& layer : column) {
                const auto& a = materials[layer.lith]; const auto& b = shallow_materials[layer.lith];
                const bool differs = a.horizontal_ms != b.horizontal_ms || a.vertical_ms != b.vertical_ms ||
                    a.specific_yield != b.specific_yield || a.specific_storage_per_m != b.specific_storage_per_m;
                if (differs && contact > layer.bottom_m && contact < layer.top_m) {
                    split.push_back({layer.bottom_m, contact, layer.lith});
                    split.push_back({contact, layer.top_m, layer.lith});
                } else split.push_back(layer);
            }
            column = std::move(split);
        }
        for (size_t j = 0; j < column.size(); ++j) {
            const auto& layer = column[j];
            Node node;
            node.cell = k; node.bottom = layer.bottom_m; node.top = layer.top_m; node.lith = layer.lith;
            node.material = shallow_depth > 0 && layer.bottom_m >= contact ?
                shallow_materials[node.lith] : materials[node.lith];
            node.surface = j+1 == column.size();
            node.maximum_head = node.surface ? surface.v[k] : maximum;
            nodes_.push_back(node);
        }
    }
    offsets_.back() = nodes_.size();
    for (size_t cell = 0; cell < surface.size(); ++cell) {
        if (!land.v[cell]) continue;
        const int row = static_cast<int>(cell/width_), col = static_cast<int>(cell%width_);
        for (size_t i = offsets_[cell]+1; i < offsets_[cell+1]; ++i) {
            const auto& up = nodes_[i]; const auto& down = nodes_[i-1];
            if (up.material.vertical_ms > 0 && down.material.vertical_ms > 0) {
                const double resistance = (up.top-up.bottom)/(2*up.material.vertical_ms)+
                                          (down.top-down.bottom)/(2*down.material.vertical_ms);
                connect({i, i-1, area_/resistance, up.bottom, up.top, true});
            }
        }
        for (const auto& direction : N4) {
            const int r = row+direction[0], c = col+direction[1];
            if (r < 0 || c < 0 || r >= height_ || c >= width_ || !land(r, c)) {
                for (size_t i = offsets_[cell]; i < offsets_[cell+1]; ++i)
                    nodes_[i].coast += 2*nodes_[i].material.horizontal_ms;
                continue;
            }
            const size_t other = static_cast<size_t>(r)*width_+c;
            // A lower neighboring land surface exposes the upper part of this
            // face. Its seepage enters that land cell, not the coastal budget.
            for (size_t i = offsets_[cell]; i < offsets_[cell+1]; ++i) {
                auto& n = nodes_[i];
                const double low = std::max(n.bottom, surface.v[other]);
                if (n.top > low && n.material.horizontal_ms > 0)
                    n.slope_faces.push_back({other, low, n.top});
            }
            if (other <= cell) continue;
            size_t a = offsets_[cell], b = offsets_[other];
            while (a < offsets_[cell+1] && b < offsets_[other+1]) {
                const auto& x = nodes_[a]; const auto& y = nodes_[b];
                const double low = std::max(x.bottom, y.bottom), high = std::min(x.top, y.top);
                const double ka = x.material.horizontal_ms, kb = y.material.horizontal_ms;
                if (high > low && ka > 0 && kb > 0)
                    connect({a, b, 2/(1/ka+1/kb), low, high, false});
                const bool advance_a = x.top <= y.top, advance_b = y.top <= x.top;
                if (advance_a) ++a;
                if (advance_b) ++b;
            }
        }
    }
}

LayeredAquiferResult LayeredAquifer::advance(const GridD& recharge, const std::vector<double>* previous,
    double dt, int max_iterations, double rtol) const {
    if (recharge.H != height_ || recharge.W != width_ || max_iterations < 1 ||
        !(std::isfinite(rtol) && rtol > 0 && rtol < 1) ||
        (previous && (previous->size() != nodes_.size() || !(std::isfinite(dt) && dt > 0))))
        throw std::invalid_argument("invalid layered aquifer solve inputs");
    LayeredAquiferResult out;
    out.surface_m3s = out.coast_m3s = out.storage_change_m3 = GridD(height_, width_, 0.0);
    out.head_m.resize(nodes_.size()); out.residual_m3s.resize(nodes_.size());
    std::vector<double> source(nodes_.size(), 0), old_storage(nodes_.size(), 0);
    for (size_t k = 0; k < land_.size(); ++k) {
        if (!land_.v[k]) continue;
        if (!(std::isfinite(recharge.v[k]) && recharge.v[k] >= 0))
            throw std::invalid_argument("invalid layered aquifer recharge");
        const double q = recharge.v[k]*area_;
        out.input_m3s += q;
        if (offsets_[k] == offsets_[k+1]) out.surface_m3s.v[k] = q;
        else source[offsets_[k+1]-1] = q;
    }
    const GridD direct_surface = out.surface_m3s;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& n = nodes_[i];
        const double h = previous ? (*previous)[i] : n.bottom;
        if (!(std::isfinite(h) && h >= n.bottom && h <= n.maximum_head))
            throw std::invalid_argument("invalid layered aquifer initial head");
        out.head_m[i] = h;
        if (previous) old_storage[i] = stored(h, n.bottom, n.top, n.material);
    }
    auto evaluate = [&](size_t i, double h, double& gradient) {
        const auto& n = nodes_[i];
        double value = n.coast*potential(h, n.bottom, n.top);
        gradient = n.coast*thickness(h, n.bottom, n.top);
        for (const auto& face : n.slope_faces) {
            value += 2*n.material.horizontal_ms*potential(h, face.bottom, face.top);
            gradient += 2*n.material.horizontal_ms*thickness(h, face.bottom, face.top);
        }
        if (previous) {
            value += area_/dt*(stored(h, n.bottom, n.top, n.material)-old_storage[i]);
            gradient += area_/dt*(h < n.top ? n.material.specific_yield :
                                                  n.material.specific_storage_per_m*(n.top-n.bottom));
        }
        for (size_t e : n.edges) {
            const auto& edge = edges_[e];
            const bool first = i == edge.a;
            const double other = out.head_m[first ? edge.b : edge.a];
            if (edge.vertical) {
                // Explicit perched correction: a dewatered lower cell cannot
                // pull with the entire head drop across its unsaturated zone.
                const double upper = first ? h : other, lower = first ? other : h;
                const double q = edge.conductance*(upper-std::max(lower, edge.bottom));
                value += first ? q : -q;
                gradient += first || h > edge.bottom ? edge.conductance : 0;
            } else {
                value += edge.conductance*(potential(h, edge.bottom, edge.top)-potential(other, edge.bottom, edge.top));
                gradient += edge.conductance*thickness(h, edge.bottom, edge.top);
            }
        }
        return value;
    };
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const auto& n = nodes_[i];
            double derivative = 0, low = n.bottom, high = n.maximum_head;
            if (evaluate(i, low, derivative) == source[i]) { out.head_m[i] = low; continue; }
            if (evaluate(i, high, derivative) <= source[i]) { out.head_m[i] = high; continue; }
            double head = std::clamp(out.head_m[i], low, high);
            for (int j = 0; j < 40; ++j) {
                const double error = evaluate(i, head, derivative)-source[i];
                if (std::abs(error) <= std::max(1e-15, source[i]*1e-12)) break;
                if (error > 0) high = head; else low = head;
                double next = derivative > 0 ? head-error/derivative : (low+high)/2;
                if (!(next > low && next < high)) next = (low+high)/2;
                head = next;
            }
            const double omega = previous ? 1.0 : 1.35;
            out.head_m[i] = std::clamp(out.head_m[i]+omega*(head-out.head_m[i]), n.bottom, n.maximum_head);
        }
        out.iterations = iteration+1; out.absolute_residual_m3s = 0;
        out.output_m3s = 0; out.total_storage_change_m3 = 0;
        out.surface_m3s = direct_surface;
        std::fill(out.coast_m3s.v.begin(), out.coast_m3s.v.end(), 0);
        std::fill(out.storage_change_m3.v.begin(), out.storage_change_m3.v.end(), 0);
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const auto& n = nodes_[i];
            const double h = out.head_m[i];
            double derivative = 0;
            const double error = source[i]-evaluate(i, h, derivative);
            const double spring = n.surface && h == n.top ? std::max(0.0, error) : 0;
            const double coast = n.coast*potential(h, n.bottom, n.top);
            const double change = previous ? area_*(stored(h, n.bottom, n.top, n.material)-old_storage[i]) : 0;
            out.surface_m3s.v[n.cell] += spring; out.coast_m3s.v[n.cell] += coast;
            for (const auto& face : n.slope_faces)
                out.surface_m3s.v[face.receiver] += 2*n.material.horizontal_ms*potential(h, face.bottom, face.top);
            out.storage_change_m3.v[n.cell] += change; out.total_storage_change_m3 += change;
            out.residual_m3s[i] = error-spring;
            out.absolute_residual_m3s += std::abs(out.residual_m3s[i]);
        }
        for (size_t k = 0; k < land_.size(); ++k) out.output_m3s += out.surface_m3s.v[k]+out.coast_m3s.v[k];
        const double tolerance = std::max(1e-12, std::max(out.input_m3s, out.output_m3s)*rtol);
        if (out.absolute_residual_m3s <= tolerance) { out.converged = true; break; }
    }
    return out;
}

LayeredAquiferResult LayeredAquifer::solve(const GridD& r, int max_iterations, double rtol) const {
    return advance(r, nullptr, 0, max_iterations, rtol);
}
LayeredAquiferResult LayeredAquifer::step(const GridD& r, const std::vector<double>& previous,
    double dt, int max_iterations, double rtol) const {
    return advance(r, &previous, dt, max_iterations, rtol);
}
std::vector<int64_t> LayeredAquifer::node_cells() const {
    std::vector<int64_t> v; for (const auto& n : nodes_) v.push_back(static_cast<int64_t>(n.cell)); return v;
}
std::vector<double> LayeredAquifer::node_bottoms() const {
    std::vector<double> v; for (const auto& n : nodes_) v.push_back(n.bottom); return v;
}
std::vector<double> LayeredAquifer::node_tops() const {
    std::vector<double> v; for (const auto& n : nodes_) v.push_back(n.top); return v;
}
std::vector<int> LayeredAquifer::node_lithologies() const {
    std::vector<int> v; for (const auto& n : nodes_) v.push_back(n.lith); return v;
}
} // namespace skyisle::island
