// 河谷剖面的连续重建；坐标以原生格距为单位，属性为米制。
// 私有实现：不改 D8 水系、河床几何或通用 nearest_propagate。
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include "skyisle/grid.hpp"

namespace skyisle::island::detail {

struct ValleyPoint {
    double row = 0, col = 0, bed = 0, depth = 0, width = 0, half_floor = 0, tangent = 0;
    double radius = INF;  // 溪涧作用半径（格）；常年河沿用调用方的原传播范围。
    uint8_t level = 0;
};

// 精确到非 work 格方块的 clearance。只存陆地相邻障碍格；从 work 内到更深障碍的
// 最短线段必先经过这些边界格。边界 BVH 不用格心距离，避免把半格海岸当作可穿过。
class ValleyDomain {
    struct Box { double r0,r1,c0,c1; int begin=0,end=0,left=-1,right=-1; };
    std::vector<Box> boxes_, nodes_;
    static double distance2(const Box& b,double row,double col) {
        const double dr=std::max({b.r0-row,row-b.r1,0.0}),dc=std::max({b.c0-col,col-b.c1,0.0});
        return dr*dr+dc*dc;
    }
    int build(int begin,int end) {
        Box n{INF,-INF,INF,-INF,begin,end};
        for(int k=begin;k<end;++k) {const auto& b=boxes_[k];n.r0=std::min(n.r0,b.r0);n.r1=std::max(n.r1,b.r1);n.c0=std::min(n.c0,b.c0);n.c1=std::max(n.c1,b.c1);}
        const int index=static_cast<int>(nodes_.size());nodes_.push_back(n);
        if(end-begin>8) {
            const bool row=n.r1-n.r0>n.c1-n.c0;const int mid=(begin+end)/2;
            std::nth_element(boxes_.begin()+begin,boxes_.begin()+mid,boxes_.begin()+end,[row](const auto& a,const auto& b){return row?a.r0<b.r0:a.c0<b.c0;});
            const int left=build(begin,mid),right=build(mid,end);nodes_[index].left=left;nodes_[index].right=right;
        }
        return index;
    }
    void nearest(int index,double row,double col,double& best) const {
        const auto& n=nodes_[index];if(distance2(n,row,col)>best)return;
        if(n.left<0) {for(int k=n.begin;k<n.end;++k)best=std::min(best,distance2(boxes_[k],row,col));}
        else {int a=n.left,b=n.right;if(distance2(nodes_[a],row,col)>distance2(nodes_[b],row,col))std::swap(a,b);nearest(a,row,col,best);nearest(b,row,col,best);}
    }
public:
    explicit ValleyDomain(const Mask& work) {
        for(int i=0;i<work.H;++i)for(int j=0;j<work.W;++j)if(!work(i,j)) {
            bool boundary=false;
            for(int di=-1;di<=1;++di)for(int dj=-1;dj<=1;++dj) {
                const int r=i+di,c=j+dj;
                boundary=boundary||(r>=0&&r<work.H&&c>=0&&c<work.W&&work(r,c));
            }
            if(boundary)boxes_.push_back({i-.5,i+.5,j-.5,j+.5});
        }
        // 栅格之外同样不是已提供的陆地。
        boxes_.push_back({-1.5,-.5,-1.5,work.W+.5});
        boxes_.push_back({work.H-.5,work.H+.5,-1.5,work.W+.5});
        boxes_.push_back({-.5,work.H-.5,-1.5,-.5});
        boxes_.push_back({-.5,work.H-.5,work.W-.5,work.W+.5});
        build(0,static_cast<int>(boxes_.size()));
    }
    double clearance(double row,double col) const {double d=INF;nearest(0,row,col,d);return std::sqrt(d);}
    double weight(double row,double col,double distance_cells) const {
        // 候选投影均在 d_min+sqrt(2) 的圆内。只有该圆完全位于 work 才混入欧氏场；
        // 下一原生格 smoothstep 过渡。d_min 与 clearance 连续，不新增影线或候选出生断墙。
        return smoothstep(clip(clearance(row,col)-distance_cells-SQRT2,0.0,1.0));
    }
};

// 原 masked 原生格场的双线性重建。finite 角点归一化保持岸边内部 C0；整数格仍逐值原样。
inline double valley_bilinear(const GridD& field,double row,double col) {
    row=clip(row,0.0,field.H-1.0);col=clip(col,0.0,field.W-1.0);
    const int r=static_cast<int>(std::floor(row)),c=static_cast<int>(std::floor(col));
    const double fr=row-r,fc=col-c;double sum=0,total=0;
    for(int di=0;di<2;++di)for(int dj=0;dj<2;++dj) {
        const double w=(di?fr:1-fr)*(dj?fc:1-fc);
        const double v=field(std::min(r+di,field.H-1),std::min(c+dj,field.W-1));
        if(w>0&&std::isfinite(v)){sum+=w*v;total+=w;}
    }
    return total>0?sum/total:NaN;
}

inline double guarded_valley_height(const ValleyDomain& domain,const GridD& legacy,
                                   double row,double col,double distance_cells,double height) {
    const double alpha=domain.weight(row,col,distance_cells),old=valley_bilinear(legacy,row,col);
    return alpha<=0?old:alpha>=1?height:old+alpha*(height-old);
}

inline double bank_bounded_valley_height(const GridD& surface,double row,double col,double natural,double carved) {
    const double floor=valley_bilinear(surface,row,col);
    // 岸不能被邻段低水面刻到其原对应源的平岸水面之下；无 source 支撑不施加边界。
    return std::isfinite(floor)?std::min(natural,std::max(carved,floor)):carved;
}
struct ValleySegment { ValleyPoint a, b; };
struct ValleySample {
    double distance = INF, height = 0, bed = 0, depth = 0, width = 0, half_floor = 0, fade = 0;
    uint8_t level = 0;
    bool present = false;
};

class ValleyProfiles {
    struct Node {
        double lo_row, hi_row, lo_col, hi_col;
        int begin, end, left = -1, right = -1;
    };
    struct Hit { int segment; double distance, t; };
    std::vector<ValleySegment> segments_;
    std::vector<Node> nodes_;
    double res_m_;

    static double bound(const Node& n, double row, double col) {
        const double dr = std::max({n.lo_row - row, row - n.hi_row, 0.0});
        const double dc = std::max({n.lo_col - col, col - n.hi_col, 0.0});
        return dr * dr + dc * dc;
    }
    Hit project(int k, double row, double col) const {
        const auto& s = segments_[k];
        const double dr = s.b.row - s.a.row, dc = s.b.col - s.a.col;
        const double l2 = dr * dr + dc * dc;
        const double t = l2 > 0 ? clip(((row - s.a.row) * dr + (col - s.a.col) * dc) / l2, 0.0, 1.0) : 0;
        return {k, std::hypot(row - s.a.row - t * dr, col - s.a.col - t * dc), t};
    }
    int build(int begin, int end) {
        Node n{INF, -INF, INF, -INF, begin, end};
        for (int k = begin; k < end; ++k) {
            const auto& s = segments_[k];
            n.lo_row = std::min({n.lo_row, s.a.row, s.b.row});
            n.hi_row = std::max({n.hi_row, s.a.row, s.b.row});
            n.lo_col = std::min({n.lo_col, s.a.col, s.b.col});
            n.hi_col = std::max({n.hi_col, s.a.col, s.b.col});
        }
        const int index = static_cast<int>(nodes_.size());
        nodes_.push_back(n);
        if (end - begin > 8) {
            const bool row_axis = n.hi_row - n.lo_row > n.hi_col - n.lo_col;
            const int mid = (begin + end) / 2;
            std::nth_element(segments_.begin() + begin, segments_.begin() + mid, segments_.begin() + end,
                [row_axis](const auto& a, const auto& b) {
                    const double ca = row_axis ? a.a.row + a.b.row : a.a.col + a.b.col;
                    const double cb = row_axis ? b.a.row + b.b.row : b.a.col + b.b.col;
                    if (ca != cb) return ca < cb;
                    if (a.a.row != b.a.row) return a.a.row < b.a.row;
                    return a.a.col < b.a.col;
                });
            const int left = build(begin, mid), right = build(mid, end);
            nodes_[index].left = left; nodes_[index].right = right;
        }
        return index;
    }
    void nearest(int index, double row, double col, Hit& best) const {
        const auto& n = nodes_[index];
        if (bound(n, row, col) > best.distance * best.distance) return;
        if (n.left < 0) {
            for (int k = n.begin; k < n.end; ++k) {
                const auto h = project(k, row, col);
                if (h.distance < best.distance) best = h;
            }
        } else {
            int a = n.left, b = n.right;
            if (bound(nodes_[a], row, col) > bound(nodes_[b], row, col)) std::swap(a, b);
            nearest(a, row, col, best); nearest(b, row, col, best);
        }
    }
    void collect(int index, double row, double col, double radius, std::vector<Hit>& hits) const {
        const auto& n = nodes_[index];
        if (bound(n, row, col) >= radius * radius) return;
        if (n.left < 0) {
            for (int k = n.begin; k < n.end; ++k) {
                const auto h = project(k, row, col);
                if (h.distance < radius) hits.push_back(h);
            }
        } else { collect(n.left, row, col, radius, hits); collect(n.right, row, col, radius, hits); }
    }
    static ValleyPoint interpolate(const ValleySegment& s, double t) {
        auto mix = [t](double a, double b) { return a + t * (b - a); };
        // 河/溪涧标签没有几何门槛；共用节点附近按连续河段重建。
        const double radius = std::isinf(s.a.radius) || std::isinf(s.b.radius) ? INF : mix(s.a.radius, s.b.radius);
        return {mix(s.a.row,s.b.row), mix(s.a.col,s.b.col), mix(s.a.bed,s.b.bed), mix(s.a.depth,s.b.depth),
                mix(s.a.width,s.b.width), mix(s.a.half_floor,s.b.half_floor), mix(s.a.tangent,s.b.tangent),
                radius, static_cast<uint8_t>(std::max(s.a.level, s.b.level))};
    }
public:
    ValleyProfiles(std::vector<ValleySegment> segments, double res_m) : segments_(std::move(segments)), res_m_(res_m) {
        if (!segments_.empty()) build(0, static_cast<int>(segments_.size()));
    }
    ValleySample sample(double row, double col) const {
        ValleySample out;
        if (nodes_.empty()) return out;
        Hit best{-1, INF, 0}; nearest(0, row, col, best);
        out.distance = best.distance * res_m_;
        if (best.distance < 1e-12) {
            const auto p = interpolate(segments_[best.segment], best.t);
            out.height = p.bed + p.depth; out.bed = p.bed; out.depth = p.depth;
            out.width = p.width; out.half_floor = p.half_floor; out.level = p.level;
            out.fade = 1; out.present = true; return out;
        }
        std::vector<Hit> hits;
        // 最近段之外只混合一个原生格的对角线以内的等距候选。
        // 紧支撑核在候选进出时为0；1/d²保持贴河处以本段为主，避免全局低面包络拓谷。
        collect(0, row, col, best.distance + SQRT2, hits);
        double total = 0, tangent = 0;
        for (const auto& h : hits) {
            const auto p = interpolate(segments_[h.segment], h.t);
            const double local = smoothstep(clip(1 - (h.distance - best.distance) / SQRT2, 0.0, 1.0));
            const double fade = std::isinf(p.radius) ? 1 : smoothstep(clip(p.radius - h.distance, 0.0, 1.0));
            const double weight = local * fade / (h.distance * h.distance);
            if (weight <= 0) continue;
            out.height += weight * (p.bed + p.depth);
            out.bed += weight * p.bed; out.depth += weight * p.depth;
            out.width += weight * p.width; out.half_floor += weight * p.half_floor;
            tangent += weight * p.tangent;
            out.fade = std::max(out.fade, local * fade);
            total += weight;
        }
        if (total > 0) {
            out.height /= total; out.bed /= total; out.depth /= total;
            out.width /= total; out.half_floor /= total;
            out.height += std::max(0.0, out.distance - out.half_floor) * tangent / total;
            out.level = interpolate(segments_[best.segment], best.t).level; out.present = true;
        }
        return out;
    }
};

}  // namespace skyisle::island::detail
