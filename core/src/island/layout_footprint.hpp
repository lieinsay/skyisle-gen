// 放置验收用原生掩膜的岸线包络；角向半径只负责提出位置，不证明两岛分开。
#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include "skyisle/island/types.hpp"

namespace skyisle::island::detail {
class LayoutFootprint {
    struct Box { double lo_x=INF,lo_y=INF,hi_x=-INF,hi_y=-INF; };
    struct Node { Box box; int lo=0,hi=0,left=-1,right=-1; };
    std::vector<std::array<double,2>> cells_,edge_;
    std::vector<int> order_;
    std::vector<Node> nodes_;
    double half_=0;
    static double distance(const Box& a,const Box& b) {
        return std::hypot(std::max({0.0,a.lo_x-b.hi_x,b.lo_x-a.hi_x}),
                          std::max({0.0,a.lo_y-b.hi_y,b.lo_y-a.hi_y}));
    }
    static Box shifted(Box b,double x,double y) {
        b.lo_x+=x;b.hi_x+=x;b.lo_y+=y;b.hi_y+=y;return b;
    }
    Box cell_box(const std::array<double,2>& p) const {return {p[0]-half_,p[1]-half_,p[0]+half_,p[1]+half_};}
    int build(int lo,int hi) {
        Box box;
        for(int k=lo;k<hi;++k) {const auto b=cell_box(cells_[order_[k]]);
            box.lo_x=std::min(box.lo_x,b.lo_x);box.lo_y=std::min(box.lo_y,b.lo_y);
            box.hi_x=std::max(box.hi_x,b.hi_x);box.hi_y=std::max(box.hi_y,b.hi_y);}
        const int n=int(nodes_.size());nodes_.push_back({box,lo,hi});
        if(hi-lo>16) {
            const int axis=box.hi_x-box.lo_x>=box.hi_y-box.lo_y?0:1,mid=(lo+hi)/2;
            std::nth_element(order_.begin()+lo,order_.begin()+mid,order_.begin()+hi,[&](int a,int b){
                return cells_[a][axis]!=cells_[b][axis]?cells_[a][axis]<cells_[b][axis]:a<b;});
            const int a=build(lo,mid),b=build(mid,hi);nodes_[n].left=a;nodes_[n].right=b;
        }
        return n;
    }
    bool near(int n,const Box& q,double gap) const {
        const auto& v=nodes_[n];if(distance(v.box,q)>=gap)return false;
        if(v.left>=0)return near(v.left,q,gap)||near(v.right,q,gap);
        for(int k=v.lo;k<v.hi;++k)if(distance(cell_box(cells_[order_[k]]),q)<gap)return true;
        return false;
    }
public:
    LayoutFootprint(const Shape& s,const std::array<double,2>& offset,double res_km) {
        // 原岸线在正格心相邻一格内，贴图再有半格取整：包络不能只用格心。
        half_=1.5*res_km;
        const Mask inner=binary_erode(s.mask,1);
        for(int r=0;r<s.mask.H;++r)for(int c=0;c<s.mask.W;++c)if(s.mask(r,c)) {
            const std::array<double,2> p={s.X(r,c)-offset[0],s.Y(r,c)-offset[1]};cells_.push_back(p);
            if(!inner(r,c))edge_.push_back(p);
        }
        order_.resize(cells_.size());std::iota(order_.begin(),order_.end(),0);
        if(!cells_.empty())build(0,int(cells_.size()));
    }
    bool separated(const LayoutFootprint& b,double dx,double dy,double gap) const {
        if(nodes_.empty()||b.nodes_.empty())return true;
        if(distance(shifted(nodes_[0].box,dx,dy),b.nodes_[0].box)>=gap)return true;
        // 对较小掩膜的全部边格查较大掩膜，既检查边交叉也检查整岛套在另一岛内。
        if(cells_.size()>b.cells_.size())return b.separated(*this,-dx,-dy,gap);
        for(const auto& p:edge_)if(b.near(0,shifted(cell_box(p),dx,dy),gap))return false;
        return true;
    }
};
} // namespace skyisle::island::detail
