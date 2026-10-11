// 全节点共同的容量压力；候选邻域仅加速，全部节点对决定最终可用域。
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "skyisle/island/territory.hpp"

namespace skyisle::island::detail {
using Point=std::array<double,2>;
using Polygon=std::vector<Point>;
inline Polygon clip_polygon(const Polygon& p,double ux,double uy,double b) {
    Polygon out;if(p.empty())return out;
    auto a=p.back();double fa=a[0]*ux+a[1]*uy-b;
    for(const auto& c:p) {
        const double fc=c[0]*ux+c[1]*uy-b;
        if((fa<=0)!=(fc<=0)) {const double t=fa/(fa-fc);out.push_back({a[0]+t*(c[0]-a[0]),a[1]+t*(c[1]-a[1])});}
        if(fc<=0)out.push_back(c);a=c;fa=fc;
    }
    return out;
}
inline double polygon_area(const Polygon& p) {
    double sum=0;if(p.empty())return 0;auto a=p.back();
    for(const auto& b:p){sum+=a[0]*b[1]-a[1]*b[0];a=b;}return .5*std::fabs(sum);
}
inline double polygon_radius(const Polygon& p) {double r=0;for(const auto& q:p)r=std::max(r,std::hypot(q[0],q[1]));return r;}
inline double spherical_bound(double t,double distance,double gap,double radius,double extent) {
    const double plain=t-.5*gap;
    if(extent<=0)return plain;
    if(!(extent<.5*PI*radius))throw std::runtime_error("群界容量：球面包络超过局部半球");
    const double half=distance/(2*radius),ct=std::cos(half);
    // 共同分界的小圆：N=(up_j-up_i)/|差|，N·p=sin((t-d/2)/R)。
    // 两侧在 asin(N·p) 上各退半 gap；界值用 sec(B/R) 作保守内近似。
    const double alpha=std::clamp((t-.5*distance)/radius,-.5*PI+.5*gap/radius,.5*PI-.5*gap/radius);
    // 整个半径 B 球冠的最大有向角为 B/R-d/(2R)。远面若已在域外，
    // 无须施加 GN 内近似（近对跖时该近似会无谓地把整片安全域排除）。
    if(radius*(alpha+half)-.5*gap>=extent)return plain;
    if(ct<=1e-12)throw std::runtime_error("群界容量：精确对跖的活跃群界需独立球面处理");
    const double threshold=std::sin(alpha-.5*gap/radius);
    const double scale=threshold<0?1./std::cos(extent/radius):1.;
    const double gnomonic=radius*(std::sin(half)+threshold*scale)/ct;
    const double factor=extent/(radius*std::tan(extent/radius));
    return std::min(plain,gnomonic*(gnomonic>=0?factor:1.));
}
class CapacitySolver {
    const PlanetView& pv_;double R_,gap_,reach_,reach_km_,box_;
    size_t n_;
    std::vector<double> lat_,lon_,sl_,cl_,r_;
    std::vector<std::vector<Limit>> edges_;
    CapacityModel result_;
    Limit pair(size_t i,size_t j,double d) const {
        const double dl=lon_[j]-lon_[i];
        const double az=std::atan2(std::sin(dl)*cl_[j],cl_[i]*sl_[j]-sl_[i]*cl_[j]*std::cos(dl));
        return {int64_t(j),d,std::sin(az),std::cos(az),d*r_[i]/(r_[i]+r_[j])};
    }
    double distance(size_t i,size_t j) const {
        return R_*std::acos(clip(sl_[i]*sl_[j]+cl_[i]*cl_[j]*std::cos(lon_[j]-lon_[i]),-1.,1.));
    }
    bool add(size_t i,const Limit& L) {
        for(const auto& e:edges_[i])if(e.node==L.node)return false;
        edges_[i].push_back(L);return true;
    }
    double bound(size_t i,const Limit& L,double own,bool spherical) const {
        return spherical_bound(L.limit_km+own-result_.pressure_km[L.node],L.dist_km,gap_,R_,spherical?result_.radius_bound_km[i]:0.);
    }
    Polygon polygon(size_t i,double own,bool spherical) const {
        Polygon p={{-box_,-box_},{box_,-box_},{box_,box_},{-box_,box_}};
        for(const auto& e:edges_[i])p=clip_polygon(p,e.ux,e.uy,bound(i,e,own,spherical));
        return p;
    }
    // 全 pair 证书：新压力或剪裁后原 reach 以外的面若能切到当前域，双方同时加入。
    int certify(bool spherical) {
        std::vector<Polygon> polys(n_);std::vector<double> extents(n_);
        for(size_t i=0;i<n_;++i){polys[i]=polygon(i,result_.pressure_km[i],spherical);extents[i]=polygon_radius(polys[i]);}
        int added=0;
        auto cuts=[&](size_t i,const Limit& e) {
            const double b=bound(i,e,result_.pressure_km[i],spherical);
            if(b>=extents[i]+1e-8)return false;
            for(const auto& p:polys[i])if(p[0]*e.ux+p[1]*e.uy>b+1e-8)return true;
            return false;
        };
        for(size_t i=0;i<n_;++i)for(size_t j=i+1;j<n_;++j) {
            const double d=distance(i,j);if(d<1e-8)throw std::runtime_error("群界容量：重复群中心");
            // 先按最小可能 plane 距离排除，不计算无关远方的方位角。
            const double t=d*r_[i]/(r_[i]+r_[j]);
            const double a=spherical_bound(t+result_.pressure_km[i]-result_.pressure_km[j],d,gap_,R_,spherical?result_.radius_bound_km[i]:0.);
            const double b=spherical_bound(d-t+result_.pressure_km[j]-result_.pressure_km[i],d,gap_,R_,spherical?result_.radius_bound_km[j]:0.);
            if(a>=extents[i]+1e-8&&b>=extents[j]+1e-8)continue;
            const auto li=pair(i,j,d),lj=pair(j,i,d);
            if(cuts(i,li)||cuts(j,lj)) {
                const bool ai=add(i,li),aj=add(j,lj);if(ai||aj)++added;
                // 证书本身只收紧域；立刻更新，不能继续对未闭合的旧 box
                // 加入数千个已被刚加入近面排除的远面。
                if(ai){polys[i]=clip_polygon(polys[i],li.ux,li.uy,bound(i,li,result_.pressure_km[i],spherical));extents[i]=polygon_radius(polys[i]);}
                if(aj){polys[j]=clip_polygon(polys[j],lj.ux,lj.uy,bound(j,lj,result_.pressure_km[j],spherical));extents[j]=polygon_radius(polys[j]);}
            }
        }
        ++result_.certificates;result_.added_pairs+=added;return added;
    }
    double required(size_t i) const {
        const double th=result_.radius_bound_km[i]/R_;
        const double jac=th>0?std::sin(th)/th:1.;
        return pv_.isl_area[i]/pv_.land_frac_cap/jac;
    }
public:
    CapacitySolver(const PlanetView& pv,double gap,double reach,double reach_km)
      :pv_(pv),R_(pv.surface_radius_km>0?pv.surface_radius_km:pv.radius_km),gap_(gap),reach_(reach),reach_km_(reach_km),box_(.5*PI*R_),n_(pv.isl_lat.size()) {
        if(!(pv.land_frac_cap>0&&pv.land_frac_cap<1))throw std::invalid_argument("群界容量：land_frac_cap 须在(0,1)内");
        lat_.resize(n_);lon_.resize(n_);sl_.resize(n_);cl_.resize(n_);r_.resize(n_);edges_.resize(n_);
        result_.pressure_km.assign(n_,0);result_.radius_bound_km.assign(n_,0);
        for(size_t i=0;i<n_;++i){lat_[i]=pv.isl_lat[i]*PI/180.;lon_[i]=pv.isl_lon[i]*PI/180.;sl_[i]=std::sin(lat_[i]);cl_[i]=std::cos(lat_[i]);r_[i]=std::sqrt(pv.isl_area[i]/PI);}
        for(size_t i=0;i<n_;++i)for(size_t j=i+1;j<n_;++j) {
            const double d=distance(i,j);
            if(d<=reach_*(r_[i]+r_[j])+reach_km_){add(i,pair(i,j,d));add(j,pair(j,i,d));}
        }
        for(auto& es:edges_)std::stable_sort(es.begin(),es.end(),[](const auto& a,const auto& b){return a.dist_km<b.dist_km;});
    }
    CapacityModel solve() {
        const bool trace=std::getenv("SKYISLE_CAPACITY_TRACE")!=nullptr;
        while(true){const int added=certify(false);if(trace)std::fprintf(stderr,"capacity initial added=%d\n",added);if(!added)break;}
        // 人口稀少的独立 probe 可以是不闭合的域；完整行星须在局部半球内闭合。
        if(n_<4){result_.boundaries=edges_;for(auto& es:result_.boundaries)for(auto& e:es)e.limit_km-=gap_*.5;return result_;}
        for(int outer=0;outer<64;++outer) {
            if(trace){size_t count=0;for(const auto& e:edges_)count+=e.size();std::fprintf(stderr,"capacity outer=%d edges=%zu\n",outer,count);}
            bool radius_changed=false;
            for(size_t i=0;i<n_;++i) {
                const double extent=polygon_radius(polygon(i,result_.pressure_km[i],false));
                if(extent>=box_*.999)throw std::runtime_error("群界容量：全节点域未在局部半球闭合");
                if(extent>result_.radius_bound_km[i]+1e-8){result_.radius_bound_km[i]=extent;radius_changed=true;}
            }
            bool converged=false;
            for(int turn=0;turn<512;++turn) {
                for(size_t i=0;i<n_;++i) {
                    const double target=required(i);
                    if(polygon_area(polygon(i,result_.pressure_km[i],true))>=target*(1.-1e-10))continue;
                    double lo=result_.pressure_km[i],hi=lo+1.;
                    while(polygon_area(polygon(i,hi,true))<target){hi=lo+2*(hi-lo);if(hi-lo>R_)throw std::runtime_error("群界容量：可用域面积不能收敛");}
                    for(int k=0;k<40;++k){const double mid=.5*(lo+hi);if(polygon_area(polygon(i,mid,true))<target)lo=mid;else hi=mid;}
                    result_.pressure_km[i]=hi;
                }
                ++result_.sweeps;double worst=0;
                for(size_t i=0;i<n_;++i)worst=std::min(worst,polygon_area(polygon(i,result_.pressure_km[i],true))/required(i)-1.);
                result_.worst_relative=worst;
                if(trace&&turn%16==0)std::fprintf(stderr,"capacity turn=%d residual=%.12g pressure=%.6f\n",turn,worst,*std::max_element(result_.pressure_km.begin(),result_.pressure_km.end()));
                if(worst>=-1e-9){converged=true;break;}
            }
            if(!converged)throw std::runtime_error("群界容量：共享压力残差未收敛");
            const int new_faces=certify(true);
            if(new_faces)continue;
            bool enclosed=true;
            for(size_t i=0;i<n_;++i)if(polygon_radius(polygon(i,result_.pressure_km[i],false))>result_.radius_bound_km[i]+1e-8)enclosed=false;
            if(!enclosed)continue;
            (void)radius_changed;
            for(auto& es:edges_)std::stable_sort(es.begin(),es.end(),[](const auto& a,const auto& b){return a.dist_km!=b.dist_km?a.dist_km<b.dist_km:a.node<b.node;});
            result_.boundaries=edges_;result_.usable_area_km2.resize(n_);
            for(size_t i=0;i<n_;++i) {
                const double th=result_.radius_bound_km[i]/R_,jac=th>0?std::sin(th)/th:1.;
                const auto poly=polygon(i,result_.pressure_km[i],true);
                result_.usable_area_km2[i]=polygon_area(poly)*jac;
                for(auto& e:result_.boundaries[i])e.limit_km=bound(i,e,result_.pressure_km[i],true);
                auto& out=result_.boundaries[i];
                out.erase(std::remove_if(out.begin(),out.end(),[&](const Limit& e){
                    for(const auto& p:poly)if(std::fabs(p[0]*e.ux+p[1]*e.uy-e.limit_km)<=1e-7)return false;return true;
                }),out.end());
                out.shrink_to_fit();
            }
            // 所有式仅依赖λ差，固定 minλ=0 的规范，不改变任何共享边。
            const double gauge=*std::min_element(result_.pressure_km.begin(),result_.pressure_km.end());
            for(auto& x:result_.pressure_km)x-=gauge;
            result_.max_pressure_km=*std::max_element(result_.pressure_km.begin(),result_.pressure_km.end());
            return std::move(result_);
        }
        throw std::runtime_error("群界容量：新活跃面与球面包络未稳定");
    }
};
} // namespace skyisle::island::detail
