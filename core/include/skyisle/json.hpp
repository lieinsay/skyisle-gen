// 极简的 JSON 值（有序对象）：第三层的记录（资源、聚落、气候）从 C++ 交给前端 / 游戏的形。
// 数分整数与浮点（Python 那边 1 与 1.0 在 JSON 里不同）；字符串一律 ASCII 代码，中文名由前端映射（PLAN-CORE 第二节）。
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace skyisle {

class Json {
public:
    enum Type { NUL, BOOL, INT, NUM, STR, ARR, OBJ };
    using Obj = std::vector<std::pair<std::string, Json>>;
    using Arr = std::vector<Json>;

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool b) : t_(BOOL), b_(b) {}
    template <class T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
    Json(T v) : t_(INT), i_(static_cast<int64_t>(v)) {}
    Json(double d) : t_(NUM), d_(d) {}
    Json(const char* s) : t_(STR), s_(s) {}
    Json(std::string s) : t_(STR), s_(std::move(s)) {}
    static Json arr() { Json j; j.t_ = ARR; return j; }
    static Json obj() { Json j; j.t_ = OBJ; return j; }
    template <class T>
    static Json arr_of(const std::vector<T>& v) {
        Json j = arr();
        for (const T& x : v) j.a_.emplace_back(x);
        return j;
    }
    static Json pair(double a, double b) { Json j = arr(); j.a_.emplace_back(a); j.a_.emplace_back(b); return j; }
    static Json ipair(int64_t a, int64_t b) { Json j = arr(); j.a_.emplace_back(a); j.a_.emplace_back(b); return j; }

    Type type() const { return t_; }
    bool is_null() const { return t_ == NUL; }
    bool as_bool() const { return b_; }
    int64_t as_int() const { return t_ == NUM ? static_cast<int64_t>(d_) : i_; }
    double as_num() const { return t_ == INT ? static_cast<double>(i_) : d_; }
    const std::string& as_str() const { return s_; }
    const Arr& items() const { return a_; }
    Arr& items() { return a_; }
    const Obj& fields() const { return o_; }
    Obj& fields() { return o_; }

    // 对象：有则替换、无则追加（保持首次出现的次序）
    Json& set(const std::string& k, Json v) {
        if (t_ != OBJ) throw std::logic_error("Json::set on non-object");
        for (auto& kv : o_)
            if (kv.first == k) {
                kv.second = std::move(v);
                return kv.second;
            }
        o_.emplace_back(k, std::move(v));
        return o_.back().second;
    }
    bool has(const std::string& k) const {
        for (const auto& kv : o_)
            if (kv.first == k) return true;
        return false;
    }
    const Json& at(const std::string& k) const {
        for (const auto& kv : o_)
            if (kv.first == k) return kv.second;
        throw std::out_of_range("Json: no key " + k);
    }
    Json& push(Json v) {
        if (t_ != ARR) throw std::logic_error("Json::push on non-array");
        a_.push_back(std::move(v));
        return a_.back();
    }
    size_t size() const { return t_ == ARR ? a_.size() : o_.size(); }

private:
    Type t_ = NUL;
    bool b_ = false;
    int64_t i_ = 0;
    double d_ = 0.0;
    std::string s_;
    Arr a_;
    Obj o_;
};

}  // namespace skyisle
