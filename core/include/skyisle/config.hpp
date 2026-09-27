// 配置：扁平键值表（"terrain.carve_k" → 0.3；列表 "territory.stretch" → {1.0, 1.6, 2.4}；字符串 "s01.calendar.mode" → "calendar_to_orbit"）。
// Python 前端把 [island] 段（第三层）或 shared / skeleton / s01–s04 段（行星层，P6c）展平传进来（布尔 → 0 / 1）；以后游戏从 TOML / 行星包读同一张表。缺键抛异常。
#pragma once

#include <map>
#include <string>
#include <vector>

namespace skyisle {

class Config {
public:
    std::map<std::string, double> num;
    std::map<std::string, std::vector<double>> vec;
    std::map<std::string, std::string> str;

    double get(const std::string& key) const;
    double get(const std::string& key, double fallback) const;
    int geti(const std::string& key) const { return static_cast<int>(get(key)); }
    bool has(const std::string& key) const { return num.count(key) > 0; }
    bool has_list(const std::string& key) const { return vec.count(key) > 0; }
    const std::vector<double>& list(const std::string& key) const;
    std::vector<double> list(const std::string& key, const std::vector<double>& fallback) const;
    const std::string& gets(const std::string& key) const;
    std::string gets(const std::string& key, const std::string& fallback) const;
    bool has_prefix(const std::string& prefix) const;   // 有没有以 prefix 开头的键（数、列表、字符串都算）：Python 的「这一段非空」
};

}  // namespace skyisle
