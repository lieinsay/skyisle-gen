// 配置：扁平键值表（"terrain.carve_k" → 0.3；列表 "territory.stretch" → {1.0, 1.6, 2.4}）。
// Python 前端把 [island] 段展平传进来（布尔 → 0 / 1，字符串跳过）；以后游戏从 TOML / 行星包读同一张表。缺键抛异常。
#pragma once

#include <map>
#include <string>
#include <vector>

namespace skyisle {

class Config {
public:
    std::map<std::string, double> num;
    std::map<std::string, std::vector<double>> vec;

    double get(const std::string& key) const;
    double get(const std::string& key, double fallback) const;
    int geti(const std::string& key) const { return static_cast<int>(get(key)); }
    bool has(const std::string& key) const { return num.count(key) > 0; }
    const std::vector<double>& list(const std::string& key) const;
    std::vector<double> list(const std::string& key, const std::vector<double>& fallback) const;
};

}  // namespace skyisle
