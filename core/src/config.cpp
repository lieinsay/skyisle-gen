#include "skyisle/config.hpp"

#include <stdexcept>

namespace skyisle {

double Config::get(const std::string& key) const {
    auto it = num.find(key);
    if (it == num.end()) throw std::out_of_range("config: missing key " + key);
    return it->second;
}

double Config::get(const std::string& key, double fallback) const {
    auto it = num.find(key);
    return it == num.end() ? fallback : it->second;
}

const std::vector<double>& Config::list(const std::string& key) const {
    auto it = vec.find(key);
    if (it == vec.end()) throw std::out_of_range("config: missing list " + key);
    return it->second;
}

std::vector<double> Config::list(const std::string& key, const std::vector<double>& fallback) const {
    auto it = vec.find(key);
    return it == vec.end() ? fallback : it->second;
}

const std::string& Config::gets(const std::string& key) const {
    auto it = str.find(key);
    if (it == str.end()) throw std::out_of_range("config: missing string " + key);
    return it->second;
}

std::string Config::gets(const std::string& key, const std::string& fallback) const {
    auto it = str.find(key);
    return it == str.end() ? fallback : it->second;
}

bool Config::has_prefix(const std::string& prefix) const {
    auto hit = [&](const auto& m) {
        auto it = m.lower_bound(prefix);
        return it != m.end() && it->first.compare(0, prefix.size(), prefix) == 0;
    };
    return hit(num) || hit(vec) || hit(str);
}

}  // namespace skyisle
