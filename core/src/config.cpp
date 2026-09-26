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

}  // namespace skyisle
