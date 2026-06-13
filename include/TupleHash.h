#pragma once

#include <tuple>
#include <functional>

struct TupleHash {
    std::size_t operator()(const std::tuple<int,int,int>& k) const {
        std::size_t seed = 0;
        seed ^= std::hash<int>{}(std::get<0>(k)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(std::get<1>(k)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(std::get<2>(k)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
