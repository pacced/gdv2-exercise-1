#pragma once

#include <array>
#include <glm/vec3.hpp>

struct OctreeNode {
    glm::ivec3 regionStart;
    glm::ivec3 regionEnd;

    float minValue;
    float maxValue;

    int type;
    std::array<int, 8> childIndices;
};

enum NodeType : int {
    EMPTY = 0,
    EMPTY_LEAF = 1,
    POPULATED = 2,
    POPULATED_LEAF = 3
};