#include "VolumeVisualisation.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>

#include <tuple>
#include <utility>
#include <vector>

#include "glm/fwd.hpp"
#include "glm/geometric.hpp"
#include "gris/GridCell.h"
#include "gris/TriangeMesh.h"
#include "gris/VolumeData.h"
#include "gris/mc_look_up.h"

#include "TupleHash.h"
#include "OctreeNode.h"

#include <omp.h>

constexpr float BARTH_W = 1.0f;

VolumeVisualisation::VolumeVisualisation(const gris::BoundingBox& bounding_box):
m_bounding_box(bounding_box) {
}

int VolumeVisualisation::getFlatIndex(const int x, const int y, const int z) const {
    const glm::ivec3 dimensions = m_volume_data.dimensions;

    assert(x < dimensions.x);
    assert(y < dimensions.y);
    assert(z < dimensions.z);

    const int stepY = dimensions.x;
    const int stepZ = stepY * dimensions.y;

    return x + y * stepY + z * stepZ;
};

float VolumeVisualisation::getDensity(int flatIndex) const {
    return m_volume_data.values[flatIndex];
}

float VolumeVisualisation::getDensity(int x, int y, int z) const {
    return m_volume_data.values[getFlatIndex(x, y, z)];
}

bool VolumeVisualisation::checkInside(int flatIndex) const {
    return getDensity(flatIndex) > m_iso_value;
}

bool VolumeVisualisation::checkInside(int x, int y, int z) const {
    return getDensity(x, y, z) > m_iso_value;
}

glm::vec3 VolumeVisualisation::edgeCutPosition(int x0, int y0, int z0, int x1, int y1, int z1) {
    float d0 = getDensity(x0, y0, z0);
    float d1 = getDensity(x1, y1, z1);;
    float t = (m_iso_value - d0) / (d1 - d0);
    return glm::mix(glm::vec3(x0, y0, z0), glm::vec3(x1, y1, z1), t);
}

glm::vec3 VolumeVisualisation::getEdgeCutByEdge(int x, int y, int z, int edge){
    switch(edge) {
        case gris::EDGE_0:  return edgeCutPosition(x    , y    , z    , x + 1, y    , z    );
        case gris::EDGE_1:  return edgeCutPosition(x + 1, y    , z    , x + 1, y    , z + 1);
        case gris::EDGE_2:  return edgeCutPosition(x + 1, y    , z + 1, x    , y    , z + 1);
        case gris::EDGE_3:  return edgeCutPosition(x    , y    , z    , x    , y    , z + 1);
        case gris::EDGE_4:  return edgeCutPosition(x    , y + 1, z    , x + 1, y + 1, z    );
        case gris::EDGE_5:  return edgeCutPosition(x + 1, y + 1, z    , x + 1, y + 1, z + 1);
        case gris::EDGE_6:  return edgeCutPosition(x + 1, y + 1, z + 1, x    , y + 1, z + 1);
        case gris::EDGE_7:  return edgeCutPosition(x    , y + 1, z + 1, x    , y + 1, z    );
        case gris::EDGE_8:  return edgeCutPosition(x    , y    , z    , x    , y + 1, z    );
        case gris::EDGE_9:  return edgeCutPosition(x + 1, y    , z    , x + 1, y + 1, z    );
        case gris::EDGE_10: return edgeCutPosition(x + 1, y    , z + 1, x + 1, y + 1, z + 1);
        case gris::EDGE_11: return edgeCutPosition(x    , y    , z + 1, x    , y + 1, z + 1);
        default: assert(false);
    }
}

void VolumeVisualisation::generateTriviateVolumeData(const glm::ivec3& dimensions, const gris::BoundingBox& bounding_box) {
    assert(dimensions.x >= 2);
    assert(dimensions.y >= 2);
    assert(dimensions.z >= 2);

    m_is_trivariate = true;
    m_bounding_box = bounding_box;

    // DONE(4.2): fill volume_data.values with values from triviate function
    m_volume_data.dimensions = dimensions;
    m_volume_data.values.resize(dimensions.x * dimensions.y * dimensions.z);

    // Golden Ratio
    const float phi = 1.61803398875f;
    const float sqPhi = phi * phi;

    const float w = BARTH_W;
    const float sqW = w * w;

    float scalingFactor = 8.0f / std::max(dimensions.x, std::max(dimensions.y, dimensions.z));

    int flatIndex = 0;
    for(int z = 0; z < dimensions.z; ++z) {
        float evalZ = (z - dimensions.z * 0.5f) * scalingFactor;
        float sqZ = evalZ * evalZ;

        for(int y = 0; y < dimensions.y; ++y) {
            float evalY = (y - dimensions.y * 0.5f) * scalingFactor;
            float sqY = evalY * evalY;

            for(int x = 0; x < dimensions.x; ++x, ++flatIndex) {
                float evalX = (x - dimensions.x * 0.5f) * scalingFactor;
                float sqX = evalX * evalX;

                float omniTerm = (sqX + sqY + sqZ - sqW);
                float barthEvaluation = 4 * (sqPhi * sqX - sqY) * (sqPhi * sqY - sqZ) * (sqPhi * sqZ - sqX) -
                                        (1.0f + 2.0f * phi) * (omniTerm * omniTerm) * sqW;

                m_volume_data.values[flatIndex] = barthEvaluation;
            }
        }
    }
}

void VolumeVisualisation::loadRawVolumeData(const std::string& path, const glm::ivec3& dimensions, const gris::BoundingBox& bounding_box) {
    assert(dimensions.x >= 2);
    assert(dimensions.y >= 2);
    assert(dimensions.z >= 2);

    m_is_trivariate = false;
    m_bounding_box = bounding_box;

    std::ifstream in(path, std::ios_base::in | std::ios_base::binary);

    m_volume_data.dimensions = dimensions;
    m_volume_data.values.resize(dimensions.x * dimensions.y * dimensions.z);

    // we know the bit depth (8 bits by now), so all that's left is to cast each char to a float between [0,1)
    for(int i = 0; i < dimensions.x * dimensions.y * dimensions.z; ++i) {
        float v = static_cast<unsigned char>(in.get()) / 256.0f;
        m_volume_data.values[i] = v;
    }
}

void VolumeVisualisation::generateMesh(float iso_value, bool dual, bool grid_snapping, float grid_snapping_distance, bool optimize_mesh) {
    m_iso_value = iso_value;

    clock_t start = clock();
    std::cout << "Mesh Generation started ..." << std::endl;

    if(dual) {
        dualMarchingCubes();
    } else {
        marchingCubesOctree();
    }

    clock_t stop = clock();
    std::cout << "Mesh Generation finised after " << static_cast<float>(stop - start) / CLOCKS_PER_SEC << " seconds" << std::endl;

    if(grid_snapping) {
        start = clock();
        std::cout << "\nGrid Snapping started ..." << std::endl;

        snapToGrid(grid_snapping_distance);

        stop = clock();
        std::cout << "Grid Snapping finised after " << static_cast<float>(stop - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
    }

    if(optimize_mesh) {
        start = clock();
        std::cout << "\nMesh Optimization started ..." << std::endl;

        cleanUpTriangleSoup();

        stop = clock();
        std::cout << "Mesh Optimization finised after " << static_cast<float>(stop - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
    }


    std::cout << "\nNormal Calculation started ..." << std::endl;
    double ompStart = omp_get_wtime();
    calculateNormals();
    double ompStop = omp_get_wtime();
    std::cout << "Normal Calculation finised after " << (ompStop - ompStart) << " seconds" << std::endl;


    std::cout << "Resulting Mesh has: " << m_mesh.vertices.size() << " vertices, " << m_mesh.triangles.size() << " triangles.\n"
              << std::endl;
}

void VolumeVisualisation::drawMesh(QOpenGLFunctions_2_1* f) const {
    // DONE(4.1): Draw mesh
    glm::vec3 min = m_bounding_box.min;
    glm::vec3 delta = (m_bounding_box.max - m_bounding_box.min) /
                    glm::vec3{
                        static_cast<float>(m_volume_data.dimensions.x - 1),
                        static_cast<float>(m_volume_data.dimensions.y - 1),
                        static_cast<float>(m_volume_data.dimensions.z - 1),
                    };


    glm::vec4 color1 = {0.2f, 0.2f, 0.5f, 1.f};
    glm::vec4 color2 = {0.4f, 0.9f, 0.2f, 1.f};

    f->glBegin(GL_TRIANGLES);
        for(const glm::ivec3& t : m_mesh.triangles) {
            glm::vec3 v0 = min + delta * m_mesh.vertices[t.x];
            glm::vec3 v1 = min + delta * m_mesh.vertices[t.y];
            glm::vec3 v2 = min + delta * m_mesh.vertices[t.z];

            glm::vec3 n0 = m_mesh.normals[t.x];
            glm::vec3 n1 = m_mesh.normals[t.y];
            glm::vec3 n2 = m_mesh.normals[t.z];

            f->glColor3fv(&color1.x);
            f->glNormal3fv(&n0.x);
            f->glVertex3fv(&v0.x);

            f->glNormal3fv(&n1.x);
            f->glVertex3fv(&v1.x);

            f->glNormal3fv(&n2.x);
            f->glVertex3fv(&v2.x);

            // std::swap(color1, color2);
        }
    f->glEnd();
}

void VolumeVisualisation::drawNormals(QOpenGLFunctions_2_1* f) const {
    // DONE(4.3): Draw normals
    glm::vec3 min = m_bounding_box.min;
    glm::vec3 delta = (m_bounding_box.max - m_bounding_box.min) /
                    glm::vec3{
                        static_cast<float>(m_volume_data.dimensions.x - 1),
                        static_cast<float>(m_volume_data.dimensions.y - 1),
                        static_cast<float>(m_volume_data.dimensions.z - 1),
                    };


    const float scale = 0.055f;

    f->glBegin(GL_LINES);
    for(size_t i = 0; i < m_mesh.vertices.size(); ++i){
        glm::vec3 v = min + delta * m_mesh.vertices[i];
        glm::vec3 n = m_mesh.normals[i];

        glm::vec3 ray = v + n * scale;

        f->glVertex3fv(&v.x);
        f->glVertex3fv(&ray.x);
    }
    f->glEnd();
}

void VolumeVisualisation::drawVolumeData(QOpenGLFunctions_2_1* f) const {
    glm::vec3 min = m_bounding_box.min;
    glm::vec3 delta = (m_bounding_box.max - m_bounding_box.min) /
        glm::vec3{
            static_cast<float>(m_volume_data.dimensions.x - 1),
            static_cast<float>(m_volume_data.dimensions.y - 1),
            static_cast<float>(m_volume_data.dimensions.z - 1),
        };

    f->glPointSize(6.0f);
    f->glBegin(GL_POINTS);
    for(int x = 0; x < m_volume_data.dimensions.x; x++) {
        for(int y = 0; y < m_volume_data.dimensions.y; y++) {
            for(int z = 0; z < m_volume_data.dimensions.z; z++) {
                if(checkInside(x, y, z)) {
                    glm::vec3 point = min + delta * glm::vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
                    f->glVertex3fv(reinterpret_cast<const float*>(&point));
                }
            }
        }
    }
    f->glEnd();
}

void VolumeVisualisation::drawOctree(QOpenGLFunctions_2_1* f) const {
    glm::vec3 min = m_bounding_box.min;
    glm::vec3 delta = (m_bounding_box.max - m_bounding_box.min) /
        glm::vec3{
            static_cast<float>(m_volume_data.dimensions.x - 1),
            static_cast<float>(m_volume_data.dimensions.y - 1),
            static_cast<float>(m_volume_data.dimensions.z - 1),
        };

    f->glDisable(GL_LIGHTING);

    f->glEnable(GL_BLEND);
    f->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    f->glEnable(GL_CULL_FACE);
    f->glCullFace(GL_BACK);

    f->glEnable(GL_POLYGON_OFFSET_FILL);
    f->glPolygonOffset(1.0f, 1.0f);

    f->glBegin(GL_TRIANGLES);
    f->glColor4f(0.0f, 0.9f, 0.2f, 0.4f);
    for(const OctreeNode& node : m_octree) {
        if(node.type == EMPTY || node.type == EMPTY_LEAF || node.type == POPULATED) {
            continue;
        }

        glm::vec3 start = node.regionStart;
        glm::vec3 end = node.regionEnd;

        start = min + delta * start;
        end = min + delta * end;

        // FRONT
        f->glVertex3f(start.x, start.y, end.z);
        f->glVertex3f(end.x, start.y, end.z);
        f->glVertex3f(end.x, end.y, end.z);

        f->glVertex3f(start.x, start.y, end.z);
        f->glVertex3f(end.x, end.y, end.z);
        f->glVertex3f(start.x, end.y, end.z);

        // BACK
        f->glVertex3f(end.x, start.y, start.z);
        f->glVertex3f(start.x, start.y, start.z);
        f->glVertex3f(start.x, end.y, start.z);

        f->glVertex3f(end.x, start.y, start.z);
        f->glVertex3f(start.x, end.y, start.z);
        f->glVertex3f(end.x, end.y, start.z);

        // LEFT
        f->glVertex3f(start.x, start.y, start.z);
        f->glVertex3f(start.x, start.y, end.z);
        f->glVertex3f(start.x, end.y, end.z);

        f->glVertex3f(start.x, start.y, start.z);
        f->glVertex3f(start.x, end.y, end.z);
        f->glVertex3f(start.x, end.y, start.z);

        // RIGHT
        f->glVertex3f(end.x, start.y, end.z);
        f->glVertex3f(end.x, start.y, start.z);
        f->glVertex3f(end.x, end.y, start.z);

        f->glVertex3f(end.x, start.y, end.z);
        f->glVertex3f(end.x, end.y, start.z);
        f->glVertex3f(end.x, end.y, end.z);

        // TOP
        f->glVertex3f(start.x, end.y, end.z);
        f->glVertex3f(end.x, end.y, end.z);
        f->glVertex3f(end.x, end.y, start.z);

        f->glVertex3f(start.x, end.y, end.z);
        f->glVertex3f(end.x, end.y, start.z);
        f->glVertex3f(start.x, end.y, start.z);

        // BOTTOM
        f->glVertex3f(start.x, start.y, start.z);
        f->glVertex3f(end.x, start.y, start.z);
        f->glVertex3f(end.x, start.y, end.z);

        f->glVertex3f(start.x, start.y, start.z);
        f->glVertex3f(end.x, start.y, end.z);
        f->glVertex3f(start.x, start.y, end.z);
    }
    f->glEnd();

    // Wireframe
    f->glBegin(GL_LINES);
    for (const OctreeNode& node : m_octree) {
        if(node.type == EMPTY_LEAF || node.type == EMPTY) {
            continue;
        }

        if(node.type == POPULATED_LEAF) {
            f->glColor4f(0.0f, 0.0f, 0.0f, 0.9f);
        } else {
            f->glColor4f(0.0f, 0.0f, 0.0f, 0.4f);
        }

        glm::vec3 start = node.regionStart;
        glm::vec3 end = node.regionEnd;

        start = min + delta * start;
        end = min + delta * end;

        f->glVertex3f(start.x, start.y, start.z);
        f->glVertex3f(end.x, start.y, start.z);

        f->glVertex3f(end.x, start.y, start.z);
        f->glVertex3f(end.x, start.y, end.z);

        f->glVertex3f(end.x, start.y, end.z);
        f->glVertex3f(start.x, start.y, end.z);

        f->glVertex3f(start.x, start.y, end.z);
        f->glVertex3f(start.x, start.y, start.z);

        f->glVertex3f(start.x, end.y, start.z);
        f->glVertex3f(end.x, end.y, start.z);

        f->glVertex3f(end.x, end.y, start.z);
        f->glVertex3f(end.x, end.y, end.z);

        f->glVertex3f(end.x, end.y, end.z);
        f->glVertex3f(start.x, end.y, end.z);

        f->glVertex3f(start.x, end.y, end.z);
        f->glVertex3f(start.x, end.y, start.z);

        f->glVertex3f(start.x, start.y, start.z);
        f->glVertex3f(start.x, end.y, start.z);

        f->glVertex3f(end.x, start.y, start.z);
        f->glVertex3f(end.x, end.y, start.z);

        f->glVertex3f(end.x, start.y, end.z);
        f->glVertex3f(end.x, end.y, end.z);

        f->glVertex3f(start.x, start.y, end.z);
        f->glVertex3f(start.x, end.y, end.z);
    }
    f->glEnd();

    f->glDisable(GL_CULL_FACE);
    f->glDisable(GL_BLEND);
    f->glDisable(GL_POLYGON_OFFSET_FILL);
}

gris::TriangleMesh VolumeVisualisation::poligonize(const gris::GridCell& grid_cell, float iso_value) {
    int index = 0;
    index |= (grid_cell.values[0] < iso_value ? gris::CORNER_0 : 0);
    index |= (grid_cell.values[1] < iso_value ? gris::CORNER_1 : 0);
    index |= (grid_cell.values[2] < iso_value ? gris::CORNER_2 : 0);
    index |= (grid_cell.values[3] < iso_value ? gris::CORNER_3 : 0);
    index |= (grid_cell.values[4] < iso_value ? gris::CORNER_4 : 0);
    index |= (grid_cell.values[5] < iso_value ? gris::CORNER_5 : 0);
    index |= (grid_cell.values[6] < iso_value ? gris::CORNER_6 : 0);
    index |= (grid_cell.values[7] < iso_value ? gris::CORNER_7 : 0);
    const glm::ivec3* triangles = gris::triangle_table[index];
    gris::TriangleMesh result;

    for(int i = 0; i < 5; i++) {
        if(triangles[i].x == -1) {
            break;
        }
        const glm::ivec3 triangle = triangles[i];
        result.triangles.push_back({
            result.vertices.size(),
            result.vertices.size() + 1,
            result.vertices.size() + 2,
        });

        for(int edge : {triangle.x, triangle.y, triangle.z}) {
            float value_a;
            glm::vec3 corner_a;
            float value_b;
            glm::vec3 corner_b;
            switch(edge) {
            case gris::EDGE_0:
                value_a = grid_cell.values[0];
                corner_a = {grid_cell.volume.min.x, grid_cell.volume.min.y, grid_cell.volume.min.z};
                value_b = grid_cell.values[1];
                corner_b = {grid_cell.volume.max.x, grid_cell.volume.min.y, grid_cell.volume.min.z};
                break;
            case gris::EDGE_1:
                value_a = grid_cell.values[1];
                corner_a = {grid_cell.volume.max.x, grid_cell.volume.min.y, grid_cell.volume.min.z};
                value_b = grid_cell.values[2];
                corner_b = {grid_cell.volume.max.x, grid_cell.volume.min.y, grid_cell.volume.max.z};
                break;
            case gris::EDGE_2:
                value_a = grid_cell.values[2];
                corner_a = {grid_cell.volume.max.x, grid_cell.volume.min.y, grid_cell.volume.max.z};
                value_b = grid_cell.values[3];
                corner_b = {grid_cell.volume.min.x, grid_cell.volume.min.y, grid_cell.volume.max.z};
                break;
            case gris::EDGE_3:
                value_a = grid_cell.values[0];
                corner_a = {grid_cell.volume.min.x, grid_cell.volume.min.y, grid_cell.volume.min.z};
                value_b = grid_cell.values[3];
                corner_b = {grid_cell.volume.min.x, grid_cell.volume.min.y, grid_cell.volume.max.z};
                break;
            case gris::EDGE_4:
                value_a = grid_cell.values[4];
                corner_a = {grid_cell.volume.min.x, grid_cell.volume.max.y, grid_cell.volume.min.z};
                value_b = grid_cell.values[5];
                corner_b = {grid_cell.volume.max.x, grid_cell.volume.max.y, grid_cell.volume.min.z};
                break;
            case gris::EDGE_5:
                value_a = grid_cell.values[5];
                corner_a = {grid_cell.volume.max.x, grid_cell.volume.max.y, grid_cell.volume.min.z};
                value_b = grid_cell.values[6];
                corner_b = {grid_cell.volume.max.x, grid_cell.volume.max.y, grid_cell.volume.max.z};
                break;
            case gris::EDGE_6:
                value_a = grid_cell.values[6];
                corner_a = {grid_cell.volume.max.x, grid_cell.volume.max.y, grid_cell.volume.max.z};
                value_b = grid_cell.values[7];
                corner_b = {grid_cell.volume.min.x, grid_cell.volume.max.y, grid_cell.volume.max.z};
                break;
            case gris::EDGE_7:
                value_a = grid_cell.values[4];
                corner_a = {grid_cell.volume.min.x, grid_cell.volume.max.y, grid_cell.volume.min.z};
                value_b = grid_cell.values[7];
                corner_b = {grid_cell.volume.min.x, grid_cell.volume.max.y, grid_cell.volume.max.z};
                break;
            case gris::EDGE_8:
                value_a = grid_cell.values[0];
                corner_a = {grid_cell.volume.min.x, grid_cell.volume.min.y, grid_cell.volume.min.z};
                value_b = grid_cell.values[4];
                corner_b = {grid_cell.volume.min.x, grid_cell.volume.max.y, grid_cell.volume.min.z};
                break;
            case gris::EDGE_9:
                value_a = grid_cell.values[1];
                corner_a = {grid_cell.volume.max.x, grid_cell.volume.min.y, grid_cell.volume.min.z};
                value_b = grid_cell.values[5];
                corner_b = {grid_cell.volume.max.x, grid_cell.volume.max.y, grid_cell.volume.min.z};
                break;
            case gris::EDGE_10:
                value_a = grid_cell.values[2];
                corner_a = {grid_cell.volume.max.x, grid_cell.volume.min.y, grid_cell.volume.max.z};
                value_b = grid_cell.values[6];
                corner_b = {grid_cell.volume.max.x, grid_cell.volume.max.y, grid_cell.volume.max.z};
                break;
            case gris::EDGE_11:
                value_a = grid_cell.values[3];
                corner_a = {grid_cell.volume.min.x, grid_cell.volume.min.y, grid_cell.volume.max.z};
                value_b = grid_cell.values[7];
                corner_b = {grid_cell.volume.min.x, grid_cell.volume.max.y, grid_cell.volume.max.z};
                break;
            default:
                assert(false);
            }
            result.vertices.push_back(glm::mix(corner_a, corner_b, (value_a - iso_value) / (value_a - value_b)));
        }
    }
    return result;
}

std::tuple<int, float, float> VolumeVisualisation::buildNode(const glm::ivec3 start, const glm::ivec3 end) {
    const glm::ivec3 span = end - start;

    if(span.x <= 0 || span.y <= 0 || span.z <= 0) {
        return { -1, 0.0f, 0.0f };
    }

    OctreeNode node;
    node.regionStart = start;
    node.regionEnd = end;
    node.childIndices.fill(-1);

    auto dim = m_volume_data.dimensions;

    const int minSize = 8;
    // TODO make it so that something like 1x12x24 is still split
    if(span.x <= minSize && span.y <= minSize && span.z <= minSize) {
        // Leaf, do not reduce anymore

        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();

        for(int z = start.z; z <= end.z; ++z) {
            for(int y = start.y; y <= end.y; ++y) {
                for(int x = start.x; x <= end.x; ++x) {
                    float gridPointValue = getDensity(x, y, z);

                    min = std::min(min, gridPointValue);
                    max = std::max(max, gridPointValue);
                }
            }
        }

        node.minValue = min;
        node.maxValue = max;

        // Does not result in a surface, can prune
        if(max < m_iso_value || min > m_iso_value) {
            node.type = EMPTY_LEAF;
            m_octree.push_back(node);
            return { -2, min, max };
        }

        int index = m_octree.size();
        node.type = POPULATED_LEAF;
        m_octree.push_back(node);
        return { index, min, max };
    }


    const glm::ivec3 middle = (start + end) / 2;

    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();

    bool anyValidChild = false;
    for(int octant = 0; octant < 8; ++octant) {
        glm::ivec3 childStart;
        glm::ivec3 childEnd;

        childStart.x = (octant & 1) ? middle.x : start.x;
        childEnd.x   = (octant & 1) ? end.x    : middle.x;

        childStart.y = (octant & 2) ? middle.y : start.y;
        childEnd.y   = (octant & 2) ? end.y    : middle.y;

        childStart.z = (octant & 4) ? middle.z : start.z;
        childEnd.z   = (octant & 4) ? end.z    : middle.z;


        std::tuple<int, float, float> childNode = buildNode(childStart, childEnd);

        int childIndex = std::get<0>(childNode);
        float childMin = std::get<1>(childNode);
        float childMax = std::get<2>(childNode);

        node.childIndices[octant] = childIndex;

        min = std::min(min, childMin);
        max = std::max(max, childMax);

        if(childIndex >= 0) {
            anyValidChild = true;
        }
    }

    node.minValue = min;
    node.maxValue = max;

    if(!anyValidChild || (max < m_iso_value || min > m_iso_value)) {
        node.type = EMPTY;
        m_octree.push_back(node);
        return { -2, min, max };
    }

    node.type = POPULATED;

    int selfIndex = m_octree.size();
    m_octree.push_back(node);

    return { selfIndex, min, max };
}

void VolumeVisualisation::buildOctree() {
    m_octree.clear();

    auto dim = m_volume_data.dimensions;

    glm::ivec3 start = {0, 0, 0};
    glm::ivec3 end = {dim.x - 1, dim.y - 1, dim.z - 1};

    buildNode(start, end);
}



void VolumeVisualisation::marchingCubesOctree() {
    m_mesh.vertices.clear();
    m_mesh.normals.clear();
    m_mesh.triangles.clear();

    const glm::ivec3& dim = m_volume_data.dimensions;

    buildOctree();

    for(OctreeNode& node : m_octree) {
        if(node.type != POPULATED_LEAF) {
            continue;
        }

        const glm::ivec3 start = node.regionStart;
        const glm::ivec3 end = node.regionEnd;

        for(int z = start.z; z < end.z; ++z) {
            for(int y = start.y; y < end.y; ++y) {
                for(int x = start.x; x < end.x; ++x) {
                    int caseIndex = 0;
                    caseIndex |= (checkInside(x    , y    , z    ) ? gris::CORNER_0 : 0);
                    caseIndex |= (checkInside(x + 1, y    , z    ) ? gris::CORNER_1 : 0);
                    caseIndex |= (checkInside(x + 1, y    , z + 1) ? gris::CORNER_2 : 0);
                    caseIndex |= (checkInside(x    , y    , z + 1) ? gris::CORNER_3 : 0);
                    caseIndex |= (checkInside(x    , y + 1, z    ) ? gris::CORNER_4 : 0);
                    caseIndex |= (checkInside(x + 1, y + 1, z    ) ? gris::CORNER_5 : 0);
                    caseIndex |= (checkInside(x + 1, y + 1, z + 1) ? gris::CORNER_6 : 0);
                    caseIndex |= (checkInside(x    , y + 1, z + 1) ? gris::CORNER_7 : 0);

                    const glm::ivec3* triangles = gris::triangle_table[caseIndex];

                    for (int i = 0; i < 5; ++i) {
                        const int* triangleEdges = &triangles[i].x;
                        if (triangleEdges[0] == -1) {
                            break;
                        }

                        int baseVertexIdx = m_mesh.vertices.size();
                        m_mesh.triangles.push_back({ baseVertexIdx, baseVertexIdx + 1, baseVertexIdx + 2 });

                        for(int j = 0; j < 3; ++j) {
                            int edge = triangleEdges[j];
                            const glm::vec3 vertex = getEdgeCutByEdge(x, y, z, edge);

                            m_mesh.vertices.push_back(vertex);
                        }
                    }
                }
            }
        }
    }
}

void VolumeVisualisation::marchingCubes2() {
    m_mesh.vertices.clear();
    m_mesh.normals.clear();
    m_mesh.triangles.clear();

    auto dim = m_volume_data.dimensions;
    size_t totalGridPoints = dim.x * dim.y * dim.z;

    std::vector<int> insideFlags(totalGridPoints);
    int flagIndex = 0;

    // Classify grid points
    for(int z = 0; z < dim.z; ++z) {
        for(int y = 0; y < dim.y; ++y) {
            for(int x = 0; x < dim.x; ++x) {
                insideFlags[flagIndex] = (checkInside(x, y, z)) ? 1 : 0;
            }
        }
    }

    std::vector<int> edgeX(totalGridPoints, -1);
    std::vector<int> edgeY(totalGridPoints, -1);
    std::vector<int> edgeZ(totalGridPoints, -1);

    std::vector<glm::vec3> verticesX;
    std::vector<glm::vec3> verticesY;
    std::vector<glm::vec3> verticesZ;

    // X edges
    for (int z = 0; z < dim.z; z++) {
        for (int y = 0; y < dim.y; y++) {
            for (int x = 0; x < dim.x - 1; x++) {
                int i0 = getFlatIndex(x    , y, z);
                int i1 = getFlatIndex(x + 1, y, z);

                if (insideFlags[i0] != insideFlags[i1]) {
                    glm::vec3 edgeCut = edgeCutPosition(x, y, z, x + 1, y, z);

                    edgeX[i0] = verticesX.size();
                    verticesX.push_back(edgeCut);
                }
            }
        }
    }

    // Y edges
    for (int z = 0; z < dim.z; z++) {
        for (int y = 0; y < dim.y - 1; y++) {
            for (int x = 0; x < dim.x; x++) {
                int i0 = getFlatIndex(x, y    , z);
                int i1 = getFlatIndex(x, y + 1, z);

                if (insideFlags[i0] != insideFlags[i1]) {
                    glm::vec3 edgeCut = edgeCutPosition(x, y, z, x, y + 1, z);

                    edgeY[i0] = verticesY.size();
                    verticesY.push_back(edgeCut);
                }
            }
        }
    }

    // Z edges
    for (int z = 0; z < dim.z - 1; z++) {
        for (int y = 0; y < dim.y; y++) {
            for (int x = 0; x < dim.x; x++) {
                int i0 = getFlatIndex(x, y, z    );
                int i1 = getFlatIndex(x, y, z + 1);

                if (insideFlags[i0] != insideFlags[i1]) {
                    glm::vec3 edgeCut = edgeCutPosition(x, y, z, x, y, z + 1);

                    edgeZ[i0] = verticesZ.size();
                    verticesZ.push_back(edgeCut);
                }
            }
        }
    }

    int offsetX = 0;
    int offsetY = verticesX.size();
    int offsetZ = verticesX.size() + verticesY.size();

    m_mesh.vertices.insert(m_mesh.vertices.end(), verticesX.begin(), verticesX.end());
    m_mesh.vertices.insert(m_mesh.vertices.end(), verticesY.begin(), verticesY.end());
    m_mesh.vertices.insert(m_mesh.vertices.end(), verticesZ.begin(), verticesZ.end());

    // Generate triangles
    for (int z = 0; z < dim.z - 1; z++) {
        for (int y = 0; y < dim.y - 1; y++) {
            for (int x = 0; x < dim.x - 1; x++) {
                int caseIdx = 0;
                if (insideFlags[getFlatIndex(x    , y    , z    )]) caseIdx |= gris::CORNER_0;
                if (insideFlags[getFlatIndex(x + 1, y    , z    )]) caseIdx |= gris::CORNER_1;
                if (insideFlags[getFlatIndex(x + 1, y    , z + 1)]) caseIdx |= gris::CORNER_2;
                if (insideFlags[getFlatIndex(x    , y    , z + 1)]) caseIdx |= gris::CORNER_3;
                if (insideFlags[getFlatIndex(x    , y + 1, z    )]) caseIdx |= gris::CORNER_4;
                if (insideFlags[getFlatIndex(x + 1, y + 1, z    )]) caseIdx |= gris::CORNER_5;
                if (insideFlags[getFlatIndex(x + 1, y + 1, z + 1)]) caseIdx |= gris::CORNER_6;
                if (insideFlags[getFlatIndex(x    , y + 1, z + 1)]) caseIdx |= gris::CORNER_7;

                const glm::ivec3* triangles = gris::triangle_table[caseIdx];

                for (int i = 0; i < 5; i++) {
                    if (triangles[i].x == -1) {
                        break;
                    }

                    int indices[3];
                    for (int j = 0; j < 3; j++) {
                        int edge = (&triangles[i].x)[j];
                        switch (edge) {
                            case gris::EDGE_0:  indices[j] = offsetX + edgeX[getFlatIndex(x    , y    , z    )]; break;
                            case gris::EDGE_1:  indices[j] = offsetZ + edgeZ[getFlatIndex(x + 1, y    , z    )]; break;
                            case gris::EDGE_2:  indices[j] = offsetX + edgeX[getFlatIndex(x    , y    , z + 1)]; break;
                            case gris::EDGE_3:  indices[j] = offsetZ + edgeZ[getFlatIndex(x    , y    , z    )]; break;
                            case gris::EDGE_4:  indices[j] = offsetX + edgeX[getFlatIndex(x    , y + 1, z    )]; break;
                            case gris::EDGE_5:  indices[j] = offsetZ + edgeZ[getFlatIndex(x + 1, y + 1, z    )]; break;
                            case gris::EDGE_6:  indices[j] = offsetX + edgeX[getFlatIndex(x    , y + 1, z + 1)]; break;
                            case gris::EDGE_7:  indices[j] = offsetZ + edgeZ[getFlatIndex(x    , y + 1, z    )]; break;
                            case gris::EDGE_8:  indices[j] = offsetY + edgeY[getFlatIndex(x    , y    , z    )]; break;
                            case gris::EDGE_9:  indices[j] = offsetY + edgeY[getFlatIndex(x + 1, y    , z    )]; break;
                            case gris::EDGE_10: indices[j] = offsetY + edgeY[getFlatIndex(x + 1, y    , z + 1)]; break;
                            case gris::EDGE_11: indices[j] = offsetY + edgeY[getFlatIndex(x    , y    , z + 1)]; break;
                            default: assert(false);
                        }
                    }

                    m_mesh.triangles.push_back({indices[0], indices[1], indices[2]});
                }
            }
        }
    }
}

void VolumeVisualisation::marchingCubes() {
    m_mesh.vertices.clear();
    m_mesh.normals.clear();
    m_mesh.triangles.clear();

    // DONE(4.1): implement marching cubes algorithm
    auto dim = m_volume_data.dimensions;

    int stepX = 1;
    int stepY = stepX * dim.x;
    int stepZ = stepY * dim.y;

    for(int z = 0; z < dim.z - 1; ++z){
        int offsetZ0 = z * stepZ;
        int offsetZ1 = (z + 1) * stepZ;

        for(int y = 0; y < dim.y - 1; ++y){
            int offsetY0 = y * stepY;
            int offsetY1 = (y + 1) * stepY;

            int offsetYZ0 = offsetZ0 + offsetY0;
            int offsetYZ1 = offsetZ1 + offsetY0;

            int offsetYZ2 = offsetZ0 + offsetY1;
            int offsetYZ3 = offsetZ1 + offsetY1;

            for(int x = 0; x < dim.x - 1; ++x){
                gris::GridCell voxel;

                voxel.values[0] = getDensity(offsetYZ0 + x        );
                voxel.values[1] = getDensity(offsetYZ0 + x + stepX);
                voxel.values[2] = getDensity(offsetYZ1 + x + stepX);
                voxel.values[3] = getDensity(offsetYZ1 + x        );
                voxel.values[4] = getDensity(offsetYZ2 + x        );
                voxel.values[5] = getDensity(offsetYZ2 + x + stepX);
                voxel.values[6] = getDensity(offsetYZ3 + x + stepX);
                voxel.values[7] = getDensity(offsetYZ3 + x        );

                glm::vec3 min(
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(z)
                );

                glm::vec3 max(
                    static_cast<float>(x + 1),
                    static_cast<float>(y + 1),
                    static_cast<float>(z + 1)
                );


                voxel.volume = gris::BoundingBox(min, max);

                auto voxelMesh = poligonize(voxel, m_iso_value);

                std::size_t vertexOffset = m_mesh.vertices.size();
                m_mesh.vertices.insert(
                    std::end(m_mesh.vertices),
                    std::begin(voxelMesh.vertices),
                    std::end(voxelMesh.vertices)
                );

                for(const glm::ivec3& t : voxelMesh.triangles){
                    m_mesh.triangles.push_back({
                        vertexOffset + t.z,
                        vertexOffset + t.y,
                        vertexOffset + t.x
                    });
                }
            }
        }
    }

    // // DONE(4.3): calculate surface normals
    // calculateNormals();
    // Moved into generateMesh()
}

void VolumeVisualisation::dualMarchingCubes() {
    m_mesh.vertices.clear();
    m_mesh.normals.clear();
    m_mesh.triangles.clear();

    // DONE(4.4): implement dual marching cubes algorithm
    const glm::ivec3& dim = m_volume_data.dimensions;
    const size_t totalGridPoints = dim.x * dim.y * dim.z;

    buildOctree();

    std::vector<glm::ivec3> validVoxels;
    for(const OctreeNode& node : m_octree){
        if(node.type != POPULATED_LEAF){
            continue;
        }

        const glm::ivec3& start = node.regionStart;
        const glm::ivec3& end = node.regionEnd;

        for(int z = start.z; z < end.z; ++z) {
            for(int y = start.y; y < end.y; ++y) {
                for(int x = start.x; x < end.x; ++x) {
                    validVoxels.push_back({x, y, z});
                }
            }
        }
    }

    const size_t voxelCount = validVoxels.size();

    std::vector<std::array<int, 4>> centroidIndicesByVoxel(totalGridPoints, {-1, -1, -1, -1});
    std::vector<std::array<int, 4>> edgeAssociationByVoxel(totalGridPoints, {0, 0, 0, 0});

    for(int i = 0; i < voxelCount; ++i) {
        const glm::ivec3& coords = validVoxels[i];

        int x = coords.x;
        int y = coords.y;
        int z = coords.z;

        int caseIdx = 0;
        caseIdx |= checkInside(x    , y    , z    ) ? gris::CORNER_0 : 0;
        caseIdx |= checkInside(x + 1, y    , z    ) ? gris::CORNER_1 : 0;
        caseIdx |= checkInside(x + 1, y    , z + 1) ? gris::CORNER_2 : 0;
        caseIdx |= checkInside(x    , y    , z + 1) ? gris::CORNER_3 : 0;
        caseIdx |= checkInside(x    , y + 1, z    ) ? gris::CORNER_4 : 0;
        caseIdx |= checkInside(x + 1, y + 1, z    ) ? gris::CORNER_5 : 0;
        caseIdx |= checkInside(x + 1, y + 1, z + 1) ? gris::CORNER_6 : 0;
        caseIdx |= checkInside(x    , y + 1, z + 1) ? gris::CORNER_7 : 0;

        const int* combinations = gris::dual_points_list[caseIdx];
        for(int j = 0; j < 4; ++j) {
            int mask = combinations[j];
            if(mask == 0) {
                break;
            }

            glm::vec3 centroid(0.0f);
            int count = 0;

            for(int e = 0; e < 12; ++e) {
                int edgeBit = 1 << e;

                if (mask & edgeBit) {
                    centroid += getEdgeCutByEdge(x, y, z, edgeBit);
                    ++count;
                }
            }

            centroid /= (float)count;

            int flatIndex = getFlatIndex(x, y, z);

            centroidIndicesByVoxel[flatIndex][j] = m_mesh.vertices.size();
            edgeAssociationByVoxel[flatIndex][j] = mask;
            m_mesh.vertices.push_back(centroid);
        }
    }

    auto getDualVertexIndex = [&](int x, int y, int z, int edge) {
        int vi = getFlatIndex(x, y, z);

        std::array<int, 4>& masks = edgeAssociationByVoxel[vi];
        for(int i = 0; i < 4; ++i) {
            int mask = masks[i];
            if(mask == 0){
                break;
            }

            if (mask & edge){
                return centroidIndicesByVoxel[vi][i];
            }
        }

        return -1;
    };

    for(int i = 0; i < voxelCount; ++i) {
        const glm::ivec3& coords = validVoxels[i];

        int x = coords.x;
        int y = coords.y;
        int z = coords.z;

        if(z > 0 && y > 0) {
            const bool entering = checkInside(x, y, z) && !checkInside(x + 1, y, z);
            const bool exiting = !checkInside(x, y, z) &&  checkInside(x + 1, y, z);

            if(entering || exiting) {
                int xvi0 = getDualVertexIndex(x, y    , z    , gris::EDGE_0);
                int xvi1 = getDualVertexIndex(x, y - 1, z    , gris::EDGE_4);
                int xvi2 = getDualVertexIndex(x, y - 1, z - 1, gris::EDGE_6);
                int xvi3 = getDualVertexIndex(x, y    , z - 1, gris::EDGE_2);

                if(xvi0 == -1 || xvi1 == -1 || xvi2 == -1 || xvi3 == -1) {
                    continue;
                }

                if(entering) {
                    m_mesh.triangles.push_back({xvi0, xvi1, xvi2});
                    m_mesh.triangles.push_back({xvi0, xvi2, xvi3});
                } else {
                    m_mesh.triangles.push_back({xvi0, xvi3, xvi2});
                    m_mesh.triangles.push_back({xvi0, xvi2, xvi1});
                }
            }
        }

        if(z > 0 && x > 0){
            const bool entering = checkInside(x, y, z) && !checkInside(x, y + 1, z);
            const bool exiting = !checkInside(x, y, z) &&  checkInside(x, y + 1, z);

            if(entering || exiting) {
                int yvi0 = getDualVertexIndex(x    , y, z    , gris::EDGE_8);
                int yvi1 = getDualVertexIndex(x    , y, z - 1, gris::EDGE_11);
                int yvi2 = getDualVertexIndex(x - 1, y, z - 1, gris::EDGE_10);
                int yvi3 = getDualVertexIndex(x - 1, y, z    , gris::EDGE_9);

                if(yvi0 == -1 || yvi1 == -1 || yvi2 == -1 || yvi3 == -1) {
                    continue;
                }

                if(entering) {
                    m_mesh.triangles.push_back({yvi0, yvi1, yvi2});
                    m_mesh.triangles.push_back({yvi0, yvi2, yvi3});
                } else {
                    m_mesh.triangles.push_back({yvi0, yvi3, yvi2});
                    m_mesh.triangles.push_back({yvi0, yvi2, yvi1});
                }
            }
        }

        if(x > 0 && y > 0){
            const bool entering = checkInside(x, y, z) && !checkInside(x, y, z + 1);
            const bool exiting = !checkInside(x, y, z) &&  checkInside(x, y, z + 1);

            if(entering || exiting) {
                int zvi0 = getDualVertexIndex(x    , y    , z, gris::EDGE_3);
                int zvi1 = getDualVertexIndex(x - 1, y    , z, gris::EDGE_1);
                int zvi2 = getDualVertexIndex(x - 1, y - 1, z, gris::EDGE_5);
                int zvi3 = getDualVertexIndex(x    , y - 1, z, gris::EDGE_7);

                if(zvi0 == -1 || zvi1 == -1 || zvi2 == -1 || zvi3 == -1) {
                    continue;
                }

                if(entering) {
                    m_mesh.triangles.push_back({zvi0, zvi1, zvi2});
                    m_mesh.triangles.push_back({zvi0, zvi2, zvi3});
                } else {
                    m_mesh.triangles.push_back({zvi0, zvi3, zvi2});
                    m_mesh.triangles.push_back({zvi0, zvi2, zvi1});
                }
            }
        }
    }
}

void VolumeVisualisation::calculateNormals() {
    if(m_mesh.vertices.size() == 0){
        return;
    }

    if(m_is_trivariate) {
        calculateNormalsAnalytical();
    } else {
        calculateNormalsFiniteDifferences();
    }
}

void VolumeVisualisation::calculateNormalsFiniteDifferences() {
    auto dim = m_volume_data.dimensions;
    const float delta = 1.0f;

    std::vector<glm::vec3> gradientCache(dim.x * dim.y * dim.z);

    auto gradient = [&](int x, int y, int z) {
        // m: minus, p: plus
        int xm = std::max(x - 1, 0);
        int xp = std::min(x + 1, dim.x - 1);

        int ym = std::max(y - 1, 0);
        int yp = std::min(y + 1, dim.y - 1);

        int zm = std::max(z - 1, 0);
        int zp = std::min(z + 1, dim.z - 1);

        return glm::vec3(
            (getDensity(xp, y , z ) - getDensity(xm, y , z )) / (2.0f * delta),
                         (getDensity(x , yp, z ) - getDensity(x , ym, z )) / (2.0f * delta),
                         (getDensity(x , y , zp) - getDensity(x , y , zm)) / (2.0f * delta)
        );
    };

    auto cachedGradient = [&](int x, int y, int z) {
        return gradientCache[z * dim.y * dim.x + y * dim.x + x];
    };

    #pragma omp parallel for collapse(3) schedule(static)
    for(int z = 0; z < dim.z; ++z) {
        for(int y = 0; y < dim.y; ++y) {
            for(int x = 0; x < dim.x; ++x) {
                const int flatIndex = z * dim.y * dim.x + y * dim.x + x;
                gradientCache[flatIndex] = gradient(x, y, z);
            }
        }
    }

    const int n = static_cast<int>(m_mesh.vertices.size());
    m_mesh.normals.resize(n);

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; ++i) {
        const glm::vec3& v = m_mesh.vertices[i];

        // Discrete Components
        int dx0 = static_cast<int>(v.x / delta);
        int dy0 = static_cast<int>(v.y / delta);
        int dz0 = static_cast<int>(v.z / delta);

        int dx1 = std::min(dx0 + 1, dim.x - 1);
        int dy1 = std::min(dy0 + 1, dim.y - 1);
        int dz1 = std::min(dz0 + 1, dim.z - 1);

        float fx = (v.x - dx0) / (delta);
        float fy = (v.y - dy0) / (delta);
        float fz = (v.z - dz0) / (delta);

        glm::vec3 g000 = cachedGradient(dx0, dy0, dz0);
        glm::vec3 g100 = cachedGradient(dx1, dy0, dz0);
        glm::vec3 g101 = cachedGradient(dx1, dy0, dz1);
        glm::vec3 g001 = cachedGradient(dx0, dy0, dz1);
        glm::vec3 g010 = cachedGradient(dx0, dy1, dz0);
        glm::vec3 g110 = cachedGradient(dx1, dy1, dz0);
        glm::vec3 g111 = cachedGradient(dx1, dy1, dz1);
        glm::vec3 g011 = cachedGradient(dx0, dy1, dz1);

        glm::vec3 g00 = glm::mix(g000, g100, fx);
        glm::vec3 g01 = glm::mix(g001, g101, fx);
        glm::vec3 g10 = glm::mix(g010, g110, fx);
        glm::vec3 g11 = glm::mix(g011, g111, fx);

        glm::vec3 g0 = glm::mix(g00, g10, fy);
        glm::vec3 g1 = glm::mix(g01, g11, fy);

        glm::vec3 normal = glm::normalize(glm::mix(g0, g1, fz));

        // -normal because normal is along density descent
        m_mesh.normals[i] = -normal;
    }
}

void VolumeVisualisation::calculateNormalsAnalytical() {
    const float phi = 1.61803398875f;
    const float sqPhi = phi * phi;

    const float w = BARTH_W;
    const float sqW = w * w;

    const float k = 1.0f + 2.0f * phi;

    glm::ivec3 dim = m_volume_data.dimensions;
    float scalingFactor = 8.0f / std::max(dim.x, std::max(dim.y, dim.z));

    const size_t vertexCount = m_mesh.vertices.size();
    m_mesh.normals.resize(vertexCount);

    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < vertexCount; ++i) {
        const glm::vec3& v = m_mesh.vertices[i];

        float x = (v.x - m_volume_data.dimensions.x * 0.5f) * scalingFactor;
        float y = (v.y - m_volume_data.dimensions.y * 0.5f) * scalingFactor;
        float z = (v.z - m_volume_data.dimensions.z * 0.5f) * scalingFactor;

        float sqX = x * x;
        float sqY = y * y;
        float sqZ = z * z;

        float A = sqPhi * sqX - sqY;
        float B = sqPhi * sqY - sqZ;
        float C = sqPhi * sqZ - sqX;
        float S = sqX + sqY + sqZ - sqW;
        float AB = A * B;
        float BC = B * C;
        float CA = C * A;

        // Gradient of Barth Sextic
        float dFdx = 2.0f * x * (sqPhi * BC - AB) - k * x * S;
        float dFdy = 2.0f * y * (sqPhi * CA - BC) - k * y * S;
        float dFdz = 2.0f * z * (sqPhi * AB - CA) - k * z * S;

        glm::vec3 normal = glm::normalize(glm::vec3(dFdx, dFdy, dFdz));
        // -normal because normal is along density descent
        m_mesh.normals[i] = -normal;
    }
}

void VolumeVisualisation::snapToGrid(float distance) {
    // DONE(4.5):
    if(m_mesh.vertices.size() == 0) {
        return;
    }

    auto approxEqual = [](const glm::vec3 x, const glm::vec3 y, const float threshold) {
        glm::vec3 diff = x - y;
        float sqrDistance = glm::dot(diff, diff);

        return sqrDistance < threshold;
    };

    const float threshold = distance * distance;

    std::unordered_map<std::tuple<int, int, int>, int, TupleHash> grid;
    std::vector<int> indexRemap(m_mesh.vertices.size());
    std::vector<glm::vec3> newVertices;

    for(int i = 0; i < m_mesh.vertices.size(); ++i) {
        const glm::vec3 v = m_mesh.vertices[i];
        const glm::vec3 nearestGridPoint = glm::round(v);

        // If vertex is approximately equal to the nearest grid point, check if another vertex was snapped to the same point;
        // if that's the case, do not add this as a new vertex and instead remap index of this vertex to the index of the previously snapped vertex
        // If vertex is not approximately equal to the nearest grid point, directly add it as a new vertex
        if(approxEqual(nearestGridPoint, v, threshold)) {
            std::tuple<int, int, int> key = { (int)nearestGridPoint.x, (int)nearestGridPoint.y, (int)nearestGridPoint.z };
            auto it = grid.find(key);

            if(it != grid.end()){
                int existentVertexIndex = it->second;
                indexRemap[i] = existentVertexIndex;
            } else {
                int newIndex = newVertices.size();
                grid[key] = newIndex;
                newVertices.push_back(nearestGridPoint);
                indexRemap[i] = newIndex;
            }
        } else {
            indexRemap[i] = newVertices.size();
            newVertices.push_back(v);
        }
    }

    for(glm::ivec3& t : m_mesh.triangles){
        t.x = indexRemap[t.x];
        t.y = indexRemap[t.y];
        t.z = indexRemap[t.z];
    }

    m_mesh.vertices = newVertices;
}

void VolumeVisualisation::cleanUpTriangleSoup() {
    // DONE(4.5):
    const float epsilon = 0.01f;
    const float sqEpsilon = epsilon * epsilon;

    // TODO might replace with map and < operator as discussed in the lecture
    std::unordered_map<std::tuple<int, int, int>, int, TupleHash> grid;
    std::vector<int> indexRemap(m_mesh.vertices.size());
    std::vector<glm::vec3> newVertices;

    for (size_t i = 0; i < m_mesh.vertices.size(); ++i) {
        const glm::vec3& v = m_mesh.vertices[i];

        // Centre Cell
        int cx = (int)std::floor(v.x / epsilon);
        int cy = (int)std::floor(v.y / epsilon);
        int cz = (int)std::floor(v.z / epsilon);

        int foundIndex = -1;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    std::tuple<int, int, int> key = { cx + dx, cy + dy, cz + dz };
                    auto it = grid.find(key);

                    if (it != grid.end()) {
                        int candidate = it->second;
                        glm::vec3 diff = v - newVertices[candidate];
                        float sqDistance = glm::dot(diff, diff);

                        if (sqDistance < sqEpsilon) {
                            foundIndex = candidate;
                            goto tripleLoopBreak;
                        }
                    }
                }
            }
        }

        tripleLoopBreak:
        // This vertex is unique so far
        if(foundIndex == -1) {
            foundIndex = newVertices.size();
            newVertices.push_back(v);
            grid[{ cx, cy, cz }] = foundIndex;
        }

        indexRemap[i] = foundIndex;
    }

    // Remap triangle indices
    for(glm::ivec3& t : m_mesh.triangles) {
        t.x = indexRemap[t.x];
        t.y = indexRemap[t.y];
        t.z = indexRemap[t.z];
    }

    // Remove degenerate triangles (those which have very little area)
    std::erase_if(m_mesh.triangles, [&](const glm::ivec3& t) {
        const glm::vec3& a = newVertices[t.x];
        const glm::vec3& b = newVertices[t.y];
        const glm::vec3& c = newVertices[t.z];

        glm::vec3 n = glm::cross(b - a, c - a);

        const float areaEpsilon = 1e-7f;
        return glm::dot(n, n) < areaEpsilon;
    });


    // Remove duplicate triangles (those which represent the same triangle)
    std::unordered_set<std::tuple<int,int,int>, TupleHash> seenTriangles;
    std::erase_if(m_mesh.triangles, [&](const glm::ivec3& t) {


        auto canonical = [](int a, int b, int c) {
            // Canonical form is the one where smallest index comes first without changing the triangle winding order
            if (a < b && a < c) { return std::tuple{ a, b, c }; }
            if (b < a && b < c) { return std::tuple{ b, c, a }; }
            return std::tuple{ c, a, b };
        };

        auto key = canonical(t.x, t.y, t.z);
        return !seenTriangles.insert(key).second;
    });

    m_mesh.vertices = newVertices;
}
