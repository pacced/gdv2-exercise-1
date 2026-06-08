#include "VolumeVisualisation.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>

#include <tuple>
#include <vector>

#include "glm/fwd.hpp"
#include "glm/geometric.hpp"
#include "gris/GridCell.h"
#include "gris/TriangeMesh.h"
#include "gris/mc_look_up.h"

constexpr float BARTH_W = 1.0f;

struct TupleHash {
    std::size_t operator()(const std::tuple<int,int,int>& k) const {
        std::size_t seed = 0;
        seed ^= std::hash<int>{}(std::get<0>(k)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(std::get<1>(k)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(std::get<2>(k)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

int calculateIndex(const glm::ivec3& grid_position, const glm::ivec3& dimensions) {
    assert(grid_position.x < dimensions.x);
    assert(grid_position.y < dimensions.y);
    assert(grid_position.z < dimensions.z);
    return grid_position.x + grid_position.y * dimensions.x + grid_position.z * dimensions.x * dimensions.y;
};

VolumeVisualisation::VolumeVisualisation(const gris::BoundingBox& bounding_box):
    m_bounding_box(bounding_box) {
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


    int stepX = 1;
    int stepY = stepX * dimensions.x;
    int stepZ = stepY * dimensions.y;

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
        ;
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
        // marchingCubes();
        marchingCubes2();
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
    calculateNormals();

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

    f->glBegin(GL_TRIANGLES);
        for(const glm::ivec3& t : m_mesh.triangles) {
            // Triangles are wound backwards for some reason
            glm::vec3 v0 = min + delta * m_mesh.vertices[t.z];
            glm::vec3 v1 = min + delta * m_mesh.vertices[t.y];
            glm::vec3 v2 = min + delta * m_mesh.vertices[t.x];

            glm::vec3 n0 = m_mesh.normals[t.z];
            glm::vec3 n1 = m_mesh.normals[t.y];
            glm::vec3 n2 = m_mesh.normals[t.x];

            f->glNormal3fv(&n0.x);
            f->glVertex3fv(&v0.x);


            f->glNormal3fv(&n1.x);
            f->glVertex3fv(&v1.x);

            f->glNormal3fv(&n2.x);
            f->glVertex3fv(&v2.x);
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
    float radius = glm::min(delta.x, glm::min(delta.y, delta.z)) / 4.f;

    f->glPointSize(6.0f);
    f->glBegin(GL_POINTS);
    for(int x = 0; x < m_volume_data.dimensions.x; x++) {
        for(int y = 0; y < m_volume_data.dimensions.y; y++) {
            for(int z = 0; z < m_volume_data.dimensions.z; z++) {
                if(m_volume_data.values[calculateIndex({x, y, z}, m_volume_data.dimensions)] >= m_iso_value) {
                    glm::vec3 point = min + delta * glm::vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
                    f->glVertex3fv(reinterpret_cast<const float*>(&point));
                }
            }
        }
    }
    f->glEnd();
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

void VolumeVisualisation::marchingCubes2() {
    m_mesh.vertices.clear();
    m_mesh.normals.clear();
    m_mesh.triangles.clear();

    const gris::VolumeData& volumeData = m_volume_data;
    float isoValue = m_iso_value;
    auto dim = volumeData.dimensions;

    const int stepX = 1;
    const int stepY = stepX * dim.x;
    const int stepZ = stepY * dim.y;

    size_t totalGridPoints = dim.x * dim.y * dim.z;

    auto voxelIdx = [&](int x, int y, int z) {
        return z * stepZ + y * stepY + x * stepX;
    };

    std::vector<int> insideFlags(totalGridPoints);
    int flagIndex = 0;

    // Classify grid points
    for(int z = 0; z < dim.z; ++z) {
        for(int y = 0; y < dim.y; ++y) {
            for(int x = 0; x < dim.x; ++x, ++flagIndex) {
                insideFlags[flagIndex] = (volumeData.values[flagIndex] < isoValue) ? 1 : 0;
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
                int i0 = voxelIdx(x    , y, z);
                int i1 = voxelIdx(x + 1, y, z);

                if (insideFlags[i0] != insideFlags[i1]) {
                    float d0 = volumeData.values[i0];
                    float d1 = volumeData.values[i1];
                    float t = (isoValue - d0) / (d1 - d0);

                    edgeX[i0] = verticesX.size();
                    verticesX.push_back(glm::mix(glm::vec3(x, y, z), glm::vec3(x + 1, y, z), t));
                }
            }
        }
    }

    // Y edges
    for (int z = 0; z < dim.z; z++) {
        for (int y = 0; y < dim.y - 1; y++) {
            for (int x = 0; x < dim.x; x++) {
                int i0 = voxelIdx(x, y    , z);
                int i1 = voxelIdx(x, y + 1, z);

                if (insideFlags[i0] != insideFlags[i1]) {
                    float d0 = volumeData.values[i0];
                    float d1 = volumeData.values[i1];
                    float t = (isoValue - d0) / (d1 - d0);

                    edgeY[i0] = verticesY.size();
                    verticesY.push_back(glm::mix(glm::vec3(x, y, z), glm::vec3(x, y + 1, z), t));
                }
            }
        }
    }

    // Z edges
    for (int z = 0; z < dim.z - 1; z++) {
        for (int y = 0; y < dim.y; y++) {
            for (int x = 0; x < dim.x; x++) {
                int i0 = voxelIdx(x, y, z    );
                int i1 = voxelIdx(x, y, z + 1);

                if (insideFlags[i0] != insideFlags[i1]) {
                    float d0 = volumeData.values[i0];
                    float d1 = volumeData.values[i1];
                    float t = (isoValue - d0) / (d1 - d0);

                    edgeZ[i0] = verticesZ.size();
                    verticesZ.push_back(glm::mix(glm::vec3(x, y, z), glm::vec3(x, y, z + 1), t));
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
                if (insideFlags[voxelIdx(x    , y    , z    )]) caseIdx |= gris::CORNER_0;
                if (insideFlags[voxelIdx(x + 1, y    , z    )]) caseIdx |= gris::CORNER_1;
                if (insideFlags[voxelIdx(x + 1, y    , z + 1)]) caseIdx |= gris::CORNER_2;
                if (insideFlags[voxelIdx(x    , y    , z + 1)]) caseIdx |= gris::CORNER_3;
                if (insideFlags[voxelIdx(x    , y + 1, z    )]) caseIdx |= gris::CORNER_4;
                if (insideFlags[voxelIdx(x + 1, y + 1, z    )]) caseIdx |= gris::CORNER_5;
                if (insideFlags[voxelIdx(x + 1, y + 1, z + 1)]) caseIdx |= gris::CORNER_6;
                if (insideFlags[voxelIdx(x    , y + 1, z + 1)]) caseIdx |= gris::CORNER_7;

                const glm::ivec3* triangles = gris::triangle_table[caseIdx];

                for (int i = 0; i < 5; i++) {
                    if (triangles[i].x == -1) {
                        break;
                    }

                    int indices[3];
                    for (int j = 0; j < 3; j++) {
                        int edge = (&triangles[i].x)[j];
                        switch (edge) {
                            case gris::EDGE_0:  indices[j] = offsetX + edgeX[voxelIdx(x    , y    , z    )]; break;
                            case gris::EDGE_1:  indices[j] = offsetZ + edgeZ[voxelIdx(x + 1, y    , z    )]; break;
                            case gris::EDGE_2:  indices[j] = offsetX + edgeX[voxelIdx(x    , y    , z + 1)]; break;
                            case gris::EDGE_3:  indices[j] = offsetZ + edgeZ[voxelIdx(x    , y    , z    )]; break;
                            case gris::EDGE_4:  indices[j] = offsetX + edgeX[voxelIdx(x    , y + 1, z    )]; break;
                            case gris::EDGE_5:  indices[j] = offsetZ + edgeZ[voxelIdx(x + 1, y + 1, z    )]; break;
                            case gris::EDGE_6:  indices[j] = offsetX + edgeX[voxelIdx(x    , y + 1, z + 1)]; break;
                            case gris::EDGE_7:  indices[j] = offsetZ + edgeZ[voxelIdx(x    , y + 1, z    )]; break;
                            case gris::EDGE_8:  indices[j] = offsetY + edgeY[voxelIdx(x    , y    , z    )]; break;
                            case gris::EDGE_9:  indices[j] = offsetY + edgeY[voxelIdx(x + 1, y    , z    )]; break;
                            case gris::EDGE_10: indices[j] = offsetY + edgeY[voxelIdx(x + 1, y    , z + 1)]; break;
                            case gris::EDGE_11: indices[j] = offsetY + edgeY[voxelIdx(x    , y    , z + 1)]; break;
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
                int voxelIndex = offsetZ0 + offsetY0 + x;

                gris::GridCell voxel;

                voxel.values[0] = m_volume_data.values[offsetYZ0 + x];
                voxel.values[1] = m_volume_data.values[offsetYZ0 + x + stepX];
                voxel.values[2] = m_volume_data.values[offsetYZ1 + x + stepX];
                voxel.values[3] = m_volume_data.values[offsetYZ1 + x];
                voxel.values[4] = m_volume_data.values[offsetYZ2 + x];
                voxel.values[5] = m_volume_data.values[offsetYZ2 + x + stepX];
                voxel.values[6] = m_volume_data.values[offsetYZ3 + x + stepX];
                voxel.values[7] = m_volume_data.values[offsetYZ3 + x];

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
                        vertexOffset + t.x,
                        vertexOffset + t.y,
                        vertexOffset + t.z
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

    // TODO(4.4): implement dual marching cubes algorithm
    const gris::VolumeData& volumeData = m_volume_data;
    float isoValue = m_iso_value;
    auto dim = volumeData.dimensions;

    const int stepX = 1;
    const int stepY = stepX * dim.x;
    const int stepZ = stepY * dim.y;

    const size_t totalGridPoints = dim.x * dim.y * dim.z;
    // const size_t totalVoxels = (dim.x - 1) * (dim.y - 1) * (dim.z - 1);

    auto voxelIdx = [&](int x, int y, int z) {
        return z * stepZ + y * stepY + x * stepX;
    };

    std::vector<std::array<int, 4>> centroidIndexGrid(totalGridPoints, {-1, -1, -1, -1});
    std::vector<std::array<int, 4>> edgeAssociationGrid(totalGridPoints, {0, 0, 0, 0});
    std::vector<glm::vec3> vertices;

    auto inside = [&](int flatIndex) {
        return m_volume_data.values[flatIndex] < isoValue;
    };

    // Edge interpolation helper
    auto edgeCut = [&](int x0, int y0, int z0, int x1, int y1, int z1) {
        float d0 = m_volume_data.values[voxelIdx(x0, y0, z0)];
        float d1 = m_volume_data.values[voxelIdx(x1, y1, z1)];
        float t = (isoValue - d0) / (d1 - d0);
        return glm::mix(glm::vec3(x0, y0, z0), glm::vec3(x1, y1, z1), t);
    };

    auto getEdgeCut = [&](int x, int y, int z, int edge) {
        switch(edge) {
            case gris::EDGE_0:  return edgeCut(x    , y    , z    , x + 1, y    , z    );
            case gris::EDGE_1:  return edgeCut(x + 1, y    , z    , x + 1, y    , z + 1);
            case gris::EDGE_2:  return edgeCut(x + 1, y    , z + 1, x    , y    , z + 1);
            case gris::EDGE_3:  return edgeCut(x    , y    , z    , x    , y    , z + 1);
            case gris::EDGE_4:  return edgeCut(x    , y + 1, z    , x + 1, y + 1, z    );
            case gris::EDGE_5:  return edgeCut(x + 1, y + 1, z    , x + 1, y + 1, z + 1);
            case gris::EDGE_6:  return edgeCut(x + 1, y + 1, z + 1, x    , y + 1, z + 1);
            case gris::EDGE_7:  return edgeCut(x    , y + 1, z + 1, x    , y + 1, z    );
            case gris::EDGE_8:  return edgeCut(x    , y    , z    , x    , y + 1, z    );
            case gris::EDGE_9:  return edgeCut(x + 1, y    , z    , x + 1, y + 1, z    );
            case gris::EDGE_10: return edgeCut(x + 1, y    , z + 1, x + 1, y + 1, z + 1);
            case gris::EDGE_11: return edgeCut(x    , y    , z + 1, x    , y + 1, z + 1);
            default: assert(false);
        }
    };

    for(int z = 0; z < dim.z - 1; ++z) {
        for(int y = 0; y < dim.y - 1; ++y) {
            for(int x = 0; x < dim.x -1; ++x) {
                int currentVoxel = voxelIdx(x, y, z);

                int caseIdx = 0;
                caseIdx |= m_volume_data.values[voxelIdx(x    , y    , z    )] < isoValue ? gris::CORNER_0 : 0;
                caseIdx |= m_volume_data.values[voxelIdx(x + 1, y    , z    )] < isoValue ? gris::CORNER_1 : 0;
                caseIdx |= m_volume_data.values[voxelIdx(x + 1, y    , z + 1)] < isoValue ? gris::CORNER_2 : 0;
                caseIdx |= m_volume_data.values[voxelIdx(x    , y    , z + 1)] < isoValue ? gris::CORNER_3 : 0;
                caseIdx |= m_volume_data.values[voxelIdx(x    , y + 1, z    )] < isoValue ? gris::CORNER_4 : 0;
                caseIdx |= m_volume_data.values[voxelIdx(x + 1, y + 1, z    )] < isoValue ? gris::CORNER_5 : 0;
                caseIdx |= m_volume_data.values[voxelIdx(x + 1, y + 1, z + 1)] < isoValue ? gris::CORNER_6 : 0;
                caseIdx |= m_volume_data.values[voxelIdx(x    , y + 1, z + 1)] < isoValue ? gris::CORNER_7 : 0;

                const int* combinations = gris::dual_points_list[caseIdx];
                for(int i = 0; i < 4; ++i) {
                    int mask = combinations[i];
                    if(mask == 0) {
                        break;
                    }

                    glm::vec3 centroid(0.0f);
                    int count = 0;

                    for(int e = 0; e < 12; ++e) {
                        int edgeBit = 1 << e;

                        if (mask & edgeBit) {
                            centroid += getEdgeCut(x, y, z, edgeBit);
                            ++count;
                        }
                    }

                    centroid /= (float)count;

                    centroidIndexGrid[currentVoxel][i] = vertices.size();
                    edgeAssociationGrid[currentVoxel][i] = mask;
                    vertices.push_back(centroid);
                }
            }
        }
    }

    auto getDualVertexIndex = [&](int x, int y, int z, int edge) {
        int vi = voxelIdx(x, y, z);

        std::array<int, 4>& masks = edgeAssociationGrid[vi];
        for(int i = 0; i < 4; ++i) {
            int mask = masks[i];
            if(mask == 0){
                break;
            }

            if (mask & edge){
                return centroidIndexGrid[vi][i];
            }
        }

        return -1;
    };

    std::vector<glm::ivec3> triangles;

    for(int z = 0; z < dim.z - 1; ++z) {
        for(int y = 0; y < dim.y - 1; ++y) {
            for(int x = 0; x < dim.x - 1; ++x) {
                if(z > 0 && y > 0){
                    const bool entering = inside(voxelIdx(x, y, z)) && !inside(voxelIdx(x + 1, y, z));
                    const bool exiting = !inside(voxelIdx(x, y, z)) && inside(voxelIdx(x + 1, y, z));

                    if(entering || exiting) {
                        int xvi0 = getDualVertexIndex(x, y    , z    , gris::EDGE_0);
                        int xvi1 = getDualVertexIndex(x, y - 1, z    , gris::EDGE_4);
                        int xvi2 = getDualVertexIndex(x, y - 1, z - 1, gris::EDGE_6);
                        int xvi3 = getDualVertexIndex(x, y    , z - 1, gris::EDGE_2);

                        if(xvi0 == -1 || xvi1 == -1 || xvi2 == -1 || xvi3 == -1) {
                            continue;
                        }

                        if(entering) {
                            triangles.push_back({xvi0, xvi1, xvi2});
                            triangles.push_back({xvi0, xvi2, xvi3});
                        } else {
                            triangles.push_back({xvi0, xvi3, xvi2});
                            triangles.push_back({xvi0, xvi2, xvi1});
                        }
                    }
                }

                if(z > 0 && x > 0){
                    const bool entering = inside(voxelIdx(x, y, z)) && !inside(voxelIdx(x, y + 1, z));
                    const bool exiting = !inside(voxelIdx(x, y, z)) && inside(voxelIdx(x, y + 1, z));

                    if(entering || exiting) {
                        int yvi0 = getDualVertexIndex(x    , y, z    , gris::EDGE_8);
                        int yvi1 = getDualVertexIndex(x    , y, z - 1, gris::EDGE_11);
                        int yvi2 = getDualVertexIndex(x - 1, y, z - 1, gris::EDGE_10);
                        int yvi3 = getDualVertexIndex(x - 1, y, z    , gris::EDGE_9);

                        if(yvi0 == -1 || yvi1 == -1 || yvi2 == -1 || yvi3 == -1) {
                            continue;
                        }

                        if(entering) {
                            triangles.push_back({yvi0, yvi1, yvi2});
                            triangles.push_back({yvi0, yvi2, yvi3});
                        } else {
                            triangles.push_back({yvi0, yvi3, yvi2});
                            triangles.push_back({yvi0, yvi2, yvi1});
                        }
                    }
                }

                if(x > 0 && y > 0){
                    const bool entering = inside(voxelIdx(x, y, z)) && !inside(voxelIdx(x, y, z + 1));
                    const bool exiting = !inside(voxelIdx(x, y, z)) && inside(voxelIdx(x, y, z + 1));

                    if(entering || exiting) {
                        int zvi0 = getDualVertexIndex(x    , y    , z, gris::EDGE_3);
                        int zvi1 = getDualVertexIndex(x - 1, y    , z, gris::EDGE_1);
                        int zvi2 = getDualVertexIndex(x - 1, y - 1, z, gris::EDGE_5);
                        int zvi3 = getDualVertexIndex(x    , y - 1, z, gris::EDGE_7);

                        if(zvi0 == -1 || zvi1 == -1 || zvi2 == -1 || zvi3 == -1) {
                            continue;
                        }

                        if(entering) {
                            triangles.push_back({zvi0, zvi1, zvi2});
                            triangles.push_back({zvi0, zvi2, zvi3});
                        } else {
                            triangles.push_back({zvi0, zvi3, zvi2});
                            triangles.push_back({zvi0, zvi2, zvi1});
                        }
                    }
                }
            }
        }
    }


    // X-edges pass
    // for(int z = 1; z < dim.z; ++z) {
    //     for(int y = 1; y < dim.y; ++y) {
    //         for(int x = 0; x < dim.x - 1; ++x) {
    //             int i0 = voxelIdx(x    , y, z);
    //             int i1 = voxelIdx(x + 1, y, z);
    //
    //             if(inside(i0) == inside(i1)) {
    //                 continue;
    //             }
    //
    //             // vi: vertex index
    //             int vi0 = getDualVertexIndex(x, y    , z    , gris::EDGE_0);
    //             int vi1 = getDualVertexIndex(x, y - 1, z    , gris::EDGE_4);
    //             int vi2 = getDualVertexIndex(x, y - 1, z - 1, gris::EDGE_6);
    //             int vi3 = getDualVertexIndex(x, y    , z - 1, gris::EDGE_2);
    //
    //             if(vi0 == -1 || vi1 == -1 || vi2 == -1 || vi3 == -1) {
    //                 continue;
    //             }
    //
    //             if(inside(i0)) {
    //                 triangles.push_back({vi0, vi1, vi2});
    //                 triangles.push_back({vi0, vi2, vi3});
    //             } else {
    //                 triangles.push_back({vi0, vi3, vi2});
    //                 triangles.push_back({vi0, vi2, vi1});
    //             }
    //         }
    //     }
    // }
    //
    // // Y-edges pass
    // for(int z = 1; z < dim.z; ++z) {
    //     for(int y = 0; y < dim.y - 1; ++y) {
    //         for(int x = 1; x < dim.x; ++x) {
    //             int i0 = voxelIdx(x, y    , z);
    //             int i1 = voxelIdx(x, y + 1, z);
    //
    //             if(inside(i0) == inside(i1)) {
    //                 continue;
    //             }
    //
    //             // vi: vertex index
    //             int vi0 = getDualVertexIndex(x    , y, z    , gris::EDGE_9);
    //             int vi1 = getDualVertexIndex(x - 1, y, z    , gris::EDGE_8);
    //             int vi2 = getDualVertexIndex(x - 1, y, z - 1, gris::EDGE_11);
    //             int vi3 = getDualVertexIndex(x    , y, z - 1, gris::EDGE_10);
    //
    //             if(vi0 == -1 || vi1 == -1 || vi2 == -1 || vi3 == -1) {
    //                 continue;
    //             }
    //
    //             if(inside(i0)) {
    //                 triangles.push_back({vi0, vi1, vi2});
    //                 triangles.push_back({vi0, vi2, vi3});
    //             } else {
    //                 triangles.push_back({vi0, vi3, vi2});
    //                 triangles.push_back({vi0, vi2, vi1});
    //             }
    //         }
    //     }
    // }
    //
    // // Z-edges pass
    // for(int z = 0; z < dim.z - 1; ++z) {
    //     for(int y = 1; y < dim.y; ++y) {
    //         for(int x = 1; x < dim.x; ++x) {
    //             int i0 = voxelIdx(x, y, z    );
    //             int i1 = voxelIdx(x, y, z + 1);
    //
    //             if(inside(i0) == inside(i1)) {
    //                 continue;
    //             }
    //
    //             // vi: vertex index
    //             int vi0 = getDualVertexIndex(x    , y    , z, gris::EDGE_1);
    //             int vi1 = getDualVertexIndex(x - 1, y    , z, gris::EDGE_3);
    //             int vi2 = getDualVertexIndex(x - 1, y - 1, z, gris::EDGE_7);
    //             int vi3 = getDualVertexIndex(x    , y - 1, z, gris::EDGE_5);
    //
    //             if(vi0 == -1 || vi1 == -1 || vi2 == -1 || vi3 == -1) {
    //                 continue;
    //             }
    //
    //             if(inside(i0)) {
    //                 triangles.push_back({vi0, vi1, vi2});
    //                 triangles.push_back({vi0, vi2, vi3});
    //             } else {
    //                 triangles.push_back({vi0, vi3, vi2});
    //                 triangles.push_back({vi0, vi2, vi1});
    //             }
    //         }
    //     }
    // }


    m_mesh.vertices.insert(m_mesh.vertices.end(), vertices.begin(), vertices.end());
    m_mesh.triangles.insert(m_mesh.triangles.end(), triangles.begin(), triangles.end());
}

void VolumeVisualisation::calculateNormals() {
    if(m_is_trivariate) {
        calculateNormalsAnalytical();
    } else {
        calculateNormalsFiniteDifferences();
    }
}

void VolumeVisualisation::calculateNormalsFiniteDifferences() {
    auto dim = m_volume_data.dimensions;

    const int stepX = 1;
    const int stepY = stepX * dim.x;
    const int stepZ = stepY * dim.y;

    const float delta = 1.0f;

    auto density = [&](int x, int y, int z) {
        return m_volume_data.values[z * stepZ + y * stepY + x * stepX];
    };

    auto gradient = [&](int x, int y, int z) {
        // m: minus, p: plus

        int xm = std::max(x - 1, 0);
        int xp = std::min(x + 1, m_volume_data.dimensions.x - 1);

        int ym = std::max(y - 1, 0);
        int yp = std::min(y + 1, m_volume_data.dimensions.y - 1);

        int zm = std::max(z - 1, 0);
        int zp = std::min(z + 1, m_volume_data.dimensions.z - 1);

        // std::cout << xp << ", " << yp << ", " << zp << ", " << xm << ", " << ym << ", " << zm << ", " << std::endl;

        return glm::vec3(
            (density(xp, y , z ) - density(xm, y , z  )) / (2.0f * delta),
            (density(x , yp, z ) - density(x , ym, z  )) / (2.0f * delta),
            (density(x , y , zp) - density(x , y , zm )) / (2.0f * delta)
        );
    };


    for(const glm::vec3& v : m_mesh.vertices) {
        // Discrete Components
        int dx0 = (int)(v.x / delta);
        int dy0 = (int)(v.y / delta);
        int dz0 = (int)(v.z / delta);

        int dx1 = std::min(dx0 + 1, dim.x - 1);
        int dy1 = std::min(dy0 + 1, dim.y - 1);
        int dz1 = std::min(dz0 + 1, dim.z - 1);

        float fx = (v.x / delta) - dx0;
        float fy = (v.y / delta) - dy0;
        float fz = (v.z / delta) - dz0;

        glm::vec3 g000 = gradient(dx0, dy0, dz0);
        glm::vec3 g100 = gradient(dx1, dy0, dz0);
        glm::vec3 g101 = gradient(dx1, dy0, dz1);
        glm::vec3 g001 = gradient(dx0, dy0, dz1);
        glm::vec3 g010 = gradient(dx0, dy1, dz0);
        glm::vec3 g110 = gradient(dx1, dy1, dz0);
        glm::vec3 g111 = gradient(dx1, dy1, dz1);
        glm::vec3 g011 = gradient(dx0, dy1, dz1);

        glm::vec3 g00 = glm::mix(g000, g100, fx);
        glm::vec3 g01 = glm::mix(g001, g101, fx);
        glm::vec3 g10 = glm::mix(g010, g110, fx);
        glm::vec3 g11 = glm::mix(g011, g111, fx);

        glm::vec3 g0 = glm::mix(g00, g10, fy);
        glm::vec3 g1 = glm::mix(g01, g11, fy);

        glm::vec3 normal = glm::normalize(glm::mix(g0, g1, fz));

        // -normal because normal is along density descent
        m_mesh.normals.push_back(-normal);
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

    for(const glm::ivec3& v : m_mesh.vertices) {
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
        m_mesh.normals.push_back(-normal);
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
                    // Reach per component
                    // float rx = dx * epsilon;
                    // float ry = dy * epsilon;
                    // float rz = dz * epsilon;
                    //
                    // if(rx * rx + ry * ry + rz * rz > sqEpsilon) {
                    //     continue;
                    // }

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

    // Remove degenerate triangles (those which collapsed into an edge or a point)
    // std::erase_if(m_mesh.triangles, [](const glm::ivec3& t){
    //    return t.x == t.y || t.y == t.z || t.x == t.z;
    // });

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
            // // Canonical form is the one where indices are sorted in ascending order without considering the triangle winding order
            // if (a > b) std::swap(a, b);
            // if (b > c) std::swap(b, c);
            // if (a > b) std::swap(a, b);
            // return std::tuple{ a, b, c };


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
