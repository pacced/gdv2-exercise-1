#pragma once

#include "gris/GridCell.h"
#include "gris/TriangeMesh.h"
#include "gris/VolumeData.h"

#include "OctreeNode.h"

class VolumeVisualisation {
public:
    explicit VolumeVisualisation(const gris::BoundingBox& bounding_box);

    void generateTriviateVolumeData(const glm::ivec3& dimensions, const gris::BoundingBox& bounding_box);
    void loadRawVolumeData(const std::string& path, const glm::ivec3& dimensions, const gris::BoundingBox& bounding_box);

    void generateMesh(float iso_value, bool dual, bool grid_snapping, float grid_snapping_distance, bool optimize_mesh);

    void drawMesh(QOpenGLFunctions_2_1* f) const;
    void drawNormals(QOpenGLFunctions_2_1* f) const;
    void drawVolumeData(QOpenGLFunctions_2_1* f) const;
    void drawOctree(QOpenGLFunctions_2_1* f) const;
    void drawBoundingBox(QOpenGLFunctions_2_1* f) const;

private:
    static gris::TriangleMesh poligonize(const gris::GridCell& grid_cell, float iso_value);

    int getFlatIndex(int x, int y, int z) const;

    float getDensity(int flatIndex) const;
    float getDensity(int x, int y, int z) const;

    bool checkInside(int flaxIndex) const;
    bool checkInside(int x, int y, int z) const;

    glm::vec3 edgeCutPosition(int x0, int y0, int z0, int x1, int y1, int z1);
    glm::vec3 getEdgeCutByEdge(int x, int y, int z, int edge);

    std::tuple<int, float, float> buildNode(const glm::ivec3 start, const glm::ivec3 end);
    void buildOctree();

    void marchingCubes();
    void marchingCubes2();
    void marchingCubesOctree();
    void dualMarchingCubes();

    void snapToGrid(float distance);
    void cleanUpTriangleSoup();

    void calculateNormals();
    void calculateNormalsFiniteDifferences();
    void calculateNormalsAnalytical();

    gris::BoundingBox m_bounding_box;
    bool m_is_trivariate;
    float m_iso_value;
    gris::VolumeData m_volume_data;
    gris::TriangleMesh m_mesh;
    std::vector<OctreeNode> m_octree;
};
