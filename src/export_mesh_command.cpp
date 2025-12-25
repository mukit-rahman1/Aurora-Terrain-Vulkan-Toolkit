#include "export_mesh_command.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t TILE0_SIZE = 256;

static void ensureDir(const std::string& path) {
    std::filesystem::create_directories(std::filesystem::path(path));
}

static bool fileExists(const std::string& path) {
    return std::filesystem::exists(std::filesystem::path(path));
}

static std::vector<uint16_t> readRawU16(const std::string& path, size_t count) {
    std::vector<uint16_t> data(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open: " + path);

    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(count * sizeof(uint16_t)));
    if (!f) throw std::runtime_error("Failed to read enough bytes: " + path);
    return data;
}


static void writeOBJ(const std::string& path,
                     const std::vector<float>& vertsXYZ,
                     const std::vector<uint32_t>& indices)
{
    std::ofstream o(path);
    if (!o) throw std::runtime_error("Failed to write: " + path);

    // vertices
    for (size_t i = 0; i < vertsXYZ.size(); i += 3) {
        o << "v " << vertsXYZ[i] << " " << vertsXYZ[i + 1] << " " << vertsXYZ[i + 2] << "\n";
    }

    // faces (OBJ is 1-based)
    for (size_t i = 0; i < indices.size(); i += 3) {
        o << "f "
          << (indices[i] + 1) << " "
          << (indices[i + 1] + 1) << " "
          << (indices[i + 2] + 1) << "\n";
    }
}

// Parse "tile_X_Y" -> (X, Y). Returns false if format unexpected.
static bool parseTileXY(const std::string& folderName, uint32_t& tx, uint32_t& ty) {
    // expected: "tile_3_5"
    const std::string prefix = "tile_";
    if (folderName.rfind(prefix, 0) != 0) return false;

    // split by '_'
    // tile_ X _ Y
    size_t p1 = folderName.find('_', 5);
    if (p1 == std::string::npos) return false;

    std::string sx = folderName.substr(5, p1 - 5);
    std::string sy = folderName.substr(p1 + 1);

    try {
        tx = static_cast<uint32_t>(std::stoul(sx));
        ty = static_cast<uint32_t>(std::stoul(sy));
        return true;
    } catch (...) {
        return false;
    }
}

static void buildGridMeshFromHeightU16(const std::vector<uint16_t>& h,
                                       uint32_t N,
                                       float spacing,
                                       float heightScale,
                                       uint32_t tileX,
                                       uint32_t tileY,
                                       std::vector<float>& outVertsXYZ,
                                       std::vector<uint32_t>& outIdx)
{
    // Vertex positions
    outVertsXYZ.clear();
    outVertsXYZ.resize(static_cast<size_t>(N) * N * 3);

    // Offset tiles so they line up in world space.
    // Using (TILE0_SIZE - 1) reduces duplicated seams when adjacent tiles meet.
    const float tileWorldStride = float(TILE0_SIZE - 1) * spacing;
    const float baseX = tileWorldStride * float(tileX);
    const float baseZ = tileWorldStride * float(tileY);

    for (uint32_t z = 0; z < N; z++) {
        for (uint32_t x = 0; x < N; x++) {
            const uint32_t i = z * N + x;

            float hx = float(x) * spacing + baseX;
            float hz = float(z) * spacing + baseZ;

            float yn = float(h[i]) / 65535.0f;   // normalized 0..1
            float hy = yn * heightScale;

            const size_t vi = (size_t)i * 3;
            outVertsXYZ[vi + 0] = hx; // x
            outVertsXYZ[vi + 1] = hy; // y
            outVertsXYZ[vi + 2] = hz; // z
        }
    }

    // Indices (two triangles per quad)
    outIdx.clear();
    outIdx.reserve(static_cast<size_t>(N - 1) * (N - 1) * 6);

    for (uint32_t z = 0; z < N - 1; z++) {
        for (uint32_t x = 0; x < N - 1; x++) {
            uint32_t i0 = z * N + x;
            uint32_t i1 = z * N + (x + 1);
            uint32_t i2 = (z + 1) * N + x;
            uint32_t i3 = (z + 1) * N + (x + 1);

            // CCW winding
            outIdx.push_back(i0); outIdx.push_back(i2); outIdx.push_back(i1);
            outIdx.push_back(i1); outIdx.push_back(i2); outIdx.push_back(i3);
        }
    }
}

int runExportMeshCommand(const ExportMeshArgs& args) {
    const std::string tilesDir = args.inDir + "/tiles";
    if (!std::filesystem::exists(tilesDir)) {
        throw std::runtime_error("Tiles folder not found: " + tilesDir);
    }

    ensureDir(args.outDir);

    std::vector<float> verts;
    std::vector<uint32_t> idx;

    size_t exported = 0;

    for (const auto& entry : std::filesystem::directory_iterator(tilesDir)) {
        if (!entry.is_directory()) continue;

        const std::string folderName = entry.path().filename().string();
        uint32_t tileX = 0, tileY = 0;
        if (!parseTileXY(folderName, tileX, tileY)) continue;

        const std::string tilePath = entry.path().string();

        for (uint32_t lod = 0; lod < args.lodCount; lod++) {
            const uint32_t N = TILE0_SIZE >> lod;
            if (N < 2) break;

            const std::string hPath = tilePath + "/lod" + std::to_string(lod) + ".height.raw";
            if (!fileExists(hPath)) continue;

            // load heights
            auto h = readRawU16(hPath, static_cast<size_t>(N) * N);

            // build mesh
            buildGridMeshFromHeightU16(h, N, args.spacing, args.heightScale,
                                       tileX, tileY, verts, idx);

            // write obj
            const std::string outObj = args.outDir + "/" + folderName + "_lod" + std::to_string(lod) + ".obj";
            writeOBJ(outObj, verts, idx);

            exported++;
        }
    }

    std::cout << "Exported " << exported << " OBJ files to: " << args.outDir << "\n";
    return 0;
}
