#pragma once
#include <string>
#include <cstdint>

struct ExportMeshArgs {
    std::string inDir  = "out/world";   // expects inDir/tiles/tile_x_y/lod*.height.raw
    std::string outDir = "out/meshes";  // where .obj files go
    uint32_t lodCount  = 5;             // default: lod0 to lod4
    float heightScale  = 100.0f;        // multiply normalized height by this
    float spacing      = 1.0f;          // dx=dz=spacing 
};

int runExportMeshCommand(const ExportMeshArgs& args);
