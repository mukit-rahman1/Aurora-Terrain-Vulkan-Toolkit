#pragma once
#include <string>
#include <cstdint>

struct ExportMeshArgs {
    std::string inDir;
    std::string outDir;
    uint32_t lodCount = 1;
    float spacing = 1.0f;
    float heightScale = 1.0f;

    bool openBlender = false;
    std::string blenderPath;   // path to blender.exe
};

int runExportMeshCommand(const ExportMeshArgs& args);
