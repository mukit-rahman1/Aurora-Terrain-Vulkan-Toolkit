#pragma once
#include <string>
#include <cstdint>

struct ViewArgs {
    std::string inDir = "out/world";
    uint32_t tileX = 0;
    uint32_t tileY = 0;
    uint32_t lod   = 0;   // 0 = 256, 1 = 128, and so on
    float heightScale = 100.0f;
    float spacing = 1.0f;
};

int runViewCommand(const ViewArgs& args);
