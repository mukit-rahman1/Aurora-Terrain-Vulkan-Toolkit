#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>

struct BuildArgs {
    std::string heightmapPath;
    std::string outDir;
    uint32_t lodCount = 5;
};

int runBuildCommand(VkDevice device,
                    VkPhysicalDevice physicalDevice,
                    VkQueue queue,
                    uint32_t computeQueueFamily,
                    const BuildArgs& args);