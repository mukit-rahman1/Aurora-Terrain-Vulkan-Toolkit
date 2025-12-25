#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

Buffer createBuffer(VkDevice device,
                    VkPhysicalDevice physicalDevice,
                    VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memProps);

void vkCheck(VkResult r, const char* msg);
std::vector<char> readFile(const std::string& path);

uint32_t findMemoryType(uint32_t typeFilter,
                        VkMemoryPropertyFlags props,
                        VkPhysicalDevice physicalDevice);

VkDescriptorSetLayout makeSetLayout(VkDevice device);

VkPipelineLayout makePipelineLayout(VkDevice device,
                                    VkDescriptorSetLayout setLayout,
                                    uint32_t pushConstantBytes);

VkPipeline makeComputePipeline(VkDevice device,
                               VkPipelineLayout pipelineLayout,
                               const std::string& spvPath,
                               VkShaderModule* outModule);

bool hasLayer(const std::vector<VkLayerProperties>& layers, const char* name);
