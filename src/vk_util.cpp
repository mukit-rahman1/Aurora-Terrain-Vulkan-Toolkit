#include "vk_util.h"
#include <vulkan/vulkan.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>


/*
STEPS IN MAIN
// ---- 1) Create instance (info and instance) ----
// ---- 2) Pick physical device ----
// ---- 3) Create logical device + compute queue ----
// ---- 4) Create input/output buffers (host visible for simplicity) ----
// ---- 5) Descriptor set layout (binding 0=input, 1=output) ----
// ---- 6) Create shader module from SPIR-V ----
// ---- 7) Create compute pipeline ----
// ---- 8) Descriptor pool + set ----
// ---- 9) Command pool + command buffer ----
// ---- 10) Submit + wait ----
// ---- 11) Read output ----
// ---- 12) Cleanup ----
*/

// ----- SETUP -----
//check for vk
void vkCheck(VkResult r, const char* msg) {
    if (r != VK_SUCCESS) {
        std::cerr << "Vulkan error: " << msg << " (VkResult=" << r << ")\n";
        std::exit(1);
    }
}

//read from path
std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + path);

    const auto size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    return buffer;
}

//figure out mem type
 uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props,
                              VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type.");
}

//buffer data structure
// struct Buffer {
//     VkBuffer buffer = VK_NULL_HANDLE;
//     VkDeviceMemory memory = VK_NULL_HANDLE;
//     VkDeviceSize size = 0;
// };


//buffer constructor
 Buffer createBuffer(VkDevice device,
                           VkPhysicalDevice physicalDevice,
                           VkDeviceSize size,
                           VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags memProps) {
    Buffer b{};
    b.size = size;

    VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCheck(vkCreateBuffer(device, &bufInfo, nullptr, &b.buffer), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, b.buffer, &req);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = req.size;
    allocInfo.memoryTypeIndex = findMemoryType(req.memoryTypeBits, memProps, physicalDevice);

    vkCheck(vkAllocateMemory(device, &allocInfo, nullptr, &b.memory), "vkAllocateMemory");
    vkCheck(vkBindBufferMemory(device, b.buffer, b.memory, 0), "vkBindBufferMemory");

    return b;
}


//descriptor set layout (resusing) --- step 5
VkDescriptorSetLayout makeSetLayout(VkDevice device) {
    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0;
    b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b0.descriptorCount = 1;
    b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding b1{};
    b1.binding = 1;
    b1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b1.descriptorCount = 1;
    b1.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding bindings[] = { b0, b1 };

    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = 2;
    info.pBindings = bindings;

    VkDescriptorSetLayout layout{};
    vkCheck(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout),
            "vkCreateDescriptorSetLayout");
    return layout;
}

//pipeline layout with push constants  --- step 5
    VkPipelineLayout makePipelineLayout(VkDevice device, VkDescriptorSetLayout setLayout, uint32_t pushConstantBytes) {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = pushConstantBytes;

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1;
        info.pSetLayouts = &setLayout;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &pcr;

        VkPipelineLayout layout{};
        vkCheck(vkCreatePipelineLayout(device, &info, nullptr, &layout),
                "vkCreatePipelineLayout");
        return layout;
    }

    //make computepipline -- step 7
    VkPipeline makeComputePipeline(VkDevice device,
                                VkPipelineLayout pipelineLayout,
                                const std::string& spvPath,
                                VkShaderModule* outModule) {
        auto code = readFile(spvPath);

        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = code.size();
        sm.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule module{};
        vkCheck(vkCreateShaderModule(device, &sm, nullptr, &module), "vkCreateShaderModule");
        if (outModule) *outModule = module;

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage = stage;
        cp.layout = pipelineLayout;

        VkPipeline pipeline{};
        vkCheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &pipeline),
                "vkCreateComputePipelines");

        return pipeline;
    }


//check if theres a layer
bool hasLayer(const std::vector<VkLayerProperties>& layers, const char* name) {
    for (const auto& l : layers) {
        if (std::strcmp(l.layerName, name) == 0) return true;
    }
    return false;
}
