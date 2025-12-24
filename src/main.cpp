#include <vulkan/vulkan.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// ----- SETUP -----
//check for vk
static void vkCheck(VkResult r, const char* msg) {
    if (r != VK_SUCCESS) {
        std::cerr << "Vulkan error: " << msg << " (VkResult=" << r << ")\n";
        std::exit(1);
    }
}

//read from path
static std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + path);

    const auto size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    return buffer;
}

//figure out mem type
static uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props,
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
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

//buffer constructor
static Buffer createBuffer(VkDevice device,
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

//check if theres a layer
static bool hasLayer(const std::vector<VkLayerProperties>& layers, const char* name) {
    for (const auto& l : layers) {
        if (std::strcmp(l.layerName, name) == 0) return true;
    }
    return false;
}


int main() {
    // ---- Settings (match shader) ----
    const std::string kShaderPath = "../shaders/double.comp.spv";
    const uint32_t N = 1024;             // number of floats
    const uint32_t LOCAL_SIZE_X = 256;   // MUST match layout(local_size_x=...) in shader

    // ---- 1) Create instance (info and instance) ----
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "TerrainForgeCompute";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "NoEngine";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_2;

    // Validation layer (just to help)
    std::vector<const char*> enabledLayers;
    {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

        const char* kValidation = "VK_LAYER_KHRONOS_validation";
        if (hasLayer(layers, kValidation)) {
            enabledLayers.push_back(kValidation);
        } else {
            std::cerr << "[Warn] Validation layer not found (ok, but debugging is harder).\n";
        }
    }

    VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instInfo.pApplicationInfo = &app;
    instInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
    instInfo.ppEnabledLayerNames = enabledLayers.empty() ? nullptr : enabledLayers.data();

    VkInstance instance = VK_NULL_HANDLE;
    vkCheck(vkCreateInstance(&instInfo, nullptr, &instance), "vkCreateInstance");


    // ---- 2) Pick physical device ----
    uint32_t devCount = 0;
    vkCheck(vkEnumeratePhysicalDevices(instance, &devCount, nullptr), "vkEnumeratePhysicalDevices(count)");
    if (devCount == 0) {
        std::cerr << "No Vulkan physical devices found.\n";
        return 1;
    }
    std::vector<VkPhysicalDevice> phys(devCount);
    vkCheck(vkEnumeratePhysicalDevices(instance, &devCount, phys.data()), "vkEnumeratePhysicalDevices(list)");

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = UINT32_MAX;

    auto scoreDevice = [](VkPhysicalDevice d) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        // Prefer discrete GPUs
        return (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 100 : 10;
    };

    int bestScore = -1;
    for (auto d : phys) {
        // Find compute queue family
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qProps(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qCount, qProps.data());

        uint32_t found = UINT32_MAX;
        for (uint32_t i = 0; i < qCount; i++) {
            if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                found = i;
                break;
            }
        }
        if (found == UINT32_MAX) continue;

        int s = scoreDevice(d);
        if (s > bestScore) {
            bestScore = s;
            physicalDevice = d;
            computeQueueFamily = found;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        std::cerr << "No suitable GPU with compute queue found.\n";
        return 1;
    }

    {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(physicalDevice, &p);
        std::cout << "Using GPU: " << p.deviceName << "\n";
        std::cout << "Compute queue family: " << computeQueueFamily << "\n";
    }

    // ---- 3) Create logical device + compute queue ----
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo qInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qInfo.queueFamilyIndex = computeQueueFamily;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;

    VkDevice device = VK_NULL_HANDLE;
    vkCheck(vkCreateDevice(physicalDevice, &devInfo, nullptr, &device), "vkCreateDevice");

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, computeQueueFamily, 0, &queue);

    // ---- 4) Create input/output buffers (host visible for simplicity) ----
    const VkDeviceSize bytes = sizeof(float) * static_cast<VkDeviceSize>(N);

    const VkMemoryPropertyFlags hostMem =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    Buffer inBuf = createBuffer(device, physicalDevice, bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMem);
    Buffer outBuf = createBuffer(device, physicalDevice, bytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMem);

    // Fill input buffer
    {
        void* mapped = nullptr;
        vkCheck(vkMapMemory(device, inBuf.memory, 0, bytes, 0, &mapped), "vkMapMemory(in)");
        float* f = reinterpret_cast<float*>(mapped);
        for (uint32_t i = 0; i < N; i++) f[i] = static_cast<float>(i);
        vkUnmapMemory(device, inBuf.memory);
    }
    // Clear output buffer
    {
        void* mapped = nullptr;
        vkCheck(vkMapMemory(device, outBuf.memory, 0, bytes, 0, &mapped), "vkMapMemory(out)");
        std::memset(mapped, 0, static_cast<size_t>(bytes));
        vkUnmapMemory(device, outBuf.memory);
    }

    // ---- 5) Descriptor set layout (binding 0=input, 1=output) ----
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

    VkDescriptorSetLayoutBinding bindings[] = {b0, b1};

    VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslInfo.bindingCount = 2;
    dslInfo.pBindings = bindings;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    vkCheck(vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &setLayout),
            "vkCreateDescriptorSetLayout");

    struct PushConsts { uint32_t N; };

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushConsts);

    VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &setLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcr;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCheck(vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout),
            "vkCreatePipelineLayout");

    // ---- 6) Create shader module from SPIR-V ----
    auto code = readFile(kShaderPath);

    VkShaderModuleCreateInfo smInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smInfo.codeSize = code.size();
    smInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    vkCheck(vkCreateShaderModule(device, &smInfo, nullptr, &shaderModule),
            "vkCreateShaderModule");

    // ---- 7) Create compute pipeline ----
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shaderModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpInfo.stage = stage;
    cpInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &pipeline),
            "vkCreateComputePipelines");

    // ---- 8) Descriptor pool + set ----
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    VkDescriptorPool descPool = VK_NULL_HANDLE;
    vkCheck(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool),
            "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &setLayout;

    VkDescriptorSet descSet = VK_NULL_HANDLE;
    vkCheck(vkAllocateDescriptorSets(device, &allocInfo, &descSet),
            "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo inInfo{};
    inInfo.buffer = inBuf.buffer;
    inInfo.offset = 0;
    inInfo.range  = bytes;

    VkDescriptorBufferInfo outInfo{};
    outInfo.buffer = outBuf.buffer;
    outInfo.offset = 0;
    outInfo.range  = bytes;

    VkWriteDescriptorSet writes[2]{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &inInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &outInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

    // ---- 9) Command pool + command buffer ----
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.queueFamilyIndex = computeQueueFamily;
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vkCheck(vkCreateCommandPool(device, &cp, nullptr, &cmdPool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cbAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAlloc.commandPool = cmdPool;
    cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkCheck(vkAllocateCommandBuffers(device, &cbAlloc, &cmd), "vkAllocateCommandBuffers");

    // Record commands
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkCheck(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &descSet, 0, nullptr);

    const uint32_t groupsX = (N + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
    PushConsts pc{N};
    vkCmdPushConstants(cmd, pipelineLayout,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   0, sizeof(PushConsts), &pc);
    vkCmdDispatch(cmd, groupsX, 1, 1);

    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    // ---- 10) Submit + wait ----
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    vkCheck(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
    vkCheck(vkQueueWaitIdle(queue), "vkQueueWaitIdle");

    // ---- 11) Read output ----
    {
        void* mapped = nullptr;
        vkCheck(vkMapMemory(device, outBuf.memory, 0, bytes, 0, &mapped), "vkMapMemory(readback)");
        float* f = reinterpret_cast<float*>(mapped);

        std::cout << "First 10 outputs:\n";
        for (uint32_t i = 0; i < 10; i++) {
            std::cout << i << " -> " << f[i] << "\n";
        }
        vkUnmapMemory(device, outBuf.memory);
    }

    // ---- 12) Cleanup ----
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);

    vkDestroyBuffer(device, inBuf.buffer, nullptr);
    vkFreeMemory(device, inBuf.memory, nullptr);
    vkDestroyBuffer(device, outBuf.buffer, nullptr);
    vkFreeMemory(device, outBuf.memory, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}
