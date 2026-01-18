#include "build_command.h"
#include "vk_util.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./third_party/stb_image_write.h"
#include "./third_party/stb_image.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

/*
build_command.cpp

1) Load Heightmap (also convert 16-bit to 32-bit. makes it easier for GPU)
2) Create Descriptor + Extract pipeline layouts. Create extract pipeline
3) Create buffers. Put hmBuff info into GPU memory
4) Create CMD pool and CMD buffer
5) Tile Loop. Dispatch 16x16 work groups in parallel. 16x16 threads each workgroups. Total 65536 invocations.
6) Clear
*/

//helpers
static void loadHeightmap16(const std::string &path, uint32_t &w, uint32_t &h, std::vector<uint16_t> &out)
{
    int iw = 0, ih = 0, c = 0;
    // stbi_load_16 gives 16-bit per channel
    uint16_t *img = stbi_load_16(path.c_str(), &iw, &ih, &c, 1);
    if (!img)
        throw std::runtime_error("Failed to load 16-bit heightmap: " + path);

    w = (uint32_t)iw;
    h = (uint32_t)ih;
    out.assign(img, img + (size_t)w * (size_t)h);
    stbi_image_free(img);
}
static void ensureDir(const std::string &path)
{
    std::filesystem::create_directories(std::filesystem::path(path));
}
static void writeRawU16(const std::string &path, const std::vector<uint16_t> &data)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Failed to write: " + path);
    f.write(reinterpret_cast<const char *>(data.data()),
            (std::streamsize)(data.size() * sizeof(uint16_t)));
}
// push constant structs
struct PCExtract
{
    uint32_t hmWidth; //tell GPU how wide orignical big img is to calc where next row starts
    uint32_t tileX;//tell gpu which exact sqr to cut
    uint32_t tileY;
};

struct PCDownsample
{
    uint32_t inSize;
};

static constexpr uint32_t TILE_SIZE = 256;
static constexpr uint32_t LOCAL_X = 16;
static constexpr uint32_t LOCAL_Y = 16;
static uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// -- Run Build Command *Parallel Computing step in stage 5
int runBuildCommand(VkDevice device,
                    VkPhysicalDevice physicalDevice,
                    VkQueue queue,
                    uint32_t computeQueueFamily,
                    const BuildArgs& args)
{
    // ---- 1) Load heightmap ----
    uint32_t hmW = 0, hmH = 0;
    std::vector<uint16_t> hmU16;
    loadHeightmap16(args.heightmapPath, hmW, hmH, hmU16);

    if (hmW == 0 || hmH == 0) throw std::runtime_error("Heightmap has 0 size.");
    if ((hmW % TILE_SIZE) != 0 || (hmH % TILE_SIZE) != 0) {
        throw std::runtime_error("Heightmap width/height must be divisible by 256 for now.");
    }

    const uint32_t tilesX = hmW / TILE_SIZE;
    const uint32_t tilesY = hmH / TILE_SIZE;

    ensureDir(args.outDir);
    ensureDir(args.outDir + "/tiles");

    // ---- 2) Create layouts + pipelines ----
    VkDescriptorSetLayout setLayout = makeSetLayout(device);
    // push constants: PCExtract is 12 bytes, so we will reseve 16 bytes
    VkPipelineLayout pipelineLayout = makePipelineLayout(device, setLayout, 16);
    //tell OS theses are empty rn
    VkShaderModule modExtract = VK_NULL_HANDLE;
    VkShaderModule modDown = VK_NULL_HANDLE;
    //create pipelines
    VkPipeline pipeExtract = makeComputePipeline(device, pipelineLayout,
        "../shaders/extract_tile.comp.spv", &modExtract);
    VkPipeline pipeDownsample = makeComputePipeline(device, pipelineLayout,
        "../shaders/downsample.comp.spv", &modDown);

    // ---- 2) Create buffers. Put hmBuff info into GPU memory ----
    const VkMemoryPropertyFlags hostMem =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    const VkDeviceSize hmBytes = sizeof(uint32_t) * (VkDeviceSize)hmW * (VkDeviceSize)hmH;
    const VkDeviceSize tileBytesMax = sizeof(uint32_t) * (VkDeviceSize)TILE_SIZE * (VkDeviceSize)TILE_SIZE;
    //Giant map. Holds entire 256x256 heightmap. source
    Buffer hmBuf  = createBuffer(device, physicalDevice, hmBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMem);
    //Has the small 256x256 cutout 
    Buffer tileA  = createBuffer(device, physicalDevice, tileBytesMax,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMem);
    //LOD downsampling from tile A. GPU reads this
    Buffer tileB  = createBuffer(device, physicalDevice, tileBytesMax,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMem);

    // upload hmU16 -> hmU32 -> hmBuf (easier for GPU when u32)
    std::vector<uint32_t> hmU32((size_t)hmW * (size_t)hmH);
    for (size_t i = 0; i < hmU32.size(); i++) hmU32[i] = (uint32_t)hmU16[i];

    {//Copy to GPU memory. Then unmap after. Block scope to make mapped local
        void* mapped = nullptr;
        vkCheck(vkMapMemory(device, hmBuf.memory, 0, hmBytes, 0, &mapped), "vkMapMemory(heightmap)");
        std::memcpy(mapped, hmU32.data(), (size_t)hmBytes);
        vkUnmapMemory(device, hmBuf.memory);
    }

    // ---- 3) Descriptor pool + descriptor set (descriptor: ptr from GPU's center to a buffer) ----
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &ps;

    VkDescriptorPool descPool = VK_NULL_HANDLE;
    vkCheck(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &setLayout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    vkCheck(vkAllocateDescriptorSets(device, &ai, &set), "vkAllocateDescriptorSets");
    //note: the B in inB stands for Buffer not tileB. Reused for every buffer
    auto updateSet2Buffers = [&](VkBuffer inB, VkDeviceSize inSize,
                                 VkBuffer outB, VkDeviceSize outSize)
    {
        VkDescriptorBufferInfo inInfo{ inB, 0, inSize };
        VkDescriptorBufferInfo outInfo{ outB, 0, outSize };

        VkWriteDescriptorSet w[2]{};

        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = set;
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[0].pBufferInfo = &inInfo;

        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = set;
        w[1].dstBinding = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[1].pBufferInfo = &outInfo;
        vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    };

    // ---- 4) Command pool + command buffer (provide GPU to-do list) ----
    VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpInfo.queueFamilyIndex = computeQueueFamily;
    cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vkCheck(vkCreateCommandPool(device, &cpInfo, nullptr, &cmdPool), "vkCreateCommandPool");//get cmdPool

    VkCommandBufferAllocateInfo cbAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbAlloc.commandPool = cmdPool; //set up cbAlloc
    cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkCheck(vkAllocateCommandBuffers(device, &cbAlloc, &cmd), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    auto submitAndWait = [&]() { //concurency
        vkCheck(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit");
        vkCheck(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
    };

    // ---- 5) Tile loop ----
    std::cout << "Building tiles: " << tilesX << " x " << tilesY
              << " | LODs=" << args.lodCount << " | tileSize=256\n";
    //If heightmap is 256x256 then there will be 16x16 = 256  workgorups. each workgroup has 16x16 threads which mean 65536 threads
    for (uint32_t ty = 0; ty < tilesY; ty++) {
        for (uint32_t tx = 0; tx < tilesX; tx++) {
            const std::string tileDir = args.outDir + "/tiles/tile_" + std::to_string(tx) + "_" + std::to_string(ty);
            ensureDir(tileDir);

            // --- LOD0 extract: hmBuf -> tileA (256x256) ---
            updateSet2Buffers(hmBuf.buffer, hmBytes, tileA.buffer, tileBytesMax);

            PCExtract pcE{ hmW, tx, ty };

            vkCheck(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer"); //clear and get new cmd
            vkCheck(vkBeginCommandBuffer(cmd, &beginInfo), "vkBeginCommandBuffer");

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeExtract);//tell GPU with math program to run
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PCExtract), &pcE);

            const uint32_t gx = ceilDiv(TILE_SIZE, LOCAL_X); //see how many group will be need if one thread will cover 16 x-axis px
            const uint32_t gy = ceilDiv(TILE_SIZE, LOCAL_Y); //see how many group will be need if one thread will cover 16 y-axis px
            vkCmdDispatch(cmd, gx, gy, 1); //Mecha-man disbatches **parallelism stage**

            vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
            submitAndWait();

            // Read back LOD0 (256*256 u32) -> write u16 raw
            std::vector<uint32_t> tileU32(TILE_SIZE * TILE_SIZE);
            {
                void* mapped = nullptr;
                vkCheck(vkMapMemory(device, tileA.memory, 0, tileBytesMax, 0, &mapped), "vkMapMemory(tileA)");
                std::memcpy(tileU32.data(), mapped, (size_t)tileBytesMax);
                vkUnmapMemory(device, tileA.memory);
            }
            std::vector<uint16_t> tileOutU16(tileU32.size());
            for (size_t i = 0; i < tileU32.size(); i++) tileOutU16[i] = (uint16_t)tileU32[i];
            writeRawU16(tileDir + "/lod0.height.raw", tileOutU16); //write to disk
        }
    }

    // ---- 6) Cleanup ----
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);

    vkDestroyPipeline(device, pipeExtract, nullptr);
    vkDestroyPipeline(device, pipeDownsample, nullptr);
    vkDestroyShaderModule(device, modExtract, nullptr);
    vkDestroyShaderModule(device, modDown, nullptr);

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);

    vkDestroyBuffer(device, hmBuf.buffer, nullptr);
    vkFreeMemory(device, hmBuf.memory, nullptr);

    vkDestroyBuffer(device, tileA.buffer, nullptr);
    vkFreeMemory(device, tileA.memory, nullptr);

    vkDestroyBuffer(device, tileB.buffer, nullptr);
    vkFreeMemory(device, tileB.memory, nullptr);

    std::cout << "Build done: " << args.outDir << "\n";
    return 0;
}
