#include "build_command.h"
#include "vk_util.h"
#include "export_mesh_command.h"

#include <vulkan/vulkan.h>
#include <iostream>
#include <string>
#include <vector>


//for args for building
static BuildArgs parseBuildArgs(int argc, char** argv) {
    BuildArgs a;
    a.heightmapPath = "src/assets/hm.png";
    a.outDir = "out/world";

    for (int i = 2; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--heightmap" && i + 1 < argc) a.heightmapPath = argv[++i];
        else if (s == "--out" && i + 1 < argc) a.outDir = argv[++i];
        else if (s == "--lods" && i + 1 < argc) a.lodCount = (uint32_t)std::stoul(argv[++i]);
        // tileSize fixed to 256 per your request
    }
    return a;
}

//for args for exporting meshs
static ExportMeshArgs parseExportArgs(int argc, char** argv) {
ExportMeshArgs a;

    for (int i = 2; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--in" && i + 1 < argc) a.inDir = argv[++i];
        else if (s == "--open-blender") a.openBlender = true;
        else if (s == "--blender" && i + 1 < argc) a.blenderPath = argv[++i];

        else if (s == "--out" && i + 1 < argc) a.outDir = argv[++i];
        else if (s == "--lods" && i + 1 < argc) a.lodCount = (uint32_t)std::stoul(argv[++i]);
        else if (s == "--scale" && i + 1 < argc) a.heightScale = std::stof(argv[++i]);
        else if (s == "--spacing" && i + 1 < argc) a.spacing = std::stof(argv[++i]);
    }
    return a;
}



int main(int argc, char** argv) {
    //set args to find with cmd
    if (argc < 2) {
        std::cout << "Usage:\n"
          << "  auroraterrian.exe build --heightmap path --out out/world --lods 5\n"
          << "  auroraterrian.exe export_mesh --in out/world --out out/meshes --lods 5 --scale 100 --spacing 1\n";

        return 0;
    }

    std::string cmd = argv[1];

// export arg
if (cmd == "export_mesh") {
    ExportMeshArgs args = parseExportArgs(argc, argv);
    try {
        return runExportMeshCommand(args);
    } catch (const std::exception& e) {
        std::cerr << "export_mesh error: " << e.what() << "\n";
        return 1;
    }
}


    // --- 1) Vulkan init ---
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "AuroraTerrain";
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


    // --- 2) find devices ---
    uint32_t devCount = 0;
    vkCheck(vkEnumeratePhysicalDevices(instance, &devCount, nullptr), "vkEnumeratePhysicalDevices(count)");
    if (devCount == 0) {
        std::cerr << "No Vulkan physical devices found.\n";
        return 1;
    }
    std::vector<VkPhysicalDevice> devs(devCount);
    vkCheck(vkEnumeratePhysicalDevices(instance, &devCount, devs.data()), "vkEnumeratePhysicalDevices(list)");

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = UINT32_MAX;

    for (auto d : devs) {
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qProps(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qCount, qProps.data());

        for (uint32_t i = 0; i < qCount; i++) {
            if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice = d;
                computeQueueFamily = i;
                break;
            }
        }
        if (physicalDevice) break;
    }

    if (!physicalDevice) { std::cerr << "No compute-capable GPU found.\n"; return 1; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qInfo.queueFamilyIndex = computeQueueFamily;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &prio;

    VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;

    VkDevice device = VK_NULL_HANDLE;
    vkCheck(vkCreateDevice(physicalDevice, &devInfo, nullptr, &device), "vkCreateDevice");

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, computeQueueFamily, 0, &queue);

    int rc = 0;

    {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(physicalDevice, &p);
        std::cout << "Using GPU: " << p.deviceName << "\n";
        std::cout << "Compute queue family: " << computeQueueFamily << "\n";
    }


    if (cmd == "build") {
        BuildArgs args = parseBuildArgs(argc, argv);
        rc = runBuildCommand(device, physicalDevice, queue, computeQueueFamily, args);
    }   else {
        std::cerr << "Unknown command: " << cmd << "\n";
        return 1;
    }

    
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return rc;
}
