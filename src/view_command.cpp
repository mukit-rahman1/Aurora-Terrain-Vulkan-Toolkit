// view_command.cpp (simple viewer: no MVP, CPU projects terrain into clip space)
#include "view_command.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

//math and mvp. No GLM
struct Vec3 { float x,y,z; };

static Vec3 v3(float x,float y,float z){ return {x,y,z}; }
static Vec3 sub(Vec3 a, Vec3 b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static Vec3 cross(Vec3 a, Vec3 b){
    return { a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x };
}
static float dot(Vec3 a, Vec3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static Vec3 norm(Vec3 v){
    float len = std::sqrt(dot(v,v));
    if (len <= 1e-8f) return v;
    return {v.x/len,v.y/len,v.z/len};
}

struct Mat4 { float m[16]; }; // column-major

static Mat4 identity(){
    Mat4 r{}; r.m[0]=r.m[5]=r.m[10]=r.m[15]=1.0f; return r;
}
static Mat4 mul(const Mat4& a, const Mat4& b){
    Mat4 r{};
    for(int c=0;c<4;c++){
        for(int r0=0;r0<4;r0++){
            r.m[c*4+r0] =
                a.m[0*4+r0]*b.m[c*4+0] +
                a.m[1*4+r0]*b.m[c*4+1] +
                a.m[2*4+r0]*b.m[c*4+2] +
                a.m[3*4+r0]*b.m[c*4+3];
        }
    }
    return r;
}

static Mat4 perspective(float fovyRad, float aspect, float zNear, float zFar){
    float f = 1.0f / std::tan(fovyRad * 0.5f);
    Mat4 r{};
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = zFar / (zNear - zFar);
    r.m[11] = -1.0f;
    r.m[14] = (zFar * zNear) / (zNear - zFar);
    return r;
}

static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up){
    Vec3 f = norm(sub(center, eye));
    Vec3 s = norm(cross(f, up));
    Vec3 u = cross(s, f);

    Mat4 r = identity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8]  = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9]  = u.z;
    r.m[2] =-f.x; r.m[6] =-f.y; r.m[10] =-f.z;

    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] =  dot(f, eye);
    return r;
}

static void die(const std::string& msg) {
    throw std::runtime_error(msg);
}

    //function for creating depth
    static VkFormat findDepthFormat(VkPhysicalDevice phys) {
        // minimal: try D32 first
        VkFormat fmt = VK_FORMAT_D32_SFLOAT;
        VkFormatProperties p{};
        vkGetPhysicalDeviceFormatProperties(phys, fmt, &p);
        if (p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) return fmt;

        // fallback
        fmt = VK_FORMAT_D24_UNORM_S8_UINT;
        vkGetPhysicalDeviceFormatProperties(phys, fmt, &p);
        if (p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) return fmt;

        die("No supported depth format");
        return VK_FORMAT_UNDEFINED;
    }



static void vkCheck(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        std::cerr << "Vulkan error: " << what << " (VkResult=" << (int)r << ")\n";
        throw std::runtime_error(what);
    }
}

static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t sz = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data(sz);
    f.read(data.data(), (std::streamsize)sz);
    return data;
}

static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    if (code.empty()) die("Shader file empty/missing");
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    vkCheck(vkCreateShaderModule(device, &ci, nullptr, &mod), "vkCreateShaderModule");
    return mod;
}

static uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    die("No suitable memory type");
    return 0;
}

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

static Buffer createBufferHostVisible(VkPhysicalDevice phys, VkDevice device,
                                     VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer b{};
    b.size = size;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device, &bi, nullptr, &b.buf), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, b.buf, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(
        phys, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    vkCheck(vkAllocateMemory(device, &ai, nullptr, &b.mem), "vkAllocateMemory");
    vkCheck(vkBindBufferMemory(device, b.buf, b.mem, 0), "vkBindBufferMemory");
    return b;
}

static void destroyBuffer(VkDevice device, Buffer& b) {
    if (b.buf) vkDestroyBuffer(device, b.buf, nullptr);
    if (b.mem) vkFreeMemory(device, b.mem, nullptr);
    b = {};
}

static std::vector<uint16_t> readRawU16(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t bytes = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes % 2 != 0) return {};
    std::vector<uint16_t> out(bytes / 2);
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)bytes);
    return out;
}

struct Vertex {
    float x, y, z;
};

static void buildGridMeshWorldSpace(const std::vector<uint16_t>& heights,
                                    uint32_t N,
                                    float heightScale,
                                    std::vector<Vertex>& outV,
                                    std::vector<uint32_t>& outI) {
    if (heights.size() != size_t(N) * size_t(N)) die("Height raw size doesn't match N*N");

    outV.resize(size_t(N) * size_t(N));
    outI.clear();
    outI.reserve(size_t(N - 1) * size_t(N - 1) * 6);

    const float inv = 1.0f / float(N - 1);

    for (uint32_t z = 0; z < N; z++) {
        for (uint32_t x = 0; x < N; x++) {
            float fx = float(x) * inv; // 0..1
            float fz = float(z) * inv; // 0..1
            float h  = float(heights[z * N + x]) / 65535.0f; // 0..1

            // Center terrain around origin, scale to ~[-0.5..0.5]
            float wx = fx - 0.5f;
            float wz = fz - 0.5f;

            // Height becomes real Y
            float wy = h * heightScale;

            outV[z * N + x] = { wx, wy, wz };
        }
    }

    for (uint32_t z = 0; z < N - 1; z++) {
        for (uint32_t x = 0; x < N - 1; x++) {
            uint32_t i0 = z * N + x;
            uint32_t i1 = z * N + (x + 1);
            uint32_t i2 = (z + 1) * N + x;
            uint32_t i3 = (z + 1) * N + (x + 1);

            outI.push_back(i0); outI.push_back(i2); outI.push_back(i1);
            outI.push_back(i1); outI.push_back(i2); outI.push_back(i3);
        }
    }
}


struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool complete() const { return graphics.has_value() && present.has_value(); }
};

static QueueFamilies findQueueFamilies(VkPhysicalDevice phys, VkSurfaceKHR surface) {
    QueueFamilies q{};

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props.data());

    for (uint32_t i = 0; i < count; i++) {
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) q.graphics = i;

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(phys, i, surface, &presentSupport);
        if (presentSupport) q.present = i;

        if (q.complete()) break;
    }
    return q;
}

static VkSurfaceFormatKHR pickSurfaceFormat(VkPhysicalDevice phys, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &count, fmts.data());

    // Prefer SRGB if possible
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return fmts[0];
}

static VkPresentModeKHR pickPresentMode(VkPhysicalDevice phys, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &count, modes.data());

    // Mailbox if available, else FIFO
    for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D pickExtent(const VkSurfaceCapabilitiesKHR& caps, GLFWwindow* win) {
    if (caps.currentExtent.width != UINT32_MAX) return caps.currentExtent;
    int w = 0, h = 0;
    glfwGetFramebufferSize(win, &w, &h);
    VkExtent2D e{};
    e.width  = std::clamp<uint32_t>((uint32_t)w, caps.minImageExtent.width, caps.maxImageExtent.width);
    e.height = std::clamp<uint32_t>((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height);
    return e;
}

int runViewCommand(const ViewArgs& args) {
    (void)args;

    // ---- Input paths (keep it simple, adjust if you want CLI later) ----
    std::string hPath = "./build/out/world/tiles/tile_0_0/lod0.height.raw";
    uint32_t lod = 0; // only used if you later build path from --lod
    (void)lod;

    // ---- Load heights ----
    auto heights = readRawU16(hPath);
    if (heights.empty()) {
        std::cerr << "FAILED to read height raw file: " << hPath << "\n";
        return 1;
    }
    std::cout << "Loaded heights: " << heights.size() << " samples from " << hPath << "\n";

    // infer N from size (assumes square)
    uint32_t N = (uint32_t)(std::sqrt((double)heights.size()) + 0.5);
    if (size_t(N) * size_t(N) != heights.size()) {
        std::cerr << "Heights not a perfect square.\n";
        return 1;
    }

    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    buildGridMeshWorldSpace(heights, N, /*heightScale=*/0.35f, verts, indices);
    std::cout << "Mesh verts: " << verts.size() << " indices: " << indices.size() << "\n";

    // ---- GLFW ----
    if (!glfwInit()) die("glfwInit failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1200, 800, "AuroraTerrain Viewer (simple)", nullptr, nullptr);
    if (!window) die("glfwCreateWindow failed");

    // ---- Vulkan instance ----
    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "AuroraTerrain";
    ai.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    ai.pEngineName = "None";
    ai.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    ai.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    ici.enabledExtensionCount = glfwExtCount;
    ici.ppEnabledExtensionNames = glfwExts;

    VkInstance instance = VK_NULL_HANDLE;
    vkCheck(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");

    // ---- Surface ----
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    vkCheck(glfwCreateWindowSurface(instance, window, nullptr, &surface), "glfwCreateWindowSurface");

    // ---- Pick physical device ----
    uint32_t devCount = 0;
    vkGetPhysicalDeviceProperties2; // keep compiler happy if headers are weird
    vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
    if (devCount == 0) die("No Vulkan devices found");
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(instance, &devCount, devs.data());

    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties pickedProps{};

    auto scoreDevice = [&](VkPhysicalDevice d) -> int {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        QueueFamilies q = findQueueFamilies(d, surface);
        if (!q.complete()) return -1;

        // must support swapchain
        uint32_t fmtCount = 0, pmCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(d, surface, &fmtCount, nullptr);
        vkGetPhysicalDeviceSurfacePresentModesKHR(d, surface, &pmCount, nullptr);
        if (fmtCount == 0 || pmCount == 0) return -1;

        int score = 0;
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
        score += (int)p.limits.maxImageDimension2D;
        return score;
    };

    int best = -1;
    for (auto d : devs) {
        int s = scoreDevice(d);
        if (s > best) { best = s; phys = d; vkGetPhysicalDeviceProperties(d, &pickedProps); }
    }
    if (!phys) die("No suitable physical device");

    std::cout << "Viewer GPU: " << pickedProps.deviceName << "\n";

    QueueFamilies qf = findQueueFamilies(phys, surface);

    // ---- Device + queues ----
    float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qcis;

    uint32_t gfxFamily = qf.graphics.value();
    uint32_t presentFamily = qf.present.value();

    if (gfxFamily == presentFamily) {
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = gfxFamily;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        qcis.push_back(qci);
    } else {
        VkDeviceQueueCreateInfo qci1{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci1.queueFamilyIndex = gfxFamily;
        qci1.queueCount = 1;
        qci1.pQueuePriorities = &prio;
        qcis.push_back(qci1);

        VkDeviceQueueCreateInfo qci2{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci2.queueFamilyIndex = presentFamily;
        qci2.queueCount = 1;
        qci2.pQueuePriorities = &prio;
        qcis.push_back(qci2);
    }

    const char* devExts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = (uint32_t)qcis.size();
    dci.pQueueCreateInfos = qcis.data();
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = devExts;

    VkDevice device = VK_NULL_HANDLE;
    vkCheck(vkCreateDevice(phys, &dci, nullptr, &device), "vkCreateDevice");

    VkQueue graphicsQ = VK_NULL_HANDLE, presentQ = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, gfxFamily, 0, &graphicsQ);
    vkGetDeviceQueue(device, presentFamily, 0, &presentQ);

    // ---- Swapchain ----
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, &caps);

    VkSurfaceFormatKHR sFmt = pickSurfaceFormat(phys, surface);
    VkPresentModeKHR pMode = pickPresentMode(phys, surface);
    VkExtent2D extent = pickExtent(caps, window);

    VkFormat depthFmt = findDepthFormat(phys);

    //depth resources
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthMem = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;

    VkImageCreateInfo ici2{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici2.imageType = VK_IMAGE_TYPE_2D;
    ici2.extent = { extent.width, extent.height, 1 };
    ici2.mipLevels = 1;
    ici2.arrayLayers = 1;
    ici2.format = depthFmt;
    ici2.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici2.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici2.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici2.samples = VK_SAMPLE_COUNT_1_BIT;

    vkCheck(vkCreateImage(device, &ici2, nullptr, &depthImage), "vkCreateImage(depth)");

    VkMemoryRequirements dreq{};
    vkGetImageMemoryRequirements(device, depthImage, &dreq);

    VkMemoryAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    dai.allocationSize = dreq.size;
    dai.memoryTypeIndex = findMemoryType(phys, dreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkCheck(vkAllocateMemory(device, &dai, nullptr, &depthMem), "vkAllocateMemory(depth)");
    vkCheck(vkBindImageMemory(device, depthImage, depthMem, 0), "vkBindImageMemory(depth)");

    VkImageViewCreateInfo dvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    dvci.image = depthImage;
    dvci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    dvci.format = depthFmt;
    dvci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    dvci.subresourceRange.levelCount = 1;
    dvci.subresourceRange.layerCount = 1;
    vkCheck(vkCreateImageView(device, &dvci, nullptr, &depthView), "vkCreateImageView(depth)");


    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface;
    sci.minImageCount = imageCount;
    sci.imageFormat = sFmt.format;
    sci.imageColorSpace = sFmt.colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t qIndices[] = { gfxFamily, presentFamily };
    if (gfxFamily != presentFamily) {
        sci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        sci.queueFamilyIndexCount = 2;
        sci.pQueueFamilyIndices = qIndices;
    } else {
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = pMode;
    sci.clipped = VK_TRUE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    vkCheck(vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain), "vkCreateSwapchainKHR");

    uint32_t scImgCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &scImgCount, nullptr);
    std::vector<VkImage> scImages(scImgCount);
    vkGetSwapchainImagesKHR(device, swapchain, &scImgCount, scImages.data());

    // ---- Image views ----
    std::vector<VkImageView> views(scImgCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < scImgCount; i++) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = scImages[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = sFmt.format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        vkCheck(vkCreateImageView(device, &vci, nullptr, &views[i]), "vkCreateImageView");
    }

    // ---- Render pass ----
    VkAttachmentDescription color{};
    color.format = sFmt.format;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format = depthFmt;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    VkAttachmentDescription atts[2] = { color, depth };

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;


    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 2;
    rpci.pAttachments = atts;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    vkCheck(vkCreateRenderPass(device, &rpci, nullptr, &renderPass), "vkCreateRenderPass");

    // ---- Framebuffers ----
    std::vector<VkFramebuffer> framebuffers(scImgCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < scImgCount; i++) {
        VkImageView attachments[2] = { views[i], depthView };

        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = renderPass;
        fci.attachmentCount = 2;
        fci.pAttachments = attachments;
        fci.width = extent.width;
        fci.height = extent.height;
        fci.layers = 1;
        vkCheck(vkCreateFramebuffer(device, &fci, nullptr, &framebuffers[i]), "vkCreateFramebuffer");
    }
    

    // ---- Pipeline (load shaders) ----
    auto vert = readFile("./shaders/terrain.vert.spv");
    if (vert.empty()) vert = readFile("../shaders/terrain.vert.spv");
    auto frag = readFile("./shaders/terrain.frag.spv");
    if (frag.empty()) frag = readFile("../shaders/terrain.frag.spv");

    if (vert.empty() || frag.empty()) {
        std::cerr << "Missing shaders. Compile and place SPIR-V at:\n"
                  << "  out/shaders/terrain.vert.spv\n"
                  << "  out/shaders/terrain.frag.spv\n"
                  << "or build/out/shaders/...\n";
        return 1;
    }

    VkShaderModule vs = createShaderModule(device, vert);
    VkShaderModule fs = createShaderModule(device, frag);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(Mat4);

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;

    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    vkCheck(vkCreatePipelineLayout(device, &plci, nullptr, &pipeLayout), "vkCreatePipelineLayout");

    VkVertexInputBindingDescription bind{};
    bind.binding = 0;
    bind.stride = sizeof(Vertex);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attr{};
    attr.location = 0;
    attr.binding = 0;
    attr.format = VK_FORMAT_R32G32B32_SFLOAT;
    attr.offset = 0;

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &bind;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = &attr;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Dynamic viewport/scissor
    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE; // keep it simple
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cbAtt;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.stencilTestEnable = VK_FALSE;

    //define gpci (Graphics Pipeline Create Info)
    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb;
    gpci.pDynamicState = &dyn;
    gpci.layout = pipeLayout;
    gpci.renderPass = renderPass;
    gpci.subpass = 0;


    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCheck(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline),
            "vkCreateGraphicsPipelines");

    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);

    // ---- Upload vertex/index buffers (host-visible: simplest) ----
    Buffer vbo = createBufferHostVisible(phys, device, verts.size() * sizeof(Vertex),
                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    Buffer ibo = createBufferHostVisible(phys, device, indices.size() * sizeof(uint32_t),
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    {
        void* p = nullptr;
        vkCheck(vkMapMemory(device, vbo.mem, 0, vbo.size, 0, &p), "vkMapMemory(vbo)");
        std::memcpy(p, verts.data(), (size_t)vbo.size);
        vkUnmapMemory(device, vbo.mem);
    }
    {
        void* p = nullptr;
        vkCheck(vkMapMemory(device, ibo.mem, 0, ibo.size, 0, &p), "vkMapMemory(ibo)");
        std::memcpy(p, indices.data(), (size_t)ibo.size);
        vkUnmapMemory(device, ibo.mem);
    }

    // ---- Command pool + command buffers ----
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = gfxFamily;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vkCheck(vkCreateCommandPool(device, &cpci, nullptr, &cmdPool), "vkCreateCommandPool");

    std::vector<VkCommandBuffer> cmdBufs(scImgCount, VK_NULL_HANDLE);
    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = cmdPool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = scImgCount;
    vkCheck(vkAllocateCommandBuffers(device, &cbi, cmdBufs.data()), "vkAllocateCommandBuffers");

    auto record = [&](uint32_t i, const Mat4& mvp) {
        vkResetCommandBuffer(cmdBufs[i], 0);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkCheck(vkBeginCommandBuffer(cmdBufs[i], &bi), "vkBeginCommandBuffer");

        VkClearValue clears[2]{};
        clears[0].color = {{0.05f, 0.05f, 0.10f, 1.0f}};
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = renderPass;
        rbi.framebuffer = framebuffers[i];
        rbi.renderArea.offset = {0, 0};
        rbi.renderArea.extent = extent;
        rbi.clearValueCount = 2;
        rbi.pClearValues = clears;

        vkCmdBeginRenderPass(cmdBufs[i], &rbi, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmdBufs[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        // push MVP here (every time recording)
        vkCmdPushConstants(cmdBufs[i], pipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &mvp);

        VkViewport viewport{};
        viewport.x = 0; viewport.y = 0;
        viewport.width  = (float)extent.width;
        viewport.height = (float)extent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D sc{};
        sc.offset = {0, 0};
        sc.extent = extent;

        vkCmdSetViewport(cmdBufs[i], 0, 1, &viewport);
        vkCmdSetScissor(cmdBufs[i], 0, 1, &sc);

        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmdBufs[i], 0, 1, &vbo.buf, &off);
        vkCmdBindIndexBuffer(cmdBufs[i], ibo.buf, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(cmdBufs[i], (uint32_t)indices.size(), 1, 0, 0, 0);

        vkCmdEndRenderPass(cmdBufs[i]);
        vkCheck(vkEndCommandBuffer(cmdBufs[i]), "vkEndCommandBuffer");
    };

    // ---- Sync: 2 frames in flight + renderFinished per swapchain image ----
    const uint32_t MAX_FRAMES = 2;

    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    std::vector<VkSemaphore> imageAvailable(MAX_FRAMES, VK_NULL_HANDLE);
    std::vector<VkFence> inFlightFences(MAX_FRAMES, VK_NULL_HANDLE);

    // per-swapchain-image renderFinished to avoid reuse VUIDs
    std::vector<VkSemaphore> renderFinished(scImgCount, VK_NULL_HANDLE);
    std::vector<VkFence> imagesInFlight(scImgCount, VK_NULL_HANDLE);

    for (uint32_t i = 0; i < MAX_FRAMES; i++) {
        vkCheck(vkCreateSemaphore(device, &si, nullptr, &imageAvailable[i]), "vkCreateSemaphore(imageAvailable)");
        vkCheck(vkCreateFence(device, &fi, nullptr, &inFlightFences[i]), "vkCreateFence(inFlightFences)");
    }
    for (uint32_t i = 0; i < scImgCount; i++) {
        vkCheck(vkCreateSemaphore(device, &si, nullptr, &renderFinished[i]), "vkCreateSemaphore(renderFinished[i])");
    }

    uint32_t frame = 0;

    // ---- Main loop ----
    while (!glfwWindowShouldClose(window)) {
        
        glfwPollEvents();

        vkCheck(vkWaitForFences(device, 1, &inFlightFences[frame], VK_TRUE, UINT64_MAX), "vkWaitForFences");

        uint32_t imgIndex = 0;
        VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailable[frame], VK_NULL_HANDLE, &imgIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR) break;
        vkCheck(acq, "vkAcquireNextImageKHR");

        // ---- Build MVP (per-frame) ----
        float aspect = extent.width / (float)extent.height;

        Mat4 proj = perspective(60.0f * 3.1415926f / 180.0f, aspect, 0.05f, 10.0f);
        // Vulkan NDC: flip Y
        proj.m[5] *= -1.0f;

        Vec3 target = v3(0, 0, 0);
        Vec3 eye    = v3(1.2f, 0.9f, 1.2f);
        Mat4 view   = lookAt(eye, target, v3(0,1,0));
        Mat4 model  = identity();

        Mat4 mvp = mul(proj, mul(view, model));

        // record this swapchain image with this frame’s MVP
        record(imgIndex, mvp);


        if (imagesInFlight[imgIndex] != VK_NULL_HANDLE) {
            vkCheck(vkWaitForFences(device, 1, &imagesInFlight[imgIndex], VK_TRUE, UINT64_MAX),
                    "vkWaitForFences(imagesInFlight)");
        }
        imagesInFlight[imgIndex] = inFlightFences[frame];

        vkCheck(vkResetFences(device, 1, &inFlightFences[frame]), "vkResetFences");

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &imageAvailable[frame];
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmdBufs[imgIndex];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished[imgIndex];

        vkCheck(vkQueueSubmit(graphicsQ, 1, &submit, inFlightFences[frame]), "vkQueueSubmit");

        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &renderFinished[imgIndex];
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain;
        present.pImageIndices = &imgIndex;

        VkResult pres = vkQueuePresentKHR(presentQ, &present);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) break;
        vkCheck(pres, "vkQueuePresentKHR");

        frame = (frame + 1) % MAX_FRAMES;
    }

    vkDeviceWaitIdle(device);

    // ---- Cleanup ----
    for (auto f : inFlightFences) if (f) vkDestroyFence(device, f, nullptr);
    for (auto s : imageAvailable) if (s) vkDestroySemaphore(device, s, nullptr);
    for (auto s : renderFinished) if (s) vkDestroySemaphore(device, s, nullptr);

    vkFreeCommandBuffers(device, cmdPool, (uint32_t)cmdBufs.size(), cmdBufs.data());
    vkDestroyCommandPool(device, cmdPool, nullptr);

    destroyBuffer(device, ibo);
    destroyBuffer(device, vbo);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeLayout, nullptr);

    for (auto fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (auto v : views) vkDestroyImageView(device, v, nullptr);

    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroyDevice(device, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
