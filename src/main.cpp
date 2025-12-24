#include <iostream>
#include <vulkan/vulkan.h>

int main() {
    std::cout << "Vulkan header version: " << VK_HEADER_VERSION << "\n";

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ci, nullptr, &instance);

    if (r != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed: " << r << "\n";
        return 1;
    }

    std::cout << "vkCreateInstance success\n";
    vkDestroyInstance(instance, nullptr);
    return 0;
}

