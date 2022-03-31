#include "vkroots/vkroots.h"

#include <iostream>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <mutex>

#include <cups/cups.h>
#include <cups/ipp.h>

#include <condition_variable>
#include <algorithm>
#include <unistd.h>
#include <unordered_map>
#include <vulkan/vulkan_core.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
namespace CUPSPrintedSurface {

    template<typename T, typename U = T>
    constexpr T align(T what, U to)
    {
        return (what + to - 1) & ~(to - 1);
    }

    static std::vector<cups_dest_t>& GetPrinters()
    {
        static std::vector<cups_dest_t> s_printers = []() {
            cups_dest_t* printers = nullptr;
            const int count = cupsGetDests2(CUPS_HTTP_DEFAULT, &printers);
            return std::vector<cups_dest_t>{ printers, printers + count };
        }();

        return s_printers;
    }

    template <typename Func>
    static void DelimitStringView(std::string_view view, std::string_view delim, Func func)
    {
        size_t pos = 0;
        while ((pos = view.find(delim)) != std::string_view::npos) {
            std::string_view token = view.substr(0, pos);
            if (!func(token))
                return;
            view = view.substr(pos + 1);
        }
        func(view);
    }

    struct VkPrintedSurfaceImpl
    {
        cups_dest_t* dest;

        uint32_t     copyCount;

        bool         enableBind;
        bool         enableCollate;
        bool         enableColor;
        bool         enableCover;
        bool         enableDuplex;
        bool         enableHolePunch;
        bool         enableSorting;
        bool         enableStaple;
    };
    static std::mutex s_PrintedSurfaceMutex;
    static std::vector<std::unique_ptr<VkPrintedSurfaceImpl>> s_PrintedSurfaces;

    struct VkPrintedImage
    {
        VkImage        image;
        VkDeviceMemory memory;
        void*          cpuPtr;
        VkFence        fence;

        VkExtent3D     extent;

        std::thread             thread;
        std::condition_variable cv;
        std::mutex              mutex;
        bool                    busy;
    };

    struct VkPrintedSwapchainImpl
    {
        VkDevice                    device;
        VkPrintedSurfaceImpl*       surface;
        uint32_t                    nextImage;

        VkCompositeAlphaFlagBitsKHR compositeAlpha;

        std::vector<VkPrintedImage> swapchainImages;
    };
    static std::mutex s_PrintedSwapchainMutex;
    static std::vector<std::unique_ptr<VkPrintedSwapchainImpl>> s_PrintedSwapchains;

    // Lock held by calling function.
    static auto FindPrintedSurface(VkSurfaceKHR surface)
    {
        auto iter = std::find_if(s_PrintedSurfaces.begin(), s_PrintedSurfaces.end(),
            [&](const std::unique_ptr<VkPrintedSurfaceImpl>& x) { return reinterpret_cast<VkSurfaceKHR>(x.get()) == surface; });
        return iter;
    }

    static auto FindPrintedSwapchain(VkSwapchainKHR swapchain)
    {
        auto iter = std::find_if(s_PrintedSwapchains.begin(), s_PrintedSwapchains.end(),
            [&](const std::unique_ptr<VkPrintedSwapchainImpl>& x) { return reinterpret_cast<VkSwapchainKHR>(x.get()) == swapchain; });
        return iter;
    }

    template <typename T, typename ArrType, typename Op>
    static VkResult ReturnVulkanArray(ArrType& arr, uint32_t *pCount, T* pOut, Op func)
    {
        const uint32_t count = uint32_t(arr.size());

        if (!pOut) {
            *pCount = count;
            return VK_SUCCESS;
        }

        const uint32_t outCount = std::min(*pCount, count);
        for (uint32_t i = 0; i < outCount; i++)
            func(pOut[i], arr[i]);

        *pCount = outCount;
        return count != outCount
            ? VK_INCOMPLETE
            : VK_SUCCESS;
    }

    template <typename T, typename ArrType>
    static VkResult ReturnVulkanArray(ArrType& arr, uint32_t *pCount, T* pOut)
    {
        return ReturnVulkanArray(arr, pCount, pOut, [](T& x, const T& y) { x = y; });
    }

    class VkInstanceOverrides
    {
    public:
        // This is a core function so it's part of Instance dispatch.
        static VkResult GetPhysicalDeviceSurfaceSupportKHR(
            const vkroots::VkInstanceDispatch* pDispatch,
                  VkPhysicalDevice             physicalDevice,
                  uint32_t                     queueFamilyIndex,
                  VkSurfaceKHR                 surface,
                  VkBool32*                    pSupported)
        {
            std::unique_lock lock{ s_PrintedSurfaceMutex };

            auto iter = FindPrintedSurface(surface);
            if (iter == s_PrintedSurfaces.end()) {
                lock.unlock();
                return pDispatch->GetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface, pSupported);
            }

            *pSupported = VK_TRUE;
            return VK_SUCCESS;
        }

        static VkResult GetPhysicalDeviceSurfaceCapabilitiesKHR(
            const vkroots::VkInstanceDispatch* pDispatch,
                  VkPhysicalDevice             physicalDevice,
                  VkSurfaceKHR                 surface,
                  VkSurfaceCapabilitiesKHR*    pSurfaceCapabilities)
        {
            std::unique_lock lock{ s_PrintedSurfaceMutex };

            auto iter = FindPrintedSurface(surface);
            if (iter == s_PrintedSurfaces.end()) {
                lock.unlock();
                return pDispatch->GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, pSurfaceCapabilities);
            }

            *pSurfaceCapabilities = VkSurfaceCapabilitiesKHR {
                .minImageCount           = 1u,
                .maxImageCount           = 0u,
                .currentExtent           = VkExtent2D{ 0xFFFFFFFFu, 0xFFFFFFFFu },
                .minImageExtent          = VkExtent2D{ 1u, 1u },
                .maxImageExtent          = VkExtent2D{ 0x7FFFFFFFu, 0x7FFFFFFFu },
                .maxImageArrayLayers     = 1u,
                .supportedTransforms     = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                .currentTransform        = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                .supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR | VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                .supportedUsageFlags     = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                           VK_IMAGE_USAGE_SAMPLED_BIT |
                                           VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                           VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
            };

            return VK_SUCCESS;
        }

        static VkResult GetPhysicalDeviceSurfaceFormatsKHR(
            const vkroots::VkInstanceDispatch* pDispatch,
                  VkPhysicalDevice     physicalDevice,
                  VkSurfaceKHR         surface,
                  uint32_t*            pSurfaceFormatCount,
                  VkSurfaceFormatKHR*  pSurfaceFormats)
        {
            std::unique_lock lock{ s_PrintedSurfaceMutex };

            auto iter = FindPrintedSurface(surface);
            if (iter == s_PrintedSurfaces.end()) {
                lock.unlock();
                return pDispatch->GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats);
            }

            static constexpr std::array<VkSurfaceFormatKHR, 4> PrintedSurfaceFormats = {{
                { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
                { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
                { VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
                { VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
            }};

            return ReturnVulkanArray(PrintedSurfaceFormats, pSurfaceFormatCount, pSurfaceFormats);
        }

        static VkResult GetPhysicalDeviceSurfacePresentModesKHR(
            const vkroots::VkInstanceDispatch* pDispatch,
                  VkPhysicalDevice             physicalDevice,
                  VkSurfaceKHR                 surface,
                  uint32_t*                    pPresentModeCount,
                  VkPresentModeKHR*            pPresentModes)
        {
            std::unique_lock lock{ s_PrintedSurfaceMutex };

            auto iter = FindPrintedSurface(surface);
            if (iter == s_PrintedSurfaces.end()) {
                lock.unlock();
                return pDispatch->GetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, pPresentModeCount, pPresentModes);
            }

            static constexpr std::array<VkPresentModeKHR, 1> PresentModes = {{
                VK_PRESENT_MODE_FIFO_KHR
            }};

            return ReturnVulkanArray(PresentModes, pPresentModeCount, pPresentModes);
        }

        static void DestroySurfaceKHR(
            const vkroots::VkInstanceDispatch* pDispatch,
                  VkInstance                   instance,
                  VkSurfaceKHR                 surface,
                  const VkAllocationCallbacks* pAllocator)
        {
            std::unique_lock lock{ s_PrintedSurfaceMutex };

            auto iter = FindPrintedSurface(surface);
            if (iter == s_PrintedSurfaces.end()) {
                lock.unlock();
                return pDispatch->DestroySurfaceKHR(instance, surface, pAllocator);
            }

            s_PrintedSurfaces.erase(iter);
        }
    };

    class VkPhysicalDeviceOverrides
    {
    public:
        static VkResult EnumeratePrintersFOOL(
            const vkroots::VkPhysicalDeviceDispatch* pDispatch,
                  VkPhysicalDevice                   physicalDevice,
                  uint32_t*                          pPrinterCount,
                  VkPrinterFOOL*                     pPrinters)
        {
            auto& printers = GetPrinters();
            const uint32_t printerCount = uint32_t(printers.size());
            if (!pPrinters) {
                *pPrinterCount = printerCount;
                return VK_SUCCESS;
            }

            return ReturnVulkanArray(printers, pPrinterCount, pPrinters, [](VkPrinterFOOL& x, cups_dest_t& y) { x = reinterpret_cast<VkPrinterFOOL>(&y); });
        }

        static VkResult GetPrinterPropertiesFOOL(
            const vkroots::VkPhysicalDeviceDispatch* pDispatch,
                  VkPhysicalDevice                   physicalDevice,
                  VkPrinterFOOL                      printer,
                  VkPrinterPropertiesFOOL*           pProperties)
        {
            cups_dest_t *dest = reinterpret_cast<cups_dest_t*>(printer);

            const char* friendlyName = cupsGetOption("printer-info", dest->num_options, dest->options);
            if (!friendlyName)
                friendlyName = cupsGetOption("printer-make-and-model", dest->num_options, dest->options);


            cups_ptype_t type = 0;
            const char *printerTypeStr = cupsGetOption("printer-type", dest->num_options, dest->options);
            if (printerTypeStr)
                type = (cups_ptype_t)atoi(printerTypeStr);

            strncpy(pProperties->deviceName, friendlyName, std::size(pProperties->deviceName));

            pProperties->supportsBind      = !!(type & CUPS_PRINTER_BIND);
            pProperties->supportsCollate   = !!(type & CUPS_PRINTER_COLLATE);
            pProperties->supportsColor     = !!(type & CUPS_PRINTER_COLOR);
            pProperties->supportsCover     = !!(type & CUPS_PRINTER_COVER);
            pProperties->supportsDuplex    = !!(type & CUPS_PRINTER_DUPLEX);
            pProperties->supportsHolePunch = !!(type & CUPS_PRINTER_PUNCH);
            pProperties->supportsSorting   = !!(type & CUPS_PRINTER_SORT);
            pProperties->supportsStaple    = !!(type & CUPS_PRINTER_STAPLE);

            pProperties->isFaxQueue = !!(type & CUPS_PRINTER_BIND);

            return VK_SUCCESS;
        }

        static VkResult CreatePrintedSurfaceFOOL(
            const vkroots::VkPhysicalDeviceDispatch* pDispatch,
                  VkPhysicalDevice                   physicalDevice,
            const VkPrintedSurfaceCreateInfoFOOL*    pCreateInfo,
            const VkAllocationCallbacks*             pAllocator,
                  VkSurfaceKHR*                      pSurface) {
            std::unique_lock lock{ s_PrintedSurfaceMutex };

            s_PrintedSurfaces.emplace_back(
                std::make_unique<VkPrintedSurfaceImpl>(VkPrintedSurfaceImpl{
                .dest            = reinterpret_cast<cups_dest_t*>(pCreateInfo->printer),
                .copyCount       = pCreateInfo->copyCount,
                .enableBind      = !!pCreateInfo->enableBind,
                .enableCollate   = !!pCreateInfo->enableCollate,
                .enableColor     = !!pCreateInfo->enableColor,
                .enableCover     = !!pCreateInfo->enableCover,
                .enableDuplex    = !!pCreateInfo->enableDuplex,
                .enableHolePunch = !!pCreateInfo->enableHolePunch,
                .enableSorting   = !!pCreateInfo->enableSorting,
                .enableStaple    = !!pCreateInfo->enableStaple,
            }));

            *pSurface = reinterpret_cast<VkSurfaceKHR>(s_PrintedSurfaces.back().get());
            return VK_SUCCESS;
        }
    };

    static uint32_t please_for_the_love_of_god_find_me_the_hvv_memory_type_index(const vkroots::VkDeviceDispatch* pDispatch, VkDevice device)
    {
        VkPhysicalDevice physicalDevice = pDispatch->PhysicalDevice;
        VkPhysicalDeviceMemoryProperties props;
        pDispatch->pPhysicalDeviceDispatch->pInstanceDispatch->GetPhysicalDeviceMemoryProperties(physicalDevice, &props);
        for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
            const auto& type = props.memoryTypes[i];

            constexpr VkMemoryPropertyFlags flags =
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT   |
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT   |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            if ((type.propertyFlags & flags) == flags)
                return i;
        }

        return ~0u;
    }

    static bool PutInkOrTonerToPaper(VkPrintedSurfaceImpl* surface, VkPrintedSwapchainImpl* swapchain, VkPrintedImage* image, int* jobId)
    {
        // What even is a dinfo?
        // Who knows! All you do is you copy it and pass it in to this function for seemingly no reason.
        cups_dinfo_t* dinfo = cupsCopyDestInfo(CUPS_HTTP_DEFAULT, surface->dest);

        ipp_status_t status;
        if ((status = cupsCreateDestJob(CUPS_HTTP_DEFAULT, surface->dest, dinfo, jobId, "Printed Frame", 0, nullptr)) > IPP_STATUS_OK_IGNORED_OR_SUBSTITUTED) {
            fprintf(stderr, "Failed to create dest job: %s\n", cupsLastErrorString());
            cupsFreeDestInfo(dinfo);
            return false;
        }

		cups_option_t *options = nullptr;
        int num_options = 0;

        {
            char this_api_sucks[64];
            snprintf(this_api_sucks, 64, "%d", surface->copyCount);
            num_options = cupsAddOption(CUPS_COPIES, this_api_sucks, num_options, &options);
        }

        if (surface->enableBind)
            num_options = cupsAddOption(CUPS_FINISHINGS, CUPS_FINISHINGS_BIND, num_options, &options);

        if (surface->enableCover)
            num_options = cupsAddOption(CUPS_FINISHINGS, CUPS_FINISHINGS_COVER, num_options, &options);

        if (surface->enableHolePunch)
            num_options = cupsAddOption(CUPS_FINISHINGS, CUPS_FINISHINGS_PUNCH, num_options, &options);

        if (surface->enableStaple)
            num_options = cupsAddOption(CUPS_FINISHINGS, CUPS_FINISHINGS_STAPLE, num_options, &options);

        if (surface->enableColor)
            num_options = cupsAddOption(CUPS_PRINT_COLOR_MODE, CUPS_PRINT_COLOR_MODE_COLOR, num_options, &options);
        else
            num_options = cupsAddOption(CUPS_PRINT_COLOR_MODE, CUPS_PRINT_COLOR_MODE_MONOCHROME, num_options, &options);

        const char* png_is_missing_from_the_header_lol = "image/png";

        if (cupsStartDestDocument(CUPS_HTTP_DEFAULT, surface->dest, dinfo, *jobId, "Printed Frame", png_is_missing_from_the_header_lol, 0, nullptr, 1) != HTTP_STATUS_CONTINUE) {
            fprintf(stderr, "Failed to create document: %s\n", cupsLastErrorString());
            cupsFreeDestInfo(dinfo);
            return false;
        }

        int imageSize = image->extent.width * image->extent.height * 4;
        unsigned char* data = (unsigned char*) malloc(imageSize);
        memcpy(data, image->cpuPtr, imageSize);
        if (swapchain->compositeAlpha == VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
            for (int y = 0; y < image->extent.height; y++) {
                for (int x = 0; x < image->extent.width; x++) {
                    data[(y * image->extent.width * 4) + (x * 4) + 3] = 255;
                }
            }
        }

        int len = 0;
        unsigned char *png = stbi_write_png_to_mem(data, 0, image->extent.width, image->extent.height, 4, &len);
        free(data);
        if (!png || !len) {
            fprintf(stderr, "Failed to convert to png\n");
            return false;
        }

        if (cupsWriteRequestData(CUPS_HTTP_DEFAULT, (const char*)png, len) != HTTP_STATUS_CONTINUE) {
            fprintf(stderr, "Failed to write req data: %s\n", cupsLastErrorString());
            cupsFreeDestInfo(dinfo);
            STBIW_FREE(png);
            return false;
        }
        STBIW_FREE(png);

        if ((status = cupsFinishDestDocument(CUPS_HTTP_DEFAULT, surface->dest, dinfo)) > IPP_STATUS_OK_IGNORED_OR_SUBSTITUTED) {
            fprintf(stderr, "Unable to send document: %s\n", cupsLastErrorString());
            cupsFreeDestInfo(dinfo);
            return false;
        }

        // Oh boy, time to free the random thing we copied to pass in for no reason.
        cupsFreeDestInfo(dinfo);
        return true;
    }

    class VkDeviceOverrides
    {
    public:
        static VkResult CreateSwapchainKHR(
            const vkroots::VkDeviceDispatch* pDispatch,
                  VkDevice                   device,
            const VkSwapchainCreateInfoKHR*  pCreateInfo,
            const VkAllocationCallbacks*     pAllocator,
                  VkSwapchainKHR*            pSwapchain)
        {
            std::unique_lock lockSurface{ s_PrintedSurfaceMutex };

            auto iter = FindPrintedSurface(pCreateInfo->surface);
            if (iter == s_PrintedSurfaces.end()) {
                lockSurface.unlock();
                return pDispatch->CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
            }

            std::unique_lock lockSwapchain{ s_PrintedSwapchainMutex };

            const uint32_t hvvTypeIndex = please_for_the_love_of_god_find_me_the_hvv_memory_type_index(pDispatch, device);
            if (hvvTypeIndex == ~0u) {
                fprintf(stderr, "Failed to find HVV memory type.\n");
                return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            }

            VkPhysicalDeviceProperties physicalDeviceProps;
            pDispatch->pPhysicalDeviceDispatch->pInstanceDispatch->GetPhysicalDeviceProperties(pDispatch->PhysicalDevice, &physicalDeviceProps);

            const uint32_t imageCount = std::max(pCreateInfo->minImageCount, 1u);

            std::vector<VkPrintedImage> swapchainImages(imageCount);
            for (uint32_t i = 0; i < imageCount; i++) {
                VkImageCreateInfo info = {
                    .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    .imageType             = VK_IMAGE_TYPE_2D,
                    .format                = pCreateInfo->imageFormat,
                    .extent                = VkExtent3D{ pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height, 1u },
                    .mipLevels             = 1,
                    .arrayLayers           = 1,
                    .samples               = VK_SAMPLE_COUNT_1_BIT,
                    .tiling                = VK_IMAGE_TILING_LINEAR,
                    .usage                 = pCreateInfo->imageUsage,
                    .sharingMode           = pCreateInfo->imageSharingMode,
                    .queueFamilyIndexCount = pCreateInfo->queueFamilyIndexCount,
                    .pQueueFamilyIndices   = pCreateInfo->pQueueFamilyIndices,
                    .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                };

                VkResult res;
                VkImage image;
                if ((res = pDispatch->CreateImage(device, &info, pAllocator, &image)) != VK_SUCCESS) {
                    fprintf(stderr, "Failed to create swapchain image for printed surface.\n");
                    return res;
                }

                VkMemoryRequirements req;
                pDispatch->GetImageMemoryRequirements(device, image, &req);

                const uint32_t imageSize = align(req.size, physicalDeviceProps.limits.minMemoryMapAlignment);

                VkMemoryAllocateInfo memoryInfo = {
                    .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .allocationSize  = imageSize,
                    .memoryTypeIndex = hvvTypeIndex,
                };
                VkDeviceMemory memory;
                if ((res = pDispatch->AllocateMemory(device, &memoryInfo, nullptr, &memory)) != VK_SUCCESS) {
                    fprintf(stderr, "Failed to allocate memory for swapchain image for printed surface.\n");
                    return res;
                }

                if ((res = pDispatch->BindImageMemory(device, image, memory, 0)) != VK_SUCCESS) {
                    fprintf(stderr, "Failed to bind image memory.\n");
                    return res;
                }

                void *cpuPtr;
                if ((res = pDispatch->MapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &cpuPtr)) != VK_SUCCESS) {
                    fprintf(stderr, "Failed to map memory for swapchain image for printed surface.\n");
                    return res;
                }

                VkFenceCreateInfo fenceInfo = {
                    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                    .flags = VK_FENCE_CREATE_SIGNALED_BIT,
                };
                VkFence fence;
                if ((res = pDispatch->CreateFence(device, &fenceInfo, nullptr, &fence)) != VK_SUCCESS) {
                    fprintf(stderr, "Failed to create fence for swapchain image for printed surface.\n");
                    return res;
                }

                swapchainImages[i].image  = image;
                swapchainImages[i].memory = memory;
                swapchainImages[i].cpuPtr = cpuPtr;
                swapchainImages[i].fence  = fence;
                swapchainImages[i].extent = info.extent;
                swapchainImages[i].busy   = false;
            }

            s_PrintedSwapchains.emplace_back(
                std::make_unique<VkPrintedSwapchainImpl>(VkPrintedSwapchainImpl{
                .device          = device,
                .surface         = iter->get(),
                .compositeAlpha  = pCreateInfo->compositeAlpha,
                .swapchainImages = std::move(swapchainImages),
            }));

            VkPrintedSwapchainImpl* swapchain = s_PrintedSwapchains.back().get();
            *pSwapchain = reinterpret_cast<VkSwapchainKHR>(swapchain);
            
            // Start the image spool thread for this swapchain
            for (uint32_t i = 0; i < imageCount; i++) {
                swapchain->swapchainImages[i].thread = std::thread{ [pDispatch, cSurface = iter->get(), cDevice = device, cSwapchain = swapchain, cPrintedImage = &swapchain->swapchainImages[i]]()
                    {
                        for(;;)
                        {
                            // Wait for the image to become busy again.
                            {
                                std::unique_lock lock(cPrintedImage->mutex);
                                cPrintedImage->cv.wait(lock, [&]{ return cPrintedImage->busy; });
                            }

                            // Wait for the fence to be signalled.
                            VkResult res;
                            if ((res = pDispatch->WaitForFences(cDevice, 1, &cPrintedImage->fence, VK_TRUE, UINT64_MAX)) == VK_SUCCESS) {
                                int jobId = 0;
                                // Print the image!
                                if (PutInkOrTonerToPaper(cSurface, cSwapchain, cPrintedImage, &jobId))
                                {
                                    // Wait for the job to complete, spinny time!
                                    if (jobId)
                                    {
                                        bool completed = false;

                                        while (!completed)
                                        {
                                            // Query the number of completed jobs 66ms
                                            // if we are in that list, yipee! We are done!
                                            //
                                            // Given we probably have multiple swapchain images in flight
                                            // (therefore, multiple jobs in flight),
                                            // 66ms here probably wont cause stalling unless you have
                                            // a super super fast printer.
                                            usleep(66'000);
                                            cups_job_t *jobs = nullptr;
                                            int jobCount = cupsGetJobs2(CUPS_HTTP_DEFAULT, &jobs, cSurface->dest->name, 1, CUPS_WHICHJOBS_COMPLETED);
                                            for (int i = 0; i < jobCount; i++) {
                                                if (jobs[i].id == jobId)
                                                    completed = true;
                                            }
                                        }
                                        
                                    }
                                }
                                else
                                    fprintf(stderr, "Failed to print the image!\n");
                            } else
                                fprintf(stderr, "Failed to wait for print to finish.\n");

                            // Mark it as not busy anymore now that the work has completed.
                            {
                                std::unique_lock lock(cPrintedImage->mutex);
                                cPrintedImage->busy = false;
                            }
                            // Wakey wakey, time for school!
                            cPrintedImage->cv.notify_all();
                        }
                    }
                };
            }

            return VK_SUCCESS;
        }

        static VkResult GetSwapchainImagesKHR(
            const vkroots::VkDeviceDispatch* pDispatch,
                  VkDevice                   device,
                  VkSwapchainKHR             swapchain,
                  uint32_t*                  pSwapchainImageCount,
                  VkImage*                   pSwapchainImages)
        {
            std::unique_lock lock{ s_PrintedSwapchainMutex };

            auto iter = FindPrintedSwapchain(swapchain);
            if (iter == s_PrintedSwapchains.end()) {
                lock.unlock();
                return pDispatch->GetSwapchainImagesKHR(device, swapchain, pSwapchainImageCount, pSwapchainImages);
            }

            return ReturnVulkanArray((*iter)->swapchainImages, pSwapchainImageCount, pSwapchainImages, [](VkImage& x, const VkPrintedImage& y) { x = y.image; });
        }

        static VkResult QueuePresentKHR(
            const vkroots::VkDeviceDispatch* pDispatch,
                  VkQueue                    queue,
            const VkPresentInfoKHR*          pPresentInfo)
        {
            std::unique_lock lock{ s_PrintedSwapchainMutex };

            // TODO: Handle multiple pSwapchains
            auto iter = FindPrintedSwapchain(pPresentInfo->pSwapchains[0]);
            if (iter == s_PrintedSwapchains.end()) {
                lock.unlock();
                return pDispatch->QueuePresentKHR(queue, pPresentInfo);
            }
            auto& swapchain = *iter;

            const uint32_t imageIndex = pPresentInfo->pImageIndices[0];
            auto& printedImage = swapchain->swapchainImages[imageIndex];

            // Submit something to signal a fence when the waits for this is done
            // this is being waited on in the thread associated with the printed image.
            {
                std::vector<VkPipelineStageFlags> dstMasks(pPresentInfo->waitSemaphoreCount);
                for (auto& flags : dstMasks)
                    flags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

                const VkSubmitInfo submitInfo = {
                    .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .waitSemaphoreCount = pPresentInfo->waitSemaphoreCount,
                    .pWaitSemaphores    = pPresentInfo->pWaitSemaphores,
                    .pWaitDstStageMask  = dstMasks.data(),
                };
                pDispatch->QueueSubmit(queue, 1, &submitInfo, printedImage.fence);
            }

            {
                // Mark the image as busy.
                std::unique_lock lock(printedImage.mutex);
                printedImage.busy = true;
            }
            printedImage.cv.notify_all();

            return VK_SUCCESS;
        }

        static VkResult AcquireNextImage2KHR(
            const vkroots::VkDeviceDispatch* pDispatch,
            VkDevice                         device,
            const VkAcquireNextImageInfoKHR* pAcquireInfo,
            uint32_t*                        pImageIndex)
        {
            std::unique_lock lock{ s_PrintedSwapchainMutex };

            auto iter = FindPrintedSwapchain(pAcquireInfo->swapchain);
            if (iter == s_PrintedSwapchains.end()) {
                lock.unlock();
                return pDispatch->AcquireNextImage2KHR(device, pAcquireInfo, pImageIndex);
            }
            auto& swapchain = *iter;

            const uint32_t imageIdx = swapchain->nextImage;
            // Wait for the work for this image to be completed if any.
            {
                auto &printedImage = swapchain->swapchainImages[imageIdx];

                std::unique_lock lock(printedImage.mutex);
                printedImage.cv.wait(lock, [&]{ return !printedImage.busy; });
            }

            // Check the state of the printer...
            {
                cups_dest_t* dest = swapchain->surface->dest;

                const char *state = cupsGetOption("printer-state", dest->num_options, dest->options);
                // "5" = the printer did a whoopsie
                // this is such a good API, I love this

                if (state && !strcmp(state, "5"))
                {
                    const char* reasons = cupsGetOption("printer-state-reasons", dest->num_options, dest->options);

                    // Fall back to BAD_LUCK for any errors we don't know about.
                    VkResult res = VK_ERROR_BAD_LUCK_FOOL;

                    // who needs an enum anyway when you can have a delimited list
                    // of completely random undocumented strings!
                    DelimitStringView(std::string_view{ reasons }, ",", [&](std::string_view reason) -> bool
                    {
                        if (reason == "media-jam") {
                            res = VK_ERROR_PAPER_JAM_FOOL;
                            return false;
                        }
                        else if (reason == "media-empty" ||
                                 reason == "media-needed") {
                            res = VK_ERROR_OUT_OF_PAPER_FOOL;
                            return false;
                        }
                        else if (reason == "toner-empty") {
                            res = VK_ERROR_OUT_OF_TONER_FOOL;
                            return false;
                        }
                        else if (reason == "marker-supply-empty") {
                            res = VK_ERROR_OUT_OF_INK_FOOL;
                            return false;
                        }   
                        else if (reason == "spool-area-full") {
                            res = VK_ERROR_SPOOL_IDLE_FOOL;
                            return false;
                        }
                        else if (reason == "cups-missing-filter-warning" ||
                                 reason == "cups-insecure-filter-warning" ||
                                 reason == "interpreter-resource-unavailable") {
                            res = VK_ERROR_FILTER_FAILED_FOOL;
                            return false;
                        }

                        // Continue parsing the reasons...
                        return true;
                    });

                    return res;
                }
            }

            // Signal that acquiring is done with a dummy queue submission.
            {
                // TODO: Better queue hueristic.
                VkQueue queue;
                pDispatch->GetDeviceQueue(device, 0, 0, &queue);

                const VkSubmitInfo submitInfo = {
                    .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .signalSemaphoreCount = pAcquireInfo->semaphore ? 1u : 0u,
                    .pSignalSemaphores    = &pAcquireInfo->semaphore,
                };
                pDispatch->QueueSubmit(queue, 1, &submitInfo, pAcquireInfo->fence);
            }

            *pImageIndex = imageIdx;

            swapchain->nextImage = (imageIdx + 1) % swapchain->swapchainImages.size();

            return VK_SUCCESS;
        }

        static VkResult AcquireNextImageKHR(
            const vkroots::VkDeviceDispatch* pDispatch,
                  VkDevice                   device,
                  VkSwapchainKHR             swapchain,
                  uint64_t                   timeout,
                  VkSemaphore                semaphore,
                  VkFence                    fence,
                  uint32_t*                  pImageIndex)
        {
            // Forward this to AcquireNextImage2KHR.
            // If your driver doesn't support it by this point,
            // sucks to be you!
            VkAcquireNextImageInfoKHR acquireInfo = {
                .sType      = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
                .swapchain  = swapchain,
                .timeout    = timeout,
                .semaphore  = semaphore,
                .fence      = fence,
                .deviceMask = 0x1,
            };

            return AcquireNextImage2KHR(pDispatch, device, &acquireInfo, pImageIndex);
        }
    };

}

VKROOTS_DEFINE_LAYER_INTERFACES(CUPSPrintedSurface::VkInstanceOverrides,
                                CUPSPrintedSurface::VkPhysicalDeviceOverrides,
                                CUPSPrintedSurface::VkDeviceOverrides);
