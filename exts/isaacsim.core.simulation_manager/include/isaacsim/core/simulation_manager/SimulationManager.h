// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
/*
Carbonite SDK API:
  https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/api/carbonite_api.html
*/

#pragma once

#define CARB_EXPORTS

#ifdef _MSC_VER
#    if OMPRIMUTILSEXPORT
#        define DllExport __declspec(dllexport)
#    else
#        define DllExport __declspec(dllimport)
#    endif
#else
#    define DllExport
#endif


#include <carb/Defines.h>
#include <carb/Interface.h>

#include <pch/UsdPCH.h>
#include <pxr/pxr.h>

#include <cstdint>
#include <functional>

namespace isaacsim
{
namespace core
{
namespace simulation_manager
{

/**
 * Carbonite interface
 */
struct ISimulationManager
{
    CARB_PLUGIN_INTERFACE("isaacsim::core::simulation_manager::ISimulationManager", 1, 0);

    // ------------------
    // custom API declaration. E.g.:
    DllExport virtual int registerDeletionCallback(const std::function<void(std::string)>& callback) = 0;
    DllExport virtual int registerPhysicsSceneAdditionCallback(const std::function<void(std::string)>& callback) = 0;
    DllExport virtual bool deregisterCallback(const int& callbackId) = 0;
    DllExport virtual void reset() = 0;
    DllExport virtual int& getCallbackIter() = 0;
    DllExport virtual void setCallbackIter(int const& val) = 0;
    DllExport virtual void enableUsdNoticeHandler(bool const& flag) = 0;
    DllExport virtual void enableFabricUsdNoticeHandler(long stageId, bool const& flag) = 0;
    DllExport virtual bool isFabricUsdNoticeHandlerEnabled(long stageId) = 0;
    // ------------------
};

} // namespace isaacsim
} // namespace core
} // namespace simulation_manager
