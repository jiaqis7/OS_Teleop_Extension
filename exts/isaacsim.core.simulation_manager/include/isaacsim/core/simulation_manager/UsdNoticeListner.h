// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// #include <omni/usd/UsdContextIncludes.h>
// #include <omni/usd/UsdContext.h>
// #include <omni/usd/UsdUtils.h>
// clang-format off
#include <pch/UsdPCH.h>
// clang-format on
#include <physxSchema/physxSceneAPI.h>
#include <pxr/base/tf/notice.h>
#include <pxr/pxr.h>

#include <functional>
#include <vector>

namespace isaacsim
{
namespace core
{
namespace simulation_manager
{

class UsdNoticeListener : public pxr::TfWeakBase
{
public:
    UsdNoticeListener();
    void handle(const pxr::UsdNotice::ObjectsChanged& objectsChanged);
    void enable(const bool& flag);
    std::map<int, std::function<void(const std::string&)>>& getDeletionCallbacks();
    std::map<int, std::function<void(const std::string&)>>& getPhysicsSceneAdditionCallbacks();
    std::map<pxr::SdfPath, pxr::PhysxSchemaPhysxSceneAPI>& getPhysicsScenes();
    int& getCallbackIter();

private:
    std::map<pxr::SdfPath, pxr::PhysxSchemaPhysxSceneAPI> m_physicsScenes;
    std::map<int, std::function<void(const std::string&)>> m_deletionCallbacks;
    std::map<int, std::function<void(const std::string&)>> m_physicsSceneAdditionCallbacks;
    int m_callbackIter;
    bool m_enableFlag;
};

} // namespace isaacsim
} // namespace core
} // namespace simulation_manager
