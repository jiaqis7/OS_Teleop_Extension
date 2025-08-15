// Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include "Component.h"
#include "ComponentManager.h"

#include <omni/usd/UtilsIncludes.h>
//
#include <omni/fabric/usd/PathConversion.h>
#include <omni/usd/UsdUtils.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaacsim
{
namespace core
{
namespace utils
{

// This custom Usd notice listener is required to clean up properly
class UsdNoticeListener : public omni::usd::UsdUtils::UsdNoticeListener<pxr::UsdNotice::ObjectsChanged>

{
public:
    UsdNoticeListener(ComponentManager* manager) : mManager(manager)
    {
        mStage = mManager->getStage();
    }

    virtual void handleNotice(const pxr::UsdNotice::ObjectsChanged& objectsChanged) override
    {
        if (mStage != objectsChanged.GetStage())
        {
            return;
        }

        for (auto& path : objectsChanged.GetResyncedPaths())
        {
            if (path.IsAbsoluteRootOrPrimPath())
            {
                // CARB_LOG_WARN("ResyncedPaths: %s", path.GetText());
                const auto& primPath = (path == PXR_NS::SdfPath::AbsoluteRootPath() ? path : path.GetPrimPath());

                // If prim is removed, remove it and its descendants from selection.
                pxr::UsdPrim prim = mStage->GetPrimAtPath(primPath);

                // CARB_LOG_INFO("Prim %s valid %d", primPath.GetString().c_str(), prim.IsValid());
                if (prim.IsValid() == false) // removed prim
                {
                    mManager->onComponentRemove(primPath);
                }
            }
        }
        for (auto& path : objectsChanged.GetChangedInfoOnlyPaths())
        {
            auto primPath =
                mStage->GetPseudoRoot().GetPath() == path ? mStage->GetPseudoRoot().GetPath() : path.GetPrimPath();

            // Update the component attached to this prim, onComponentChange checks to see if the prim exists in
            // this BridgeApplication
            pxr::UsdPrim prim = mStage->GetPrimAtPath(primPath);
            mManager->onComponentChange(prim);
        }
    }

private:
    pxr::UsdStageWeakPtr mStage = nullptr;
    ComponentManager* mManager = nullptr;
};


template <class ComponentType>
class BridgeApplicationBase : public ComponentManager, public pxr::TfWeakBase
{
public:
    /**
     * @brief Construct a new Application object
     *
     */
    BridgeApplicationBase()
    {
    }

    /**
     * @brief Destroy the Isaac Application object
     *
     */

    ~BridgeApplicationBase()
    {
        mNoticeListener.reset();
        deleteAllComponents();
    }

    /**
     * @brief Set the USD stage for this application
     *
     * @param stage
     */
    virtual void initialize(pxr::UsdStageWeakPtr stage)
    {
        mStage = stage;
        mNoticeListener = std::make_unique<UsdNoticeListener>(this);
        mNoticeListener->registerListener();
    }

    /**
     * @brief Tick the application and all components
     * Pure virtual, must be defined by the child class
     * @param dt
     */
    virtual void tick(double dt) = 0;

    /**
     * @brief Initialize components from the current stage
     *
     */
    virtual void initComponents()
    {
        PXR_NS::UsdStageCache& cache = PXR_NS::UsdUtilsStageCache::Get();
        omni::fabric::UsdStageId stageId = { static_cast<uint64_t>(cache.GetId(mStage).ToLongInt()) };
        omni::fabric::IStageReaderWriter* iStageReaderWriter =
            carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
        omni::fabric::StageReaderWriterId stageInProgress = iStageReaderWriter->get(stageId);
        usdrt::UsdStageRefPtr usdrtStage = usdrt::UsdStage::Attach(stageId, stageInProgress);

        const std::vector<std::string> componentIsAVector = getComponentIsAVector();
        for (const std::string& componentIsA : componentIsAVector)
        {
            const std::vector<usdrt::SdfPath> componentPaths =
                usdrtStage->GetPrimsWithTypeName(usdrt::TfToken(componentIsA));

            for (const usdrt::SdfPath& usdrtPath : componentPaths)
            {
                const omni::fabric::PathC pathC(usdrtPath);
                const pxr::SdfPath usdPath = omni::fabric::toSdfPath(pathC);
                pxr::UsdPrim prim = mStage->GetPrimAtPath(usdPath);

                onComponentAdd(prim);
            }
        }
    }

    virtual std::vector<std::string> getComponentIsAVector() const = 0;


    /**
     * @brief Create a supported component in this application
     * Pure virtual, must be defined by the child class
     * @param prim
     */
    virtual void onComponentAdd(const pxr::UsdPrim& prim) = 0;

    /**
     * @brief Update properties of this prim (onComponentChange)
     *
     * @param prim
     */
    virtual void onComponentChange(const pxr::UsdPrim& prim)
    {
        // update properties of this prim (onComponentChange)
        if (mComponents.find(prim.GetPath().GetString()) != mComponents.end())
        {
            mComponents[prim.GetPath().GetString()]->onComponentChange();
        }
    }

    /**
     * @brief Call any components that are only updated when physics steps occur
     *
     * @param dt
     */
    virtual void onPhysicsStep(float dt)
    {
    }

    /**
     * @brief Delete component
     *
     * @param prim
     */
    virtual void onComponentRemove(const pxr::SdfPath& primPath)
    {
        std::unique_lock<std::mutex> lck(mComponentMtx);
        // Delete component for any children of this prim
        for (auto it = mComponents.begin(); it != mComponents.end();)
        {
            // CARB_LOG_WARN("Check: Prim %s %s", primPath.GetString().c_str(), it->first.c_str());
            // if ((it->first).find(primPath.GetString()) != std::string::npos)
            if (pxr::SdfPath(it->first).HasPrefix(primPath))
            {
                CARB_LOG_INFO("Delete: Prim %s %s", primPath.GetString().c_str(), it->first.c_str());
                it->second.reset();
                it = mComponents.erase(it);
            }
            else
            {
                it++;
            }
        }
    }

    /** Remove all components and perform cleanup
     * @brief
     *
     */
    virtual void deleteAllComponents()
    {
        std::unique_lock<std::mutex> lck(mComponentMtx);
        for (auto& component : mComponents)
        {
            component.second.reset();
        }
        mComponents.clear();
    }

protected:
    std::unordered_map<std::string, std::unique_ptr<ComponentType>> mComponents;
    std::unique_ptr<UsdNoticeListener> mNoticeListener;
    std::mutex mComponentMtx;
};

typedef BridgeApplicationBase<Component> BridgeApplication;


}
}
}
