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
class ComponentManager
{
public:
    /**
     * @brief Construct a new ComponentManager object
     *
     */
    ComponentManager()
    {
    }

    /**
     * @brief Destroy the ComponentManager object
     *
     */

    ~ComponentManager()
    {
    }

    /**
     * @brief Set the USD stage for this ComponentManager
     *
     * @param stage
     */
    virtual void initialize(pxr::UsdStageWeakPtr stage)
    {
        mStage = stage;
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
    virtual void initComponents() = 0;

    /**
     * @brief Optional function that runs after start is pressed
     *
     */
    virtual void onStart()
    {
    }

    /**
     * @brief Optional function that runs after stop is pressed
     *
     */
    virtual void onStop()
    {
    }
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
    virtual void onComponentChange(const pxr::UsdPrim& prim) = 0;

    /**
     * @brief Delete component
     *
     * @param prim
     */
    virtual void onComponentRemove(const pxr::SdfPath& primPath) = 0;

    /** Remove all components and perform cleanup
     * @brief
     *
     */
    virtual void deleteAllComponents() = 0;

    /**
     * @brief Get the Stage object
     *
     * @return pxr::UsdStageWeakPtr
     */
    pxr::UsdStageWeakPtr getStage()
    {
        return mStage;
    }

protected:
    pxr::UsdStageWeakPtr mStage = nullptr;

    double mTimeSeconds = 0; // current time in seconds
    int64_t mTimeNanoSeconds = 0; // current time in nano seconds
    double mTimeDelta = 0; // delta time for current tick
};
}
}
}
