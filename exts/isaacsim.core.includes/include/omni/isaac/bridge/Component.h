// Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once
#if defined(_WIN32)
#    include <usdrt/scenegraph/usd/usd/stage.h>
#else
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-variable"
#    include <usdrt/scenegraph/usd/usd/stage.h>
#    pragma GCC diagnostic pop
#endif

#include <string>
#include <vector>

namespace isaacsim
{
namespace core
{
namespace utils
{

/**
 * @brief Base class which defines a component in an Application that is attached to a USD prim
 */
template <class PrimType>
class ComponentBase
{
public:
    virtual ~ComponentBase()
    {
    }
    /**
     * @brief Set the USD prim and stage for this application
     *
     * @param prim
     * @param stage
     */
    virtual void initialize(const PrimType& prim, pxr::UsdStageWeakPtr stage)
    {
        mPrim = prim;
        mStage = stage;
        mDoStart = true;
        mUsdrtStage =
            usdrt::UsdStage::Attach({ static_cast<uint64_t>(pxr::UsdUtilsStageCache::Get().GetId(stage).ToLongInt()) });
    }

    /**
     * @brief Function that runs after start is pressed
     *
     */
    virtual void onStart() = 0;

    /** @brief Function that runs after stop is pressed
     *
     */
    virtual void onStop()
    {
    }
    /**
     * @brief Function that is called each physics step
     *
     */
    virtual void onPhysicsStep(float dt)
    {
    }

    /**
     * @brief Function that is called each frame that is rendered
     *
     */
    virtual void onRenderEvent()
    {
    }

    /**
     * @brief Called every frame
     *
     */
    virtual void tick() = 0;

    /**
     * @brief Called every time the Prim is changed
     *
     */
    virtual void onComponentChange() = 0;

    /**
     * @brief Update timestamps for component
     *
     * @param timeSeconds
     * @param dt
     * @param timeNano
     */
    virtual void updateTimestamp(double timeSeconds, double dt, int64_t timeNano)
    {
        this->mTimeSeconds = timeSeconds;
        this->mTimeDelta = dt;
        this->mTimeNanoSeconds = timeNano;
    }
    /**
     * @brief Get the USD Prim object
     *
     * @return PrimType&
     */
    PrimType& getPrim()
    {
        return mPrim;
    }
    /**
     * @brief Return value of enabled flag
     *
     * @return true
     * @return false
     */
    bool getEnabled()
    {
        return mEnabled;
    }

    uint64_t getSequenceNumber()
    {
        return mSequenceNumber;
    }
    bool mDoStart = true; // whether start should be called on this component

protected:
    // USD reference to prim that stores settings for this component
    PrimType mPrim;
    // USD stage that the prim is in
    pxr::UsdStageWeakPtr mStage = nullptr;
    usdrt::UsdStageRefPtr mUsdrtStage = nullptr;


    double mTimeSeconds = 0; // current time in seconds
    int64_t mTimeNanoSeconds = 0; // current time in nano seconds
    double mTimeDelta = 0; // delta time for current tick

    uint64_t mSequenceNumber = 0;
    bool mEnabled = true; // whether this component is enabled or not.
};

typedef ComponentBase<pxr::UsdPrim> Component;


}
}
}
