// Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

// clang-format off
#include <pch/UsdPCH.h>
// clang-format on

#include <carb/Defines.h>
#include <carb/Types.h>
#include <carb/events/EventsUtils.h>

#include <omni/usd/UsdContextIncludes.h>
//
#include <omni/timeline/ITimeline.h>
#include <omni/timeline/TimelineTypes.h>
#include <omni/usd/UsdContext.h>
/**
 * @brief Base class for nodes that automatically reset when stop is pressed.
 *
 */
class BaseResetNode
{

public:
    /**
     * @brief Construct a new object
     *
     */
    BaseResetNode()
    {
        // When the node is created, we create a stage event subscription
        // The idea is that node is reset whenever stop is pressed
        mTimeline = carb::getCachedInterface<omni::timeline::ITimeline>();
        mTimelineEventSub = carb::events::createSubscriptionToPopByType(
            mTimeline->getTimelineEventStream(),
            static_cast<carb::events::EventType>(omni::timeline::TimelineEventType::eStop),
            [this](carb::events::IEvent* e) { reset(); }, 0, "IsaacSimOGNTimelineEventHandler");
    }
    /**
     * @brief Destroy the object, clear the stage event subscription
     *
     */
    ~BaseResetNode()
    {
        mTimelineEventSub = nullptr;
    }

    /**
     * @brief This reset function is pure virtual and must be defined by the derived class;
     *
     */
    virtual void reset() = 0;

private:
    carb::events::ISubscriptionPtr mTimelineEventSub;
    omni::timeline::ITimeline* mTimeline = nullptr;
};
