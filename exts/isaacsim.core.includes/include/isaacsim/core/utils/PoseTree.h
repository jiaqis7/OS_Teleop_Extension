// Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#pragma once

#include "Conversions.h"
#include "Pose.h"
#include "UsdUtilities.h"

#include <foundation/PxTransform.h>
#include <isaacSensorSchema/isaacRtxLidarSensorAPI.h>
#include <omni/usd/UsdUtils.h>
#include <physx/include/foundation/PxTransform.h>
#include <usdrt/scenegraph/usd/rt/xformable.h>

#include <DynamicControl.h>


using namespace omni::isaac::dynamic_control;
using namespace isaacsim::core::utils::conversions;
using namespace isaacsim::core::utils;
namespace isaacsim
{
namespace core
{

namespace utils
{

namespace posetree
{


class PoseTree
{
public:
    PoseTree(const uint64_t& stageId, omni::isaac::dynamic_control::DynamicControl* dynamicControlPtr)
    {
        // Store the USD and USDRT stage references from the stage ID
        mUsdStage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(static_cast<long>(stageId)));
        mUsdrtStage = usdrt::UsdStage::Attach({ (stageId) });

        mDynamicControlPtr = dynamicControlPtr;
    }
    /**
     * @brief Set the Parent Prim Path
     *
     * @param parentPath SDF path for parent prim
     * @param parentFrame frame name for parent prim
     */
    void setParentPrimPath(const pxr::SdfPath& parentPath, const std::string& parentFrame)
    {
        mParentPath = parentPath;
        mParentFrame = parentFrame;
    }

    /**
     * @brief Set the Target Prim Paths
     *
     * @param targets list of sdk prim paths
     */
    void setTargetPrimPaths(const pxr::SdfPathVector& targets)
    {
        mTargets = targets;
    }

    /**
     * @brief traverses the full pose tree and calls processTransform for each transform
     *
     * @param processTransform user defined function that can be used to handle a new transform
     */
    void processAllFrames(
        std::function<void(const std::string&, const std::string&, const ::physx::PxTransform&)>& processTransform)
    {
        // If the parent prim path is not empty, get the type of prim and its pose.
        if (!mParentPath.IsEmpty())
        {
            DcObjectType type = mDynamicControlPtr->peekObjectType(mParentPath.GetString().c_str());
            if (type == eDcObjectRigidBody)
            {
                mParentPose = getRigidBodyPose(mParentPath);
            }
            else if (type == eDcObjectNone || type == eDcObjectArticulation)
            {
                mParentPose = getXformPose(mParentPath);
            }

            mParentFrame = getUniqueFrameName(GetName(mUsdStage->GetPrimAtPath(mParentPath)), mParentPath.GetString());
        }
        // For each target prim determine its type and compute the associated poses
        for (pxr::SdfPath primPath : mTargets)
        {
            DcObjectType type = mDynamicControlPtr->peekObjectType(primPath.GetString().c_str());
            if (type == eDcObjectArticulation)
            {
                DcHandle artculationHandle = mDynamicControlPtr->getArticulation(primPath.GetString().c_str());
                DcHandle rootBody = mDynamicControlPtr->getArticulationRootBody(artculationHandle);
                ::physx::PxTransform body1Pose = asPxTransform(mDynamicControlPtr->getRigidBodyPose(rootBody));

                std::string framePath(mDynamicControlPtr->getRigidBodyPath(rootBody));
                std::string bodyName = GetName(mUsdStage->GetPrimAtPath(pxr::SdfPath(framePath)));

                if (!mParentPath.IsEmpty())
                {
                    body1Pose = mParentPose.transformInv(body1Pose);
                }
                std::string childFrameId = getUniqueFrameName(bodyName, framePath);
                if (mParentFrame != childFrameId)
                {
                    // articulations always have an extra transform to the base link/rigid body
                    processTransform(mParentFrame, childFrameId, body1Pose);
                }
                size_t numDofs = mDynamicControlPtr->getArticulationBodyCount(artculationHandle);
                for (size_t j = 0; j < numDofs; j++)
                {
                    DcHandle parentBody = mDynamicControlPtr->getArticulationBody(artculationHandle, j);
                    ::physx::PxTransform body0Pose = asPxTransform(mDynamicControlPtr->getRigidBodyPose(parentBody));
                    std::string parentPath(mDynamicControlPtr->getRigidBodyPath(parentBody));
                    std::string parentName = GetName(mUsdStage->GetPrimAtPath(pxr::SdfPath(parentPath)));
                    size_t numJoints = mDynamicControlPtr->getRigidBodyChildJointCount(parentBody);
                    for (size_t k = 0; k < numJoints; k++)
                    {
                        DcHandle joint = mDynamicControlPtr->getRigidBodyChildJoint(parentBody, k);
                        DcHandle child_body = mDynamicControlPtr->getJointChildBody(joint);


                        ::physx::PxTransform body1Pose = asPxTransform(mDynamicControlPtr->getRigidBodyPose(child_body));
                        ::physx::PxTransform body0Tbody1(body0Pose.transformInv(body1Pose));
                        std::string framePath(mDynamicControlPtr->getRigidBodyPath(child_body));
                        auto bodyName = GetName(mUsdStage->GetPrimAtPath(pxr::SdfPath(framePath)));

                        processTransform(getUniqueFrameName(parentName, parentPath),
                                         getUniqueFrameName(bodyName, framePath), body0Tbody1);
                    }
                }
            }
            else if (type == eDcObjectRigidBody)
            {
                ::physx::PxTransform body1Pose = getRigidBodyPose(primPath);

                std::string childFrameId =
                    getUniqueFrameName(GetName(mUsdStage->GetPrimAtPath(primPath)), primPath.GetString());
                if (mParentFrame != childFrameId)
                {
                    if (!mParentPath.IsEmpty())
                    {
                        body1Pose = mParentPose.transformInv(body1Pose);
                    }


                    processTransform(mParentFrame, childFrameId, body1Pose);
                }
            }
            else if (type == eDcObjectNone)
            {
                pxr::UsdPrim prim = mUsdStage->GetPrimAtPath(primPath);

                ::physx::PxTransform body1Pose = getXformPose(primPath);


                if (prim.IsA<pxr::UsdGeomCamera>() && !prim.HasAPI<pxr::IsaacSensorIsaacRtxLidarSensorAPI>())
                {
                    // Regular camera (not RTXLidar), Rotate 180 degrees about x-axis
                    // pxr::GfMatrix4d(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);
                    ::physx::PxQuat omniTCamera(1, 0, 0, 0);
                    body1Pose = body1Pose * ::physx::PxTransform(omniTCamera);
                }

                if (!mParentPath.IsEmpty())
                {
                    body1Pose = mParentPose.transformInv(body1Pose);
                }

                processTransform(mParentFrame, getUniqueFrameName(GetName(prim), primPath.GetString()), body1Pose);
            }
        }
    }

    /**
     * @brief Get the Rigid Body Pose of a prim from physics
     *
     * @param path
     * @return ::physx::PxTransform
     */
    ::physx::PxTransform getRigidBodyPose(const pxr::SdfPath& path)
    {
        DcHandle rigidBodyHandle = mDynamicControlPtr->getRigidBody(path.GetString().c_str());
        return asPxTransform(mDynamicControlPtr->getRigidBodyPose(rigidBodyHandle));
    }
    /**
     * @brief Get the pose of the prim via fabric if it exists, or usd
     *
     * @param path path to the prim
     * @return ::physx::PxTransform
     */
    ::physx::PxTransform getXformPose(const pxr::SdfPath& path)
    {
        return asPxTransform(isaacsim::core::utils::pose::computeWorldXformNoCache(mUsdStage, mUsdrtStage, path));
    }
    /**
     * @brief Get the unique name to use for frame. if two frames have the same name the full usd path with underscores
     * is used.
     *
     * @param frame
     * @param path
     * @return std::string
     */
    std::string getUniqueFrameName(const std::string& frame, const std::string& path)
    {
        std::string name(frame);
        if (mRenamedFrames.find(path) != mRenamedFrames.end())
        {
            mPublishedFrames[frame] = true;
            return mRenamedFrames[path];
        }
        else if (mPublishedFrames.find(frame) == mPublishedFrames.end())
        {
            mRenamedFrames[path] = frame;
            mPublishedFrames[frame] = true;
        }

        else
        {
            name = path;
            std::replace(name.begin(), name.end(), '/', '_');
            name = name.substr(1);
            CARB_LOG_WARN(
                "Frame with name %s already exists. Overriding frame name for %s to %s (you can add the attribute isaac:nameOverride to remove this warning)",
                frame.c_str(), path.c_str(), name.c_str());
            mRenamedFrames[path] = name;
            mPublishedFrames[name] = true;
        }
        return name;
    }

private:
    pxr::SdfPath mParentPath;
    std::string mParentFrame;
    pxr::SdfPathVector mTargets;
    ::physx::PxTransform mParentPose = ::physx::PxTransform(::physx::PxIdentity);

    pxr::UsdStageRefPtr mUsdStage;
    usdrt::UsdStageRefPtr mUsdrtStage;
    omni::isaac::dynamic_control::DynamicControl* mDynamicControlPtr = nullptr;

    std::map<std::string, std::string> mRenamedFrames;
    std::map<std::string, bool> mPublishedFrames;
};
}
}
}
}
