// Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

// #include <carb/filesystem/IFileSystem.h>

#include <DynamicControl.h>
// clang-format off
#include <omni/usd/UtilsIncludes.h>
#include <omni/usd/UsdUtils.h>
// clang-format on

#include <isaacsim/core/utils/Conversions.h>


namespace isaacsim
{
namespace core
{
namespace utils
{
namespace transforms
{
using omni::isaac::dynamic_control::DcHandle;
using omni::isaac::dynamic_control::DcObjectType;
using omni::isaac::dynamic_control::DcTransform;
/**
 * @brief Set the transform of the object
 *
 * @param mDynamicControlPtr
 * @param prim
 * @param pxBodyTranslation
 * @param pxBodyRotation
 */
inline void setTransform(pxr::UsdPrim& prim, pxr::GfVec3f pxBodyTranslation, pxr::GfQuatf pxBodyRotation)
{
    // TODO: Handle world rotation as well
    // NOTE: reverting this for now, rigid body sink publishes global so teleport should be global too
    // auto newTranslation = pxBodyTranslation; // + parentToWorldMat.ExtractTranslation();
    omni::isaac::dynamic_control::DynamicControl* mDynamicControlPtr =
        carb::getCachedInterface<omni::isaac::dynamic_control::DynamicControl>();
    DcTransform t = isaacsim::core::utils::conversions::asDcTransform(pxBodyTranslation, pxBodyRotation);

    if (mDynamicControlPtr->isSimulating())
    {
        DcObjectType primType = mDynamicControlPtr->peekObjectType(prim.GetPath().GetString().c_str());
        if (primType == omni::isaac::dynamic_control::eDcObjectArticulation)
        {

            DcHandle artculationHandle = mDynamicControlPtr->getArticulation(prim.GetPath().GetString().c_str());
            mDynamicControlPtr->wakeUpArticulation(artculationHandle);
            DcHandle rigidBodyHandle = mDynamicControlPtr->getArticulationRootBody(artculationHandle);
            mDynamicControlPtr->setRigidBodyPose(rigidBodyHandle, t);
            mDynamicControlPtr->setRigidBodyLinearVelocity(rigidBodyHandle, { 0, 0, 0 });
            mDynamicControlPtr->setRigidBodyAngularVelocity(rigidBodyHandle, { 0, 0, 0 });
            mDynamicControlPtr->setRigidBodyPose(rigidBodyHandle, t);
            mDynamicControlPtr->setRigidBodyLinearVelocity(rigidBodyHandle, { 0, 0, 0 });
            mDynamicControlPtr->setRigidBodyAngularVelocity(rigidBodyHandle, { 0, 0, 0 });
            return;
        }
        else if (primType == omni::isaac::dynamic_control::eDcObjectRigidBody)
        {
            DcHandle rigidBodyHandle = mDynamicControlPtr->getRigidBody(prim.GetPath().GetString().c_str());
            mDynamicControlPtr->wakeUpRigidBody(rigidBodyHandle);
            mDynamicControlPtr->setRigidBodyPose(rigidBodyHandle, t);
            mDynamicControlPtr->setRigidBodyLinearVelocity(rigidBodyHandle, { 0, 0, 0 });
            mDynamicControlPtr->setRigidBodyAngularVelocity(rigidBodyHandle, { 0, 0, 0 });
            return;
        }
    }
    // In case we are not simulating or the object was a regular prim, go down this path
    {
        pxr::GfTransform usdBodyPose;
        usdBodyPose.SetTranslation(pxBodyTranslation);
        usdBodyPose.SetRotation(pxr::GfRotation(pxBodyRotation));
        // Pose is global so offset by parent pose
        pxr::GfMatrix4d parentToWorldMat =
            pxr::UsdGeomXformable(prim).ComputeParentToWorldTransform(pxr::UsdTimeCode::Default());
        omni::usd::UsdUtils::setLocalTransformMatrix(prim, usdBodyPose.GetMatrix() * parentToWorldMat.GetInverse());
    }
}

/**
 * @brief Sets the scale of the object
 *
 * @param prim
 * @param pxBodyScale
 */
inline void setScale(pxr::UsdPrim& prim, pxr::GfVec3f pxBodyScale)
{
    omni::isaac::dynamic_control::DynamicControl* mDynamicControlPtr =
        carb::getCachedInterface<omni::isaac::dynamic_control::DynamicControl>();
    bool doScale = true;

    if (mDynamicControlPtr->isSimulating())
    {
        DcObjectType primType = mDynamicControlPtr->peekObjectType(prim.GetPath().GetString().c_str());
        doScale = (primType == omni::isaac::dynamic_control::eDcObjectNone);
    }
    if (doScale)
    {
        auto currentTransformMat = omni::usd::UsdUtils::getLocalTransformMatrix(prim);
        pxr::GfMatrix4d scaleMat;
        scaleMat.SetScale(pxBodyScale);
        auto scaledTransformMat = scaleMat * currentTransformMat;
        omni::usd::UsdUtils::setLocalTransformMatrix(prim, scaledTransformMat);
    }
}
}
}
}
}
