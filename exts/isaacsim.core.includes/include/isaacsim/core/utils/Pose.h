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
#include "UsdUtilities.h"

#include <foundation/PxTransform.h>
#include <omni/usd/UsdUtils.h>
#include <physx/include/foundation/PxTransform.h>

#if defined(_WIN32)
#    include <usdrt/scenegraph/usd/rt/xformable.h>
#else
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-variable"
#    include <usdrt/scenegraph/usd/rt/xformable.h>
#    pragma GCC diagnostic pop
#endif

using namespace isaacsim::core::utils::conversions;
using namespace isaacsim::core::utils;
namespace isaacsim
{
namespace core
{

namespace utils
{

namespace pose
{

static usdrt::GfMatrix4d computeWorldXformNoCache(pxr::UsdStageRefPtr usdStage,
                                                  usdrt::UsdStageRefPtr usdrtStage,
                                                  const pxr::SdfPath& path,
                                                  pxr::UsdTimeCode timecode = pxr::UsdTimeCode::Default())
{
    if (usdrtStage->HasPrimAtPath(path.GetString()))
    {
        usdrt::UsdPrim usdrtPrim = usdrtStage->GetPrimAtPath(path.GetString());
        usdrt::RtXformable usdrtXformable = usdrt::RtXformable(usdrtPrim);

        if (usdrtXformable.HasWorldXform())
        {
            usdrt::GfVec3d worldPos(0);
            usdrt::GfQuatf worldOrient(1);
            usdrt::GfVec3f worldScale(1);
            auto worldPositionAttribute = usdrtXformable.GetWorldPositionAttr();
            if (worldPositionAttribute.HasValue())
            {
                usdrtXformable.GetWorldPositionAttr().Get(&worldPos, usdrt::UsdTimeCode(timecode.GetValue()));
            }
            auto worldOrientationAttribute = usdrtXformable.GetWorldOrientationAttr();
            if (worldOrientationAttribute.HasValue())
            {
                usdrtXformable.GetWorldOrientationAttr().Get(&worldOrient, usdrt::UsdTimeCode(timecode.GetValue()));
            }
            auto worldScaleAttribute = usdrtXformable.GetWorldScaleAttr();
            if (worldScaleAttribute.HasValue())
            {
                usdrtXformable.GetWorldScaleAttr().Get(&worldScale, usdrt::UsdTimeCode(timecode.GetValue()));
            }
            usdrt::GfMatrix4d rot, scale, result{};
            scale.SetScale(usdrt::GfVec3d(worldScale));
            rot.SetRotate(usdrt::GfQuatd(worldOrient));
            result = scale * rot;
            result.SetTranslateOnly(worldPos);
            return result;
        }
        else if (usdrtXformable.HasLocalXform())
        {
            usdrt::GfMatrix4d localMat(1);
            usdrtXformable.GetLocalMatrixAttr().Get(&localMat, usdrt::UsdTimeCode(timecode.GetValue()));
            pxr::UsdPrim parentPrim = usdStage->GetPrimAtPath(path).GetParent();
            usdrt::GfMatrix4d parentXform = usdrt::GfMatrix4d(1.0);
            if (parentPrim)
            {
                parentXform = computeWorldXformNoCache(usdStage, usdrtStage, parentPrim.GetPath(), timecode);
            }

            return localMat * parentXform;
        }
    }
    {
        pxr::UsdPrim prim = usdStage->GetPrimAtPath(path);

        usdrt::GfMatrix4d localMat(1.0);
        if (PXR_NS::UsdGeomXformable xformable = PXR_NS::UsdGeomXformable(prim))
        {
            bool dontCare;
            xformable.GetLocalTransformation(reinterpret_cast<PXR_NS::GfMatrix4d*>(&localMat), &dontCare, timecode);
        }
        pxr::UsdPrim parentPrim = prim.GetParent();
        usdrt::GfMatrix4d parentXform = usdrt::GfMatrix4d(1.0);
        if (parentPrim)
        {
            parentXform = computeWorldXformNoCache(usdStage, usdrtStage, parentPrim.GetPath(), timecode);
        }
        return localMat * parentXform;
    }
}
/**
 * @brief Get the relative transform matrix from the source prim ot the target prim
 *
 * @param sourcePrim: source prim from which frame to compute the relative transform
 * @param targetPrim: target prim to which fram to compute the relative transform
 *
 * @return usdrt/Gf/Matrix4d
 */
inline usdrt::GfMatrix4d getRelativeTransform(pxr::UsdStageRefPtr usdStage,
                                              usdrt::UsdStageRefPtr usdrtStage,
                                              const pxr::SdfPath& sourcePrim,
                                              const pxr::SdfPath& targetPrim)
{
    pxr::UsdStagePtr stage = omni::usd::UsdContext::getContext()->getStage();
    // column major transform matrix
    usdrt::GfMatrix4d sourceToWorldColumnMajorTransform = computeWorldXformNoCache(usdStage, usdrtStage, sourcePrim);
    usdrt::GfMatrix4d targetToWorldColumnMajorTransform = computeWorldXformNoCache(usdStage, usdrtStage, targetPrim);

    usdrt::GfMatrix4d worldToTargetColumnMajorTransform = targetToWorldColumnMajorTransform.GetInverse();
    usdrt::GfMatrix4d sourceToTargetColumnMajorTransform =
        worldToTargetColumnMajorTransform * sourceToWorldColumnMajorTransform;

    return sourceToTargetColumnMajorTransform;
}

}
}
}
}
