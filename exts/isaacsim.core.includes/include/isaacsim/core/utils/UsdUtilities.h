// Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include <carb/InterfaceUtils.h>
#include <carb/logging/Log.h>

#include <omni/usd/UsdContext.h>
#include <omni/usd/UsdUtils.h>

#include <chrono>
#include <iostream>
#include <string>

#include <omni/usd-abi/IRenderProductPrim.h>
namespace isaacsim
{
namespace core
{
namespace utils
{

static const PXR_NS::TfToken kIsaacNameOveride("isaac:nameOverride");

inline pxr::UsdPrim getCameraPrimFromRenderProduct(const std::string& renderProductPathString)
{
    pxr::UsdStagePtr stage = omni::usd::UsdContext::getContext()->getStage();
    omni::usd::IPathAbi* iPath = carb::getCachedInterface<omni::usd::IPathAbi>();
    omni::usd::IRenderProductPrimFactory* iPrimFactory = carb::getCachedInterface<omni::usd::IRenderProductPrimFactory>();
    if (!stage || !iPath || !iPrimFactory)
        return pxr::UsdPrim();
    omni::usd::PathH renderProductPathH = iPath->getHandle(renderProductPathString.c_str());
    omni::usd::IRenderProductPrimPtr rp =
        iPrimFactory->createPrimFromStage(omni::usd::UsdContext::getContext()->getStageId(), renderProductPathH);
    if (!rp)
        return pxr::UsdPrim();
    return stage->GetPrimAtPath(pxr::SdfPath(iPath->getText(rp->getCameraPath())));
}

inline pxr::UsdAttribute getCameraAttributeFromRenderProduct(const std::string& attributeString,
                                                             const std::string& renderProductPathString)
{
    pxr::UsdPrim cameraPrim = getCameraPrimFromRenderProduct(renderProductPathString);
    if (!cameraPrim.IsValid())
        return pxr::UsdAttribute();
    return cameraPrim.GetAttribute(pxr::TfToken(attributeString.c_str()));
}

template <class T>
void safeGetAttribute(const pxr::UsdAttribute& attr, T& inputValue)
{
    // if (attr.IsValid())
    // {
    if (attr.HasValue())
    {
        attr.Get(&inputValue);
    }
    else
    {
        CARB_LOG_WARN("USD attribute %s does not exist, using default", attr.GetName().GetString().c_str());
    }
    // }
    // else
    // {
    //     CARB_LOG_ERROR_ONCE(
    //         "USD attribute is INVALID %s, you will only be warned once, so you probably want to fix whatever called "
    //         "isaacsim::core::utils::safeGetAttribute", attr.GetName().GetString().c_str());
    // }
}

inline std::string GetName(const pxr::UsdPrim& prim)
{
    std::string primName = prim.GetName().GetString();
    if (prim.HasAttribute(kIsaacNameOveride))
    {
        safeGetAttribute<std::string>(prim.GetAttribute(kIsaacNameOveride), primName);
    }
    return primName;
}


inline pxr::SdfPath getSdfPathFromUint64(uint64_t path_token)
{
#if defined(_WIN32)
#    pragma warning(push)
#    pragma warning(disable : 4996)
    return reinterpret_cast<const pxr::SdfPath&>(path_token);
#    pragma warning(pop)
#else
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wstrict-aliasing"
    return reinterpret_cast<const pxr::SdfPath&>(path_token);
#    pragma GCC diagnostic pop
#endif
}

}
}
}
