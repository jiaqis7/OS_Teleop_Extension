// Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once
#include <carb/Types.h>

#include <usdrt/gf/matrix.h>
#include <usdrt/gf/quat.h>
#include <usdrt/gf/vec.h>

#include <DynamicControl.h>
#include <PxActor.h>
namespace isaacsim
{
namespace core
{
namespace utils
{
namespace conversions
{
/**
 * @brief Convert a carb::Float3 into a pxr::GfVec3f
 *
 * @param v
 * @return pxr::GfVec3f
 */
inline pxr::GfVec3f asGfVec3f(const carb::Float3& v)
{
    return pxr::GfVec3f(v.x, v.y, v.z);
}
/**
 * @brief Convert a carb::Float4 into a pxr::GfQuatf
 *
 * @param q
 * @return pxr::GfQuatf
 */
inline pxr::GfQuatf asGfQuatf(const carb::Float4& q)
{
    return pxr::GfQuatf(q.w, q.x, q.y, q.z);
}

/**
 * @brief Convert a carb::Float3 into a pxr::GfVec3d
 *
 * @param v
 * @return pxr::GfVec3d
 */
inline pxr::GfVec3d asGfVec3d(const carb::Float3& v)
{
    return pxr::GfVec3d(v.x, v.y, v.z);
}
/**
 * @brief Convert a carb::Float4 into a pxr::GfQuatd
 *
 * @param q
 * @return pxr::GfQuatd
 */
inline pxr::GfQuatd asGfQuatd(const carb::Float4& q)
{
    return pxr::GfQuatd(q.w, q.x, q.y, q.z);
}

/**
 * @brief  Convert a carb::Float4 into a pxr::GfRotation
 *
 * @param q
 * @return pxr::GfRotation
 */
inline pxr::GfRotation asGfRotation(const carb::Float4& q)
{
    return pxr::GfRotation(asGfQuatd(q));
}

/**
 * @brief  Convert a DcTransform into a pxr::GfTransform
 *
 * @param pose
 * @return pxr::GfTransform
 */
inline pxr::GfTransform asGfTransform(const omni::isaac::dynamic_control::DcTransform& pose)
{
    pxr::GfTransform trans;
    trans.SetRotation(asGfRotation(pose.r));
    trans.SetTranslation(asGfVec3d(pose.p));
    return trans;
}

/**
 * @brief  Convert a carb::Float3 and a carb::Float4 into a pxr::GfTransform
 *
 * @param p
 * @param r
 * @return pxr::GfTransform
 */
inline pxr::GfTransform asGfTransform(const carb::Float3& p, const carb::Float4& r)
{
    pxr::GfTransform trans;
    trans.SetRotation(asGfRotation(r));
    trans.SetTranslation(asGfVec3d(p));
    return trans;
}

/**
 * @brief Convert a DcTransform to a GfMatrix4f
 *
 * @param input
 * @return pxr::GfMatrix4f
 */
inline pxr::GfMatrix4f asGfMatrix4f(const omni::isaac::dynamic_control::DcTransform& input)
{
    pxr::GfMatrix4f mat;
    mat.SetTranslateOnly(asGfVec3f(input.p));
    mat.SetRotateOnly(pxr::GfMatrix3f(asGfQuatf(input.r)));
    return mat;
}
/**
 * @brief Convert a DcTransform to a Transposed GfMatrix4f
 *
 * @param input
 * @return pxr::GfMatrix4f
 */
inline pxr::GfMatrix4f asGfMatrix4fT(const omni::isaac::dynamic_control::DcTransform& input)
{
    pxr::GfMatrix4f mat;
    mat.SetTranslateOnly(asGfVec3f(input.p));
    mat.SetRotateOnly(pxr::GfMatrix3f(asGfQuatf(input.r)));
    return mat.GetTranspose();
}

/**
 * @brief Convert a DcTransform to a GfMatrix4d
 *
 * @param input
 * @return pxr::GfMatrix4d
 */
inline pxr::GfMatrix4d asGfMatrix4d(const omni::isaac::dynamic_control::DcTransform& input)
{
    pxr::GfMatrix4d mat;
    mat.SetTranslateOnly(asGfVec3d(input.p));
    mat.SetRotateOnly(pxr::GfMatrix3d(asGfQuatd(input.r)));
    return mat;
}

/**
 * @brief convert pxr::GfVec3f to carb::Float3
 *
 * @param v
 * @return carb::Float3
 */
inline carb::Float3 asCarbFloat3(const pxr::GfVec3f& v)
{
    return carb::Float3{ v[0], v[1], v[2] };
}

/**
 * @brief convert pxr::GfVec3d to carb::Float3
 *
 * @param v
 * @return carb::Float3
 */
inline carb::Float3 asCarbFloat3(const pxr::GfVec3d& v)
{
    return carb::Float3{ static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]) };
}


/**
 * @brief convert pxr::GfVec4f to carb::Float4
 *
 * @param v
 * @return carb::Float4
 */
inline carb::Float4 asCarbFloat4(const pxr::GfQuatf& v)
{
    const pxr::GfVec3f& imag = v.GetImaginary();
    return carb::Float4{ imag[0], imag[1], imag[2], v.GetReal() };
}


/**
 * @brief convert pxr::GfVec4f to carb::Float4
 *
 * @param v
 * @return carb::Float4
 */
inline carb::Float4 asCarbFloat4(const pxr::GfQuatd& v)
{
    const pxr::GfVec3d& imag = v.GetImaginary();
    return carb::Float4{ static_cast<float>(imag[0]), static_cast<float>(imag[1]), static_cast<float>(imag[2]),
                         static_cast<float>(v.GetReal()) };
}

/**
 * @brief convert carb::Float3 into PxVec3
 *
 * @param v
 * @return ::physx::PxVec3
 */
inline ::physx::PxVec3 asPxVec3(const carb::Float3& v)
{
    return ::physx::PxVec3{ v.x, v.y, v.z };
}

/**
 * @brief convert pxr::GfVec3f into PxVec3
 *
 * @param v
 * @return ::physx::PxVec3
 */
inline ::physx::PxVec3 asPxVec3(const pxr::GfVec3f& v)
{
    return ::physx::PxVec3{ v[0], v[1], v[2] };
}

/**
 * @brief convert usdrt::GfVec3f into PxVec3
 *
 * @param v
 * @return ::physx::PxVec3
 */
inline ::physx::PxVec3 asPxVec3(const usdrt::GfVec3f& v)
{
    return ::physx::PxVec3{ v[0], v[1], v[2] };
}


/**
 * @brief convert pxr::GfVec3d into PxVec3
 *
 * @param v
 * @return ::physx::PxVec3
 */
inline ::physx::PxVec3 asPxVec3(const pxr::GfVec3d& v)
{
    return ::physx::PxVec3{ static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]) };
}

/**
 * @brief convert usdrt::GfVec3d into PxVec3
 *
 * @param v
 * @return ::physx::PxVec3
 */
inline ::physx::PxVec3 asPxVec3(const usdrt::GfVec3d& v)
{
    return ::physx::PxVec3{ static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]) };
}

/**
 * @brief Convert carb::Float4 into PxQuat
 *
 * @param q
 * @return ::physx::PxQuat
 */
inline ::physx::PxQuat asPxQuat(const carb::Float4& q)
{
    return ::physx::PxQuat{ q.x, q.y, q.z, q.w };
}

/**
 * @brief Convert pxr::GfQuatf into PxQuat
 *
 * @param q
 * @return ::physx::PxQuat
 */
inline ::physx::PxQuat asPxQuat(const pxr::GfQuatf& v)
{
    const pxr::GfVec3f& imag = v.GetImaginary();
    return ::physx::PxQuat{ imag[0], imag[1], imag[2], v.GetReal() };
}


/**
 * @brief Convert usdrt::GfQuatf into PxQuat
 *
 * @param q
 * @return ::physx::PxQuat
 */
inline ::physx::PxQuat asPxQuat(const usdrt::GfQuatf& v)
{
    const usdrt::GfVec3f& imag = v.GetImaginary();
    return ::physx::PxQuat{ imag[0], imag[1], imag[2], v.GetReal() };
}


/**
 * @brief Convert pxr::GfQuatd into PxQuat
 *
 * @param q
 * @return ::physx::PxQuat
 */
inline ::physx::PxQuat asPxQuat(const pxr::GfQuatd& v)
{
    const pxr::GfVec3d& imag = v.GetImaginary();
    return ::physx::PxQuat{ static_cast<float>(imag[0]), static_cast<float>(imag[1]), static_cast<float>(imag[2]),
                            static_cast<float>(v.GetReal()) };
}

/**
 * @brief Convert usdrt::GfQuatd into PxQuat
 *
 * @param q
 * @return ::physx::PxQuat
 */
inline ::physx::PxQuat asPxQuat(const usdrt::GfQuatd& v)
{
    const usdrt::GfVec3d& imag = v.GetImaginary();
    return ::physx::PxQuat{ static_cast<float>(imag[0]), static_cast<float>(imag[1]), static_cast<float>(imag[2]),
                            static_cast<float>(v.GetReal()) };
}


/**
 * @brief  Convert a DcTransform into a pxTransform
 *
 * @param pose
 * @return ::physx::PxTransform
 */
inline ::physx::PxTransform asPxTransform(const omni::isaac::dynamic_control::DcTransform& pose)
{
    return ::physx::PxTransform{ asPxVec3(pose.p), asPxQuat(pose.r) };
}

/**
 * @brief Convert GfTransform to PxTransform
 *
 * @param trans
 * @return ::physx::PxTransform
 */
inline ::physx::PxTransform asPxTransform(const pxr::GfTransform& trans)
{
    ::physx::PxTransform p;
    const pxr::GfVec3d& pos = trans.GetTranslation();
    const pxr::GfQuatd& rot = trans.GetRotation().GetQuat();

    p.p.x = static_cast<float>(pos[0]);
    p.p.y = static_cast<float>(pos[1]);
    p.p.z = static_cast<float>(pos[2]);
    p.q.x = static_cast<float>(rot.GetImaginary()[0]);
    p.q.y = static_cast<float>(rot.GetImaginary()[1]);
    p.q.z = static_cast<float>(rot.GetImaginary()[2]);
    p.q.w = static_cast<float>(rot.GetReal());
    return p;
}

/**
 * @brief Convert usdrt::GfMatrix4d to PxTransform
 *
 * @param trans
 * @return ::physx::PxTransform
 */
inline ::physx::PxTransform asPxTransform(const usdrt::GfMatrix4d& mat)
{
    ::physx::PxTransform p;
    const pxr::GfMatrix4d* gfMat = reinterpret_cast<const pxr::GfMatrix4d*>(&mat);
    pxr::GfTransform trans(*gfMat);
    const pxr::GfVec3d& pos = trans.GetTranslation();
    const pxr::GfQuatd& rot = trans.GetRotation().GetQuat();

    p.p.x = static_cast<float>(pos[0]);
    p.p.y = static_cast<float>(pos[1]);
    p.p.z = static_cast<float>(pos[2]);
    p.q.x = static_cast<float>(rot.GetImaginary()[0]);
    p.q.y = static_cast<float>(rot.GetImaginary()[1]);
    p.q.z = static_cast<float>(rot.GetImaginary()[2]);
    p.q.w = static_cast<float>(rot.GetReal());
    return p;
}

/**
 * @brief Converts a usdrt Gf translation and orientation to
 *
 * @param translation
 * @param orientation
 * @return ::physx::PxTransform
 */
inline ::physx::PxTransform asPxTransform(const usdrt::GfVec3d& translation, const usdrt::GfQuatd& orientation)
{
    ::physx::PxTransform p;
    p.p.x = static_cast<float>(translation[0]);
    p.p.y = static_cast<float>(translation[1]);
    p.p.z = static_cast<float>(translation[2]);
    p.q.x = static_cast<float>(orientation.GetImaginary()[0]);
    p.q.y = static_cast<float>(orientation.GetImaginary()[1]);
    p.q.z = static_cast<float>(orientation.GetImaginary()[2]);
    p.q.w = static_cast<float>(orientation.GetReal());
    return p;
}

/**
 * @brief Create a DcTransform from a GfVec3f and a GfQuatf
 *
 * @param p
 * @param q
 * @return omni::isaac::dynamic_control::DcTransform
 */
inline omni::isaac::dynamic_control::DcTransform asDcTransform(const pxr::GfVec3f& p, const pxr::GfQuatf& q)
{
    omni::isaac::dynamic_control::DcTransform t;
    t.p = asCarbFloat3(p);
    t.r = asCarbFloat4(q);
    return t;
}

/**
 * @brief Create a DcTransform from a GfVec3d and a GfQuatd
 *
 * @param p
 * @param q
 * @return omni::isaac::dynamic_control::DcTransform
 */
inline omni::isaac::dynamic_control::DcTransform asDcTransform(const pxr::GfVec3d& p, const pxr::GfQuatd& q)
{
    omni::isaac::dynamic_control::DcTransform t;
    t.p = asCarbFloat3(p);
    t.r = asCarbFloat4(q);
    return t;
}
}
}
}
}
