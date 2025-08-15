// Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include <DynamicControlTypes.h>
#include <cmath>

namespace isaacsim
{
namespace core
{
namespace utils
{
namespace math
{
/**
 * @brief Cross product between carb::Float3
 *
 *  @param[in] carb::Float3.
 *  @param[in] carb::Float3.
 *  @return carb::Float3
 */
inline carb::Float3 cross(const carb::Float3& b, const carb::Float3& c)
{
    return carb::Float3{ b.y * c.z - b.z * c.y, b.z * c.x - b.x * c.z, b.x * c.y - b.y * c.x };
}

/**
 * @brief Dot product between carb::Float3
 *
 *  @param[in] carb::Float3.
 *  @param[in] carb::Float3.
 *  @return float
 */
inline float dot(const carb::Float3& v1, const carb::Float3& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

/**
 * @brief Dot product between carb::Float4
 *
 *  @param[in] carb::Float4.
 *  @param[in] carb::Float4.
 *  @return float
 */
inline float dot(const carb::Float4& v1, const carb::Float4& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}


/**
 * @brief Inverse of float4 as a quaternion
 *
 *  @param[in] carb::Float4.
 *  @return carb::Float4.
 */
inline carb::Float4 inverse(const carb::Float4& q)
{
    return carb::Float4{ -q.x, -q.y, -q.z, q.w };
}

/**
 * @brief Multiply carb::Float3 with a float
 *
 *  @param[in] carb::Float3.
 *  @param[in] float.
 *  @return carb::Float3
 */
inline carb::Float3 operator*(const carb::Float3& a, const float x)
{
    return carb::Float3{ a.x * x, a.y * x, a.z * x };
}

/**
 * @brief Multiply carb::Float4 with a float
 *
 *  @param[in] carb::Float4.
 *  @param[in] float.
 *  @return carb::Float4
 */
inline carb::Float4 operator*(const carb::Float4& a, const float x)
{
    return carb::Float4{ a.x * x, a.y * x, a.z * x, a.w * x };
}

/**
 * @brief Quaternion Multiplication between carb::Float4 and carb::Float4
 *
 *  @param[in] carb::Float4.
 *  @param[in] carb::Float4.
 *  @return carb::Float4
 */
inline carb::Float4 operator*(const carb::Float4& a, const carb::Float4& b)
{
    return carb::Float4{ a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z, a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
                         a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}

/**
 * @brief Add two carb::Float3
 *
 *  @param[in] carb::Float3.
 *  @param[in] carb::Float3.
 *  @return carb::Float3
 */
inline carb::Float3 operator+(const carb::Float3& a, const carb::Float3& b)
{
    return carb::Float3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

/**
 * @brief Subtract two carb::Float3
 *
 *  @param[in] carb::Float3.
 *  @param[in] carb::Float3.
 *  @return carb::Float3
 */
inline carb::Float3 operator-(const carb::Float3& a, const carb::Float3& b)
{
    return carb::Float3{ a.x - b.x, a.y - b.y, a.z - b.z };
}


/**
 * @brief Add two carb::Float4
 *
 *  @param[in] carb::Float4.
 *  @param[in] carb::Float4.
 *  @return carb::Float4
 */
inline carb::Float4 operator+(const carb::Float4& a, const carb::Float4& b)
{
    return carb::Float4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}


/**
 * @brief Subtract two carb::Float4
 *
 *  @param[in] carb::Float4.
 *  @param[in] carb::Float4.
 *  @return carb::Float4
 */
inline carb::Float4 operator-(const carb::Float4& a, const carb::Float4& b)
{
    return carb::Float4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}


/**
 * @brief Normalize carb::Float4
 *
 *  @param[in] carb::Float4.
 *  @return carb::Float4.
 */
inline carb::Float4 normalize(const carb::Float4& q)
{
    float lSq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (lSq > 0.0f)
    {
        float invL = 1.0f / std::sqrt(lSq);

        return q * invL;
    }
    else
    {
        return carb::Float4{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
}

/**
 * @brief Normalize carb::Float3
 *
 *  @param[in] carb::Float3.
 *  @return carb::Float3.
 */
inline carb::Float3 normalize(const carb::Float3& q)
{
    float lSq = q.x * q.x + q.y * q.y + q.z * q.z;
    if (lSq > 0.0f)
    {
        float invL = 1.0f / sqrtf(lSq);

        return q * invL;
    }
    else
    {
        return carb::Float3{ 0.0f, 0.0f, 0.0f };
    }
}

/**
 * @brief Rotate a vector carb::Float3 by quaternion carb::Float4
 *
 *  @param[in] carb::Float4.
 *  @param[in] carb::Float3.
 *  @return carb::Float3.
 */
inline carb::Float3 rotate(const carb::Float4& q, const carb::Float3 x)
{
    const carb::Float3 v_q = carb::Float3{ q.x, q.y, q.z };
    return x * (2.0f * q.w * q.w - 1.0f) + cross(v_q, x) * q.w * 2.0f + v_q * dot(v_q, x) * 2.0f;
}


/**
 * @brief Multiple two DcTransform objects
 *
 *  @param[in] DcTransform.
 *  @param[in] DcTransform.
 *  @return DcTransform.
 */
inline omni::isaac::dynamic_control::DcTransform operator*(const omni::isaac::dynamic_control::DcTransform& self,
                                                           const omni::isaac::dynamic_control::DcTransform& other)
{
    return omni::isaac::dynamic_control::DcTransform{ rotate(self.r, other.p) + self.p, normalize(self.r * other.r) };
}

/**
 * @brief Get Inverse of DcTransform
 *
 *  @param[in] DcTransform.
 *  @return DcTransform.
 */
inline omni::isaac::dynamic_control::DcTransform inverse(const omni::isaac::dynamic_control::DcTransform& transform)
{
    omni::isaac::dynamic_control::DcTransform t;
    t.r = inverse(transform.r);
    t.p = rotate(t.r, transform.p) * -1.0f;
    return t;
}

/**
 * @brief computes local pose of b in a space
 *
 *  @param[in] pxr::DcTransform.
 *  @param[in] pxr::DcTransform.
 *  @return pxr::DcTransform.
 */
inline omni::isaac::dynamic_control::DcTransform transformInv(const omni::isaac::dynamic_control::DcTransform& a,
                                                              const omni::isaac::dynamic_control::DcTransform& b)
{
    carb::Float4 qconj;
    qconj.w = a.r.w;
    qconj.x = -a.r.x;
    qconj.y = -a.r.y;
    qconj.z = -a.r.z;
    float invqnorm = 1.0f / (qconj.w * qconj.w + qconj.x * qconj.x + qconj.y * qconj.y + qconj.z * qconj.z);

    carb::Float4 qinv;
    qinv.w = qconj.w * invqnorm;
    qinv.x = -qconj.x * invqnorm;
    qinv.y = -qconj.y * invqnorm;
    qinv.z = -qconj.z * invqnorm;

    carb::Float4 qv = a.r;
    qv.w = 0;
    qv.x = b.p.x - a.p.x;
    qv.y = b.p.y - a.p.y;
    qv.z = b.p.z - a.p.z;

    carb::Float4 result = (qconj * (qv * qinv));
    carb::Float4 res_quat = (b.r * qconj);
    omni::isaac::dynamic_control::DcTransform res;
    res.p.x = result.x;
    res.p.y = result.y;
    res.p.z = result.z;
    res.r = res_quat;
    return res;
}

/**
 * @brief computes local pose of b in a space
 *
 *  @param[in] pxr::GfTransform.
 *  @param[in] pxr::GfTransform.
 *  @return pxr::GfTransform.
 */
inline pxr::GfTransform transformInv(const pxr::GfTransform& a, const pxr::GfTransform& b)
{
    pxr::GfRotation qinv = a.GetRotation().GetQuat().GetConjugate();
    pxr::GfTransform trans;
    trans.SetRotation(b.GetRotation() * qinv);
    trans.SetTranslation(qinv.TransformDir(b.GetTranslation() - a.GetTranslation()));
    return trans;
}

/**
 * @brief Rotate x axis basis vector by quaternion carb::Float4
 *
 *  @param[in] carb::Float4.
 *  @return carb::Float3.
 */
inline carb::Float3 getBasisVectorX(const carb::Float4& q)
{
    return rotate(q, carb::Float3{ 1.0f, 0.0f, 0.0f });
}

/**
 * @brief Rotate y axis basis vector by quaternion carb::Float4
 *
 *  @param[in] carb::Float4.
 *  @return carb::Float3.
 */
inline carb::Float3 getBasisVectorY(const carb::Float4& q)
{
    return rotate(q, carb::Float3{ 0.0f, 1.0f, 0.0f });
}

/**
 * @brief Rotate z axis basis vector by quaternion carb::Float4
 *
 *  @param[in] carb::Float4.
 *  @return carb::Float3.
 */
inline carb::Float3 getBasisVectorZ(const carb::Float4& q)
{
    return rotate(q, carb::Float3{ 0.0f, 0.0f, 1.0f });
}


/**
 * @brief Linear interpolation between two vectors
 *
 *  @param[in] carb::Float3.
 *  @param[in] carb::Float3.
 *  @param[in] float.
 *  @return carb::Float3.
 */
inline carb::Float3 lerp(const carb::Float3& start, const carb::Float3& end, const float t)
{
    return start + ((end - start) * t);
}

/**
 * @brief Linear interpolation between two vectors
 *
 *  @param[in] carb::Float4.
 *  @param[in] carb::Float4.
 *  @param[in] float.
 *  @return carb::Float4.
 */
inline carb::Float4 lerp(const carb::Float4& start, const carb::Float4& end, const float t)
{
    return normalize(start + ((end - start) * t));
}

/**
 * @brief Spherical Linear interpolation between two vectors
 *
 *  @param[in] carb::Float3.
 *  @param[in] carb::Float3.
 *  @param[in] float.
 *  @return carb::Float3.
 */
inline carb::Float4 slerp(const carb::Float4& start, const carb::Float4& end, const float t)
{
    float dot_value = dot(start, end);
    carb::Float4 s{ start.x, start.y, start.z, start.w };
    if (dot_value < 0)
    {
        s = start * -1.0f;
        dot_value = -dot_value;
    }

    if (dot_value > 0.9995)
    {
        return lerp(s, end, t);
    }

    float theta_0 = acos(dot_value);
    float s_t_0 = sin(theta_0);

    float theta = theta_0 * t;
    float s_t = sin(theta);

    float s1 = s_t / s_t_0;
    float s0 = cos(theta) - dot_value * s1;

    return (s * s0) + (end * s1);
}

/**
 * @brief Linear interpolation between two Transforms
 *
 *  @param[in] omni::isaac::dynamic_control::DcTransform.
 *  @param[in] omni::isaac::dynamic_control::DcTransform.
 *  @param[in] float.
 *  @return omni::isaac::dynamic_control::DcTransform.
 */
inline omni::isaac::dynamic_control::DcTransform lerp(const omni::isaac::dynamic_control::DcTransform& a,
                                                      const omni::isaac::dynamic_control::DcTransform& b,
                                                      const float t)
{
    return omni::isaac::dynamic_control::DcTransform{ lerp(a.p, b.p, t), lerp(a.r, b.r, t) };
}

/**
 * @brief Spherical Linear interpolation between two Transforms
 *
 *  @param[in] omni::isaac::dynamic_control::DcTransform.
 *  @param[in] omni::isaac::dynamic_control::DcTransform.
 *  @param[in] float.
 *  @return omni::isaac::dynamic_control::DcTransform.
 */
inline omni::isaac::dynamic_control::DcTransform slerp(const omni::isaac::dynamic_control::DcTransform& a,
                                                       const omni::isaac::dynamic_control::DcTransform& b,
                                                       const float t)
{
    return omni::isaac::dynamic_control::DcTransform{ lerp(a.p, b.p, t), slerp(a.r, b.r, t) };
}

/**
 * @brief Compute look at rotation based on camera position, target location and up vector
 *
 * @param camera
 * @param target
 * @param up
 * @return pxr::GfQuatf
 */
inline pxr::GfQuatf lookAt(const pxr::GfVec3f& camera, const pxr::GfVec3f& target, const pxr::GfVec3f& up)
{
    pxr::GfVec3f F = (target - camera).GetNormalized();
    pxr::GfVec3f R = pxr::GfCross(F, up).GetNormalized();
    pxr::GfVec3f U = pxr::GfCross(R, F);

    float trace = R[0] + U[1] + F[2];
    if (trace > 0.0f)
    {
        float s = 0.5f / sqrtf(trace + 1.0f);
        return pxr::GfQuatf(0.25f / s, pxr::GfVec3f((U[2] - F[1]) * s, (F[0] - R[2]) * s, (R[1] - U[0]) * s));
    }
    else
    {
        if (R[0] > U[1] and R[0] > F[2])
        {
            float s = 2.0f * sqrtf(1.0f + R[0] - U[1] - F[2]);
            return pxr::GfQuatf((U[2] - F[1]) / s, pxr::GfVec3f(0.25f * s, (U[0] + R[1]) / s, (F[0] + R[2]) / s));
        }
        else if (U[1] > F[2])
        {
            float s = 2.0f * sqrtf(1.0f + U[1] - R[0] - F[2]);
            return pxr::GfQuatf((F[0] - R[2]) / s, pxr::GfVec3f((U[0] + R[1]) / s, 0.25f * s, (F[1] + U[2]) / s));
        }
        else
        {
            float s = 2.0f * sqrtf(1.0f + F[2] - R[0] - U[1]);
            return pxr::GfQuatf((R[1] - U[0]) / s, pxr::GfVec3f((F[0] + R[2]) / s, (F[1] + U[2]) / s, 0.25f * s));
        }
    }
}

/**
 * @brief Rounds to nearest Nth decimal place
 *
 * @param input decimal number input
 * @param place Nth place to round to, ex: 10000.0 for rounding to nearest ten-thousandths. Must be a positive value.
 * @return double
 */
inline double roundNearest(double input, double place)
{
    if (place > 0.0)
    {
        return floor(input * place + 0.5) / place;
    }

    return input;
}

}
}
}
}
