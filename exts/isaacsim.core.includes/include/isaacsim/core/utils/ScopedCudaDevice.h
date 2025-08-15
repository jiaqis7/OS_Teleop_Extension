// Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once
#include <carb/cuda/CudaRuntime.h>
#include <carb/logging/Log.h> // CudaRintime.h does not have CARB_LOG_ERROR

#include <cuda.h>

namespace isaacsim
{
namespace core
{
namespace utils
{
class ScopedDevice
{
public:
    ScopedDevice(const int device = -1) : mDevice(device)
    {

        // if we want cpu or can't get a cuda device, then do nothing.
        if (device == -1 || cudaGetDevice(&mOldDevice) != cudaError::cudaSuccess)
        {
            mOldDevice = mDevice = -1;
            return;
        }

        // if we want a device, and its not the current threads host device, then set it.
        if (mDevice != mOldDevice)
        {
            CUDA_CHECK(cudaSetDevice(mDevice));
            // NOTE: what do you want to do with an error here?  set mOldDevice to mDevice so you do nothing in dtor?
        }
    }

    ~ScopedDevice()
    {
        // return device back to what we had before if we set earlier
        if (mDevice != mOldDevice)
        {
            CUDA_CHECK(cudaSetDevice(mOldDevice));
        }
    }

private:
    int mDevice = 0; // The device we want to use
    int mOldDevice = 0; // current threads host device
};


class ScopedCudaTextureObject final
{
    cudaTextureObject_t _texObj = 0;

public:
    ScopedCudaTextureObject(cudaMipmappedArray_t mmarr, int mipLevel = 0)
    {
        if (!mmarr)
            return;
        cudaArray_t levelArray = 0;
        CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, mmarr, mipLevel));
        if (!levelArray)
            return;
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = levelArray;
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;
        CUDA_CHECK(cudaCreateTextureObject(&_texObj, &resDesc, &texDesc, nullptr));
    }

    ~ScopedCudaTextureObject()
    {
        if (_texObj)
        {
            CUDA_CHECK(cudaDestroyTextureObject(_texObj));
        }
    }

    operator cudaTextureObject_t&()
    {
        return _texObj;
    }
};

}
}
}
