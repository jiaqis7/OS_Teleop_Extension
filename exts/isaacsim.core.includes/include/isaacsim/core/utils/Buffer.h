// Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#pragma once

#include "ScopedCudaDevice.h"

#include <carb/cuda/CudaRuntime.h>
#include <carb/cudainterop/CudaInterop.h>

#include <cuda.h>
#include <iostream>
#include <vector>

namespace isaacsim
{
namespace core
{
namespace utils
{
enum class eMemoryType
{
    Host = 0,
    Device = 1,
};


template <typename T>
class Buffer
{
public:
    virtual ~Buffer()
    {
    }
    virtual void resize(size_t size) = 0;
    virtual T* data() const = 0;
    virtual size_t size() const = 0;
    size_t sizeofType() const
    {
        return sizeof(T);
    }
    size_t sizeInBytes() const
    {
        return size() * sizeofType();
    }

    eMemoryType type() const
    {
        return mMemoryType;
    }

protected:
    eMemoryType mMemoryType;
};

template <typename T>
class DeviceBufferBase : public Buffer<T>
{
    using Buffer<T>::mMemoryType;

public:
    DeviceBufferBase(const size_t& size = 0, const int device = -1)
    {
        mMemoryType = eMemoryType::Device;
        mDevice = device;
        resize(size);
    }
    virtual ~DeviceBufferBase()
    {
        ScopedDevice scopedDevice(mDevice);

        CUDA_CHECK(cudaFree(mBuffer));
        mBuffer = nullptr;
    }
    virtual void setDevice(const int device = -1)
    {
        if (device != mDevice)
        {
            // if the device doesn't match and we had a buffer allocated, release it on the old device and switch
            if (mBuffer)
            {
                ScopedDevice scopedDevice(mDevice);
                CUDA_CHECK(cudaFree(mBuffer));
                mBuffer = nullptr;
            }
            mDevice = device;
            resize(mSize);
        }
    }
    virtual void resize(size_t size)
    {
        if ((size != mSize && size > 0) || mBuffer == nullptr)
        {
            ScopedDevice scopedDevice(mDevice);
            if (mBuffer)
            {
                CUDA_CHECK(cudaFree(mBuffer));
                mBuffer = nullptr;
            }
            if (size > 0)
            {
                CUDA_CHECK(cudaMalloc(&mBuffer, size * sizeof(T)));
            }
            mSize = size;
        }
    }
    virtual T* data() const
    {
        return mBuffer;
    }
    virtual size_t size() const
    {
        return mSize;
    }
    virtual void copy(const void* src, size_t size, enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
    {
        ScopedDevice scopedDevice(mDevice);
        CUDA_CHECK(cudaMemcpy(mBuffer, src, size * sizeof(T), kind));
    }
    virtual void copyAsync(const void* src, size_t size, enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
    {
        ScopedDevice scopedDevice(mDevice);
        CUDA_CHECK(cudaMemcpyAsync(mBuffer, src, size * sizeof(T), kind));
    }
    void debugPrint(const std::string& start, const std::string& end)
    {
        ScopedDevice scopedDevice(mDevice);
        printf("%s", start.c_str());
        std::vector<T> hostBuffer(mSize);
        CUDA_CHECK(cudaMemcpyAsync(hostBuffer.data(), mBuffer, mSize * sizeof(T), cudaMemcpyDeviceToHost));
        for (size_t i; i < mSize; ++i)
        {
            std::cout << hostBuffer[i];
            if (i != mSize - 1)
                std::cout << ", ";
        }
        printf("%s", end.c_str());
    }

private:
    T* mBuffer = nullptr;
    size_t mSize = 0;
    int mDevice = 0;
};
template <typename T>
class HostBufferBase : public Buffer<T>
{
    using Buffer<T>::mMemoryType;

public:
    HostBufferBase(size_t size = 0)
    {
        mMemoryType = eMemoryType::Host;
        resize(size);
    }
    virtual void resize(size_t size)
    {
        mBuffer.resize(size);
    }
    virtual void resize(size_t size, const T& val)
    {
        mBuffer.resize(size, val);
    }
    virtual T* data() const
    {
        return (T*)mBuffer.data();
    }
    virtual size_t size() const
    {
        return mBuffer.size();
    }

    std::vector<T> mBuffer;
};

typedef DeviceBufferBase<uint8_t> DeviceBuffer;
typedef HostBufferBase<uint8_t> HostBuffer;
}
}
}
