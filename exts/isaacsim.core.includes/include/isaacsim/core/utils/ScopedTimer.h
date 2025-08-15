// Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace isaacsim
{
namespace core
{
namespace utils
{
class ScopedTimer
{
public:
    ScopedTimer(const std::string& message)
    {
        mMessage = message;
        mStart = std::chrono::steady_clock::now();
    }

    ~ScopedTimer()
    {
        mStop = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> diff = mStop - mStart;
        std::cout << mMessage << " : " << diff.count() << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> mStart;
    std::chrono::time_point<std::chrono::steady_clock> mStop;
    std::string mMessage;
};

}
}
}
