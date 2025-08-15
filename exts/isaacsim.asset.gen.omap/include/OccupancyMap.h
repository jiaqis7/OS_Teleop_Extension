// Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include <carb/Defines.h>
#include <carb/Types.h>


namespace isaacsim
{

namespace asset
{
namespace gen
{
namespace omap
{
struct OccupancyMap
{


    CARB_PLUGIN_INTERFACE("isaacsim::asset::gen::omap::OccupancyMap", 0, 1);

    void(CARB_ABI* generateMap)();
    void(CARB_ABI* update)();
    void(CARB_ABI* setTransform)(carb::Float3 inputOrigin, carb::Float3 minPoint, carb::Float3 maxPoint);
    void(CARB_ABI* setCellSize)(float cellSize);
    std::vector<carb::Float3>(CARB_ABI* getOccupiedPositions)();
    std::vector<carb::Float3>(CARB_ABI* getFreePositions)();
    carb::Float3(CARB_ABI* getMinBound)();
    carb::Float3(CARB_ABI* getMaxBound)();
    carb::Int3(CARB_ABI* getDimensions)();
    std::vector<float>(CARB_ABI* getBuffer)();
    std::vector<char>(CARB_ABI* getColoredByteBuffer)(const carb::Int4& occupied,
                                                      const carb::Int4& unoccupied,
                                                      const carb::Int4& unknown);
};
}
}
}
}
