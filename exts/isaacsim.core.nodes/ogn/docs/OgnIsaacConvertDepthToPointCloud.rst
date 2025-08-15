.. _isaacsim_core_nodes_IsaacConvertDepthToPointCloud_1:

.. _isaacsim_core_nodes_IsaacConvertDepthToPointCloud:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Depth to Point Cloud
    :keywords: lang-en omnigraph node isaacCore nodes isaac-convert-depth-to-point-cloud


Isaac Depth to Point Cloud
==========================

.. <description>

Converts a 32FC1 image buffer into Point Cloud data

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Buffer Size (*inputs:bufferSize*)", "``uint``", "Size (in bytes) of the buffer (0 if the input is a texture)", "0"
    "Cuda Device Index (*inputs:cudaDeviceIndex*)", "``int``", "Index of the device where the data lives (-1 for host data)", "-1"
    "Data Ptr (*inputs:dataPtr*)", "``uint64``", "Pointer to the raw rgba array data", "0"
    "Exec In (*inputs:execIn*)", "``execution``", "The input execution port", "None"
    "Focal Length (*inputs:focalLength*)", "``float``", "focal length", "0.0"
    "Format (*inputs:format*)", "``uint64``", "Format", "33"
    "Height (*inputs:height*)", "``uint``", "Buffer array height, same as input", "0"
    "Horizontal Aperture (*inputs:horizontalAperture*)", "``float``", "horizontal aperture", "0.0"
    "Vertical Aperture (*inputs:verticalAperture*)", "``float``", "vertical aperture", "0.0"
    "Width (*inputs:width*)", "``uint``", "Buffer array width, same as input", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Buffer Size (*outputs:bufferSize*)", "``uint``", "Size (in bytes) of the buffer (0 if the input is a texture)", "None"
    "Cuda Device Index (*outputs:cudaDeviceIndex*)", "``int``", "Index of the device where the data lives (-1 for host data)", "-1"
    "Data Ptr (*outputs:dataPtr*)", "``uint64``", "Pointer to the rgb buffer data", "0"
    "Exec Out (*outputs:execOut*)", "``execution``", "Output execution triggers when lidar sensor has completed a full scan", "None"
    "Height (*outputs:height*)", "``uint``", "Height of point cloud buffer, will always return 1", "1"
    "Width (*outputs:width*)", "``uint``", "number of points in point cloud buffer", "0"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacConvertDepthToPointCloud"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacConvertDepthToPointCloud.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Depth to Point Cloud"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacConvertDepthToPointCloudDatabase"
    "Python Module", "isaacsim.core.nodes"

