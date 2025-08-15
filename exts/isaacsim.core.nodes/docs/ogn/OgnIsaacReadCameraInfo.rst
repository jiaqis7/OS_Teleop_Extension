.. _isaacsim_core_nodes_IsaacReadCameraInfo_2:

.. _isaacsim_core_nodes_IsaacReadCameraInfo:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Read Camera Info
    :keywords: lang-en omnigraph node isaacCore nodes isaac-read-camera-info


Isaac Read Camera Info
======================

.. <description>

Isaac Sim node that reads camera info for a viewport

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Render Product Path (*inputs:renderProductPath*)", "``token``", "Path of the render product", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Camera Fisheye Params (*outputs:cameraFisheyeParams*)", "``float[]``", "Camera fisheye projection parameters", "None"
    "Focal Length (*outputs:focalLength*)", "``float``", "focal length", "None"
    "Height (*outputs:height*)", "``uint``", "Height for output image", "None"
    "Horizontal Aperture (*outputs:horizontalAperture*)", "``float``", "horizontal aperture", "None"
    "Horizontal Offset (*outputs:horizontalOffset*)", "``float``", "horizontal offset", "None"
    "Physical Distortion Coefficients (*outputs:physicalDistortionCoefficients*)", "``float[]``", "physical distortion model used for approximation, empty if not specified on camera prim", "None"
    "Physical Distortion Model (*outputs:physicalDistortionModel*)", "``token``", "physical distortion model used for approximation, empty if not specified on camera prim", "None"
    "Projection Type (*outputs:projectionType*)", "``token``", "projection type", "None"
    "Vertical Aperture (*outputs:verticalAperture*)", "``float``", "vertical aperture", "None"
    "Vertical Offset (*outputs:verticalOffset*)", "``float``", "vertical offset", "None"
    "Width (*outputs:width*)", "``uint``", "Width for output image", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacReadCameraInfo"
    "Version", "2"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacReadCameraInfo.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Read Camera Info"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacReadCameraInfoDatabase"
    "Python Module", "isaacsim.core.nodes"

