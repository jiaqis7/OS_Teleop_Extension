.. _isaacsim_core_nodes_IsaacSetCameraOnRenderProduct_1:

.. _isaacsim_core_nodes_IsaacSetCameraOnRenderProduct:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Set Camera
    :keywords: lang-en omnigraph node isaacCore nodes isaac-set-camera-on-render-product


Isaac Set Camera
================

.. <description>

Isaac Sim node that sets the camera prim of an existing render product

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Camera Prim (*inputs:cameraPrim*)", "``target``", "Usd prim reference to the camera associated with this render product", "None"
    "Exec In (*inputs:execIn*)", "``execution``", "Input execution trigger", "None"
    "Render Product Path (*inputs:renderProductPath*)", "``token``", "Path of the render product", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec Out (*outputs:execOut*)", "``execution``", "Output execution trigger", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacSetCameraOnRenderProduct.svg"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Set Camera"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacSetCameraOnRenderProductDatabase"
    "Python Module", "isaacsim.core.nodes"

