.. _isaacsim_core_nodes_IsaacPassthroughImagePtr_1:

.. _isaacsim_core_nodes_IsaacPassthroughImagePtr:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Passthrough Image Pointer
    :keywords: lang-en omnigraph node isaacCore nodes isaac-passthrough-image-ptr


Isaac Passthrough Image Pointer
===============================

.. <description>

Isaac Sim Node passes a pointer through without changing it

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
    "Data Ptr (*inputs:dataPtr*)", "``uint64``", "pointer to pass throuh", "0"
    "Exec In (*inputs:execIn*)", "``execution``", "The input execution port", "None"
    "Format (*inputs:format*)", "``uint64``", "Format", "0"
    "Height (*inputs:height*)", "``uint``", "Buffer array height", "0"
    "Width (*inputs:width*)", "``uint``", "Buffer array width", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Buffer Size (*outputs:bufferSize*)", "``uint``", "Size (in bytes) of the buffer (0 if the input is a texture)", "None"
    "Cuda Device Index (*outputs:cudaDeviceIndex*)", "``int``", "Index of the device where the data lives (-1 for host data)", "-1"
    "Data Ptr (*outputs:dataPtr*)", "``uint64``", "pointer to pass throuh", "None"
    "Exec Out (*outputs:execOut*)", "``execution``", "The output execution port", "None"
    "Format (*outputs:format*)", "``uint64``", "Format", "None"
    "Height (*outputs:height*)", "``uint``", "Buffer array height", "None"
    "Width (*outputs:width*)", "``uint``", "Buffer array width", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacPassthroughImagePtr"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacPassthroughImagePtr.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Passthrough Image Pointer"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacPassthroughImagePtrDatabase"
    "Python Module", "isaacsim.core.nodes"

