.. _isaacsim_core_nodes_IsaacReadTimes_1:

.. _isaacsim_core_nodes_IsaacReadTimes:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Read Times
    :keywords: lang-en omnigraph node isaacCore nodes isaac-read-times


Isaac Read Times
================

.. <description>

Read time related value from Fabric.

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec In (*inputs:execIn*)", "``execution``", "The input execution port", "None"
    "Gpu (*inputs:gpu*)", "``uint64``", "Pointer to shared context containing gpu foundations valid only in the postRender graph.", "0"
    "Render Results (*inputs:renderResults*)", "``uint64``", "Render results", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Duration Denominator (*outputs:durationDenominator*)", "``uint64``", "durationDenominator", "None"
    "Duration Numerator (*outputs:durationNumerator*)", "``int64``", "durationNumerator", "None"
    "Exec Out (*outputs:execOut*)", "``execution``", "The output execution port", "None"
    "External Time Of Sim Ns (*outputs:externalTimeOfSimNs*)", "``int64``", "externalTimeOfSimNs", "None"
    "Frame Number (*outputs:frameNumber*)", "``int64``", "frameNumber", "None"
    "Rational Time Of Sim Denominator (*outputs:rationalTimeOfSimDenominator*)", "``uint64``", "rationalTimeOfSimDenominator", "None"
    "Rational Time Of Sim Numerator (*outputs:rationalTimeOfSimNumerator*)", "``int64``", "rationalTimeOfSimNumerator", "None"
    "Sample Time Offset In Sim Frames (*outputs:sampleTimeOffsetInSimFrames*)", "``uint64``", "sampleTimeOffsetInSimFrames", "None"
    "Simulation Time (*outputs:simulationTime*)", "``double``", "Current Simulation Time in Seconds", "None"
    "Monotonic Simulation Time (*outputs:simulationTimeMonotonic*)", "``double``", "Current Monotonic Simulation Time in Seconds", "None"
    "swhFrameNumber (*outputs:swhFrameNumber*)", "``int64``", "swhFrameNumber", "None"
    "System Time (*outputs:systemTime*)", "``double``", "Current System Time in Seconds", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacReadTimes"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacReadTimes.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "__tokens", "[""IsaacFabricTimes""]"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacReadTimesDatabase"
    "Python Module", "isaacsim.core.nodes"

