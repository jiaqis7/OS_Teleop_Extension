.. _isaacsim_replicator_domain_randomization_OgnIntervalFiltering_1:

.. _isaacsim_replicator_domain_randomization_OgnIntervalFiltering:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Interval Filter
    :keywords: lang-en omnigraph node Replicator domain_randomization ogn-interval-filtering


Interval Filter
===============

.. <description>

Outputs indices if their frame count is a multiple of the interval

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.replicator.domain_randomization<ext_isaacsim_replicator_domain_randomization>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec In (*inputs:execIn*)", "``execution``", "exec", "None"
    "Frame Counts (*inputs:frameCounts*)", "``int[]``", "the current frame number for every environment", "[]"
    "Ignore Interval (*inputs:ignoreInterval*)", "``bool``", "if true, will just pass indices to downstream node", "False"
    "Indices (*inputs:indices*)", "``int[]``", "a list of indices to use in case of ignoring interval", "[]"
    "Interval (*inputs:interval*)", "``int``", "randomization will take place once every `interval` frames", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec Out (*outputs:execOut*)", "``execution``", "exec", "None"
    "Indices (*outputs:indices*)", "``int[]``", "the indices that are at the interval and need to be randomized", "None"
    "On Reset (*outputs:on_reset*)", "``bool``", "indicates whether an on_reset context triggered the execution", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.replicator.domain_randomization.OgnIntervalFiltering"
    "Version", "1"
    "Extension", "isaacsim.replicator.domain_randomization"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Interval Filter"
    "Categories", "Replicator"
    "__categoryDescriptions", "Replicator,Write Attribute"
    "Generated Class Name", "OgnIntervalFilteringDatabase"
    "Python Module", "isaacsim.replicator.domain_randomization"

