.. SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _analysis:

Analysis
========

This section describes post-hoc analyses that can be performed on solved models
to extract insights about the environmental and health impacts of food
consumption.

Marginal Damages
----------------

The marginal damages analysis computes the GHG emissions and health impacts
attributable to each unit of food consumed. This provides a consumption-centric
view of impacts, tracing emissions through trade and processing networks back
to production, and computing health effects based on dose-response curve
derivatives at current intake levels.

Concepts
~~~~~~~~

**GHG intensity** measures the greenhouse gas emissions per unit of food
consumed (kg CO₂e per kg food). Unlike production-based accounting, this
consumption-attributed metric traces emissions through the entire supply chain:
if wheat is grown in one country, milled into flour, and consumed in another,
the emissions from farming, processing, and transport are all attributed to the
final consumption.

**Health impact** measures the years of life lost (YLL) per unit of food
consumed. This is computed as the marginal effect—the derivative of the
dose-response curve at current population intake levels. Foods with protective
effects (fruits, vegetables, legumes) have negative values, while foods
associated with health risks (processed meat, excess red meat) have positive
values.

GHG Attribution Methodology
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GHG attribution uses a flow-based approach via sparse matrix algebra.
The network of production, processing, and trade links forms a directed graph
where each node (bus) receives material from upstream and passes it downstream.
Emissions occur at production links (e.g., fertilizer N₂O, enteric CH₄).

The key insight is that emission intensity propagates through the network:
the intensity at any bus equals its direct emissions plus the weighted average
of upstream intensities. This gives a linear system:

.. math::

   \rho = e + M \rho

where :math:`\rho` is the vector of emission intensities at each bus,
:math:`e` is the vector of direct emission contributions, and :math:`M` is
the weighted adjacency matrix (flow fractions). Solving
:math:`(I - M)\rho = e` yields the consumption-attributed intensity at each
food bus.

Health Attribution Methodology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Health impacts are computed by evaluating the slope of the piecewise-linear
dose-response curves at current intake levels. For each (health cluster, risk
factor) pair:

1. Current per-capita intake is computed from consumption flows and population
2. The slope of the log-relative-risk curve at this intake is determined
3. The chain rule converts this to YLL per unit intake change:

   .. math::

      \frac{d(\text{YLL})}{d(\text{intake})} =
      \frac{\text{YLL}_\text{base}}{\text{RR}_\text{ref}} \cdot \text{RR} \cdot
      \frac{d(\log \text{RR})}{d(\text{intake})}

4. Units are converted from YLL per g/capita/day to YLL per kg food

The result captures how marginal changes in consumption affect population
health outcomes, accounting for where each country currently sits on the
dose-response curve.

Sample Results
~~~~~~~~~~~~~~

The following figures show consumption-weighted global averages of GHG
intensity and health impacts by food group:

.. _fig-analysis-ghg:

.. figure:: _static/figures/analysis_marginal_ghg.png
   :alt: Bar chart showing GHG intensity by food group
   :align: center
   :width: 80%

   Global average GHG intensity by food group (consumption-weighted). Animal
   products (red meat, dairy) show the highest emissions per kg, while
   plant-based foods generally have lower intensities.

.. _fig-analysis-yll:

.. figure:: _static/figures/analysis_marginal_yll.png
   :alt: Bar chart showing health impact by food group
   :align: center
   :width: 80%

   Global average health impact by food group (consumption-weighted). Negative
   values indicate protective effects (fruits, vegetables, legumes, whole
   grains), while positive values indicate health risks. The magnitude reflects
   the marginal impact at current global intake levels.

Running the Analysis
~~~~~~~~~~~~~~~~~~~~

The marginal damages analysis runs automatically as part of the workflow when
plotting targets are requested. The relevant Snakemake targets are:

.. code-block:: bash

   # Extract marginal damages for a scenario
   tools/smk -j4 --configfile config/<name>.yaml -- \
       results/{name}/analysis/scen-default/marginal_damages.csv

   # Generate global average plots
   tools/smk -j4 --configfile config/<name>.yaml -- \
       results/{name}/plots/scen-default/marginal_ghg_global.pdf \
       results/{name}/plots/scen-default/marginal_yll_global.pdf

Output files:

``results/{name}/analysis/scen-{scenario}/marginal_damages.csv``
   Per-country, per-food-group marginal damages including:

   - ``consumption_mt``: Total consumption (million tonnes)
   - ``ghg_mtco2e_per_mt``: GHG intensity (MtCO₂e per Mt, equivalent to kg/kg)
   - ``yll_myll_per_mt``: Health impact (million YLL per Mt)
   - ``ghg_usd_per_t``: Monetized GHG damage (USD per tonne)
   - ``health_usd_per_t``: Monetized health damage (USD per tonne)

``results/{name}/plots/scen-{scenario}/marginal_damages_global.csv``
   Consumption-weighted global averages by food group.
