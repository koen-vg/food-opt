.. SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _analysis:

Analysis
========

This section describes post-hoc analyses that can be performed on solved models
to extract insights about production, consumption, and the environmental and
health impacts of food systems.

.. _statistics-extraction:

Statistics Extraction
---------------------

The statistics extraction produces standardized CSV files summarizing key model
outputs. These files provide a consistent interface for downstream analysis and
visualization, extracting data from the solved PyPSA network using actual
dispatch flows rather than capacity-based estimates.

Running the Extraction
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Extract all statistics for a scenario
   tools/smk -j4 --configfile config/<name>.yaml -- \
       results/{name}/analysis/scen-default/crop_production.csv

   # Or request any downstream plot to trigger extraction automatically

Output Files
~~~~~~~~~~~~

All statistics are written to ``results/{name}/analysis/scen-{scenario}/``.

**crop_production.csv** — Crop production by crop, region, and country

.. csv-table::
   :header: Column, Type, Unit, Description

   ``crop``, string, —, "Crop identifier (e.g., ``wheat``, ``maize``, ``grassland``)"
   ``region``, string, —, "Production region identifier"
   ``country``, string, —, "ISO 3166-1 alpha-3 country code"
   ``production_mt``, float, Mt, "Production quantity in megatonnes"

Sources include single-crop production links, grassland production, and
multicropping links (where multiple crops share the same land).

**land_use.csv** — Land allocation by crop, region, resource class, and water supply

.. csv-table::
   :header: Column, Type, Unit, Description

   ``crop``, string, —, "Crop identifier"
   ``region``, string, —, "Production region identifier"
   ``resource_class``, string, —, "Land suitability class (e.g., ``VS``, ``S``, ``MS``)"
   ``water_supply``, string, —, "Water regime (``rainfed`` or ``irrigated``)"
   ``country``, string, —, "ISO 3166-1 alpha-3 country code"
   ``area_mha``, float, Mha, "Cultivated area in million hectares"

For multicropping systems, land area is attributed to individual crops
proportionally by their yield (efficiency) on that land.

**animal_production.csv** — Livestock product output by product and country

.. csv-table::
   :header: Column, Type, Unit, Description

   ``product``, string, —, "Product identifier (e.g., ``dairy``, ``meat-cattle``, ``eggs``)"
   ``country``, string, —, "ISO 3166-1 alpha-3 country code"
   ``production_mt``, float, Mt, "Production quantity in megatonnes"

**food_consumption.csv** — Food consumption and macronutrients by food and country

.. csv-table::
   :header: Column, Type, Unit, Description

   ``food``, string, —, "Food identifier (e.g., ``wheat``, ``bread``, ``beef``)"
   ``country``, string, —, "ISO 3166-1 alpha-3 country code"
   ``consumption_mt``, float, Mt, "Total consumption in megatonnes"
   ``protein_mt``, float, Mt, "Protein content in megatonnes"
   ``carb_mt``, float, Mt, "Carbohydrate content in megatonnes"
   ``fat_mt``, float, Mt, "Fat content in megatonnes"
   ``cal_pj``, float, PJ, "Energy content in petajoules"
   ``consumption_g_per_person_day``, float, g/person/day, "Per-capita daily consumption"
   ``protein_g_per_person_day``, float, g/person/day, "Per-capita daily protein intake"
   ``carb_g_per_person_day``, float, g/person/day, "Per-capita daily carbohydrate intake"
   ``fat_g_per_person_day``, float, g/person/day, "Per-capita daily fat intake"
   ``cal_kcal_per_person_day``, float, kcal/person/day, "Per-capita daily energy intake"

**food_group_consumption.csv** — Consumption aggregated by food group and country

Has the same columns as ``food_consumption.csv``, except with ``food_group``
instead of ``food``. Food groups aggregate related foods (e.g., ``cereals``,
``fruits``, ``red_meat``) for higher-level analysis.

Example Usage
~~~~~~~~~~~~~

Load statistics in Python for custom analysis:

.. code-block:: python

   import pandas as pd

   # Load crop production
   production = pd.read_csv("results/opt/analysis/scen-default/crop_production.csv")

   # Total wheat production globally
   wheat_total = production[production["crop"] == "wheat"]["production_mt"].sum()

   # Load consumption with per-capita values
   consumption = pd.read_csv("results/opt/analysis/scen-default/food_consumption.csv")

   # Average per-capita protein intake
   avg_protein = consumption["protein_g_per_person_day"].mean()

GHG Intensity
-------------

The GHG intensity analysis computes greenhouse gas emissions attributable to
each unit of food consumed. This provides a consumption-centric view of
impacts, tracing emissions through trade and processing networks back to
production.

**GHG intensity** measures the greenhouse gas emissions per unit of food
consumed (kg CO₂e per kg food). Unlike production-based accounting, this
consumption-attributed metric traces emissions through the entire supply chain:
if wheat is grown in one country, milled into flour, and consumed in another,
the emissions from farming, processing, and transport are all attributed to the
final consumption.

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

Running the GHG Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Extract GHG intensity for a scenario
   tools/smk -j4 --configfile config/<name>.yaml -- \
       results/{name}/analysis/scen-default/ghg_intensity.csv

Output file:

``results/{name}/analysis/scen-{scenario}/ghg_intensity.csv``
   Per-country, per-food GHG intensity including:

   .. csv-table::
      :header: Column, Type, Unit, Description

      ``country``, string, —, "ISO 3166-1 alpha-3 country code"
      ``food``, string, —, "Food identifier"
      ``food_group``, string, —, "Food group"
      ``consumption_mt``, float, Mt, "Consumption quantity"
      ``ghg_kgco2e_per_kg``, float, kgCO2e/kg, "GHG intensity"
      ``ghg_usd_per_t``, float, USD/t, "Monetized GHG damage"

Health Impacts
--------------

The health impacts analysis computes marginal years of life lost (YLL) per
unit of food consumed, based on dose-response curve derivatives at current
population intake levels.

**Health impact** measures the years of life lost (YLL) per unit of food
consumed. This is computed as the marginal effect—the derivative of the
dose-response curve at current population intake levels. Foods with protective
effects (fruits, vegetables, legumes) have negative values, while foods
associated with health risks (processed meat, excess red meat) have positive
values.

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

4. Units are converted from YLL per g/capita/day to YLL per Mt food

The result captures how marginal changes in consumption affect population
health outcomes, accounting for where each country currently sits on the
dose-response curve.

Running the Health Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Extract health impacts for a scenario
   tools/smk -j4 --configfile config/<name>.yaml -- \
       results/{name}/analysis/scen-default/health_impacts.csv

Output file:

``results/{name}/analysis/scen-{scenario}/health_impacts.csv``
   Per-country, per-food-group health impacts including:

   .. csv-table::
      :header: Column, Type, Unit, Description

      ``country``, string, —, "ISO 3166-1 alpha-3 country code"
      ``food_group``, string, —, "Food group (risk factor)"
      ``yll_per_mt``, float, YLL/Mt, "Years of life lost per megatonne"
      ``health_usd_per_t``, float, USD/t, "Monetized health damage"

Sample Results
~~~~~~~~~~~~~~

The following figures show consumption-weighted global averages of GHG
intensity and health impacts by food group:

.. _fig-analysis-ghg:

.. figure:: https://github.com/Sustainable-Solutions-Lab/food-opt/releases/download/doc-figures/analysis_marginal_ghg.png
   :alt: Bar chart showing GHG intensity by food group
   :align: center
   :width: 80%

   Global average GHG intensity by food group (consumption-weighted). Animal
   products (red meat, dairy) show the highest emissions per kg, while
   plant-based foods generally have lower intensities.

.. _fig-analysis-yll:

.. figure:: https://github.com/Sustainable-Solutions-Lab/food-opt/releases/download/doc-figures/analysis_marginal_yll.png
   :alt: Bar chart showing health impact by food group
   :align: center
   :width: 80%

   Global average health impact by food group (consumption-weighted). Negative
   values indicate protective effects (fruits, vegetables, legumes, whole
   grains), while positive values indicate health risks. The magnitude reflects
   the marginal impact at current global intake levels.

Generating Global Average Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate global average plots
   tools/smk -j4 --configfile config/<name>.yaml -- \
       results/{name}/plots/scen-default/marginal_ghg_global.pdf \
       results/{name}/plots/scen-default/marginal_yll_global.pdf

``results/{name}/plots/scen-{scenario}/ghg_health_global.csv``
   Consumption-weighted global averages by food group.
