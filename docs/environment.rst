.. SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
..
.. SPDX-License-Identifier: CC-BY-4.0

Environmental Impacts
=====================

Overview
--------

The environmental module accounts for greenhouse gas emissions, land use change and nitrogen pollution from food production. These impacts are monetized and included in the objective function via configurable prices/penalties.

This is currently a work in progress and not all relevant environmental impacts are implemented and monetized yet.

Greenhouse Gas Emissions
-------------------------

The model tracks three major greenhouse gases using 100-year global warming potentials (GWP100):

* **CO₂** (GWP = 1): From land use change, fuel combustion
* **CH₄** (GWP = 28): From enteric fermentation (ruminants), rice paddies, manure
* **N₂O** (GWP = 265): From nitrogen fertilizer application, manure

All emissions are aggregated to CO₂-equivalent (internally tracked in MtCO₂-eq; the configured price still applies per tonne) for carbon pricing.

Implementation notes (buses, stores, links)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimisation model represents environmental flows with three PyPSA components that are worth keeping in mind:

* **Buses** act as balance sheets. Process components report raw emissions to the ``co2`` and ``ch4`` buses, while a dedicated ``ghg`` bus tracks the combined CO₂-equivalent balance.
* **Links** move quantities between buses, applying efficiencies that encode global warming potentials. ``convert_co2_to_ghg`` has efficiency 1.0, and ``convert_ch4_to_ghg`` uses the configured ``emissions.ch4_to_co2_factor`` (27.2 by default). Every megatonne of CH₄ (after scaling from tonnes) therefore appears on the ``ghg`` bus weighted by its 100-year GWP.
* **Stores** accumulate quantities over the horizon. The extendable ``ghg`` store sits on the combined bus and is priced at ``emissions.ghg_price``. Because neither the ``co2`` nor ``ch4`` buses have stores, their flows must pass through the conversion links before the objective is charged.

With this structure the linear program keeps separate ledgers for each greenhouse gas while charging the objective using a single priced stock of CO₂-equivalent. Scenario files can tighten or relax climate policy simply by changing the configuration values—no code modifications are required.

Sources of Emissions
~~~~~~~~~~~~~~~~~~~~

**Crop Production**:
  * N₂O from fertilizer (direct and indirect)
  * CH₄ from flooded rice cultivation
  * CO₂ from machinery/fuel (if included)

**Livestock**:
  * CH₄ from enteric fermentation (ruminants) - see :ref:`enteric-fermentation`
  * CH₄ and N₂O from manure management (all animals) - see :ref:`manure-management`
  * CO₂ from feed production (indirect)

**Land Use Change**:
  * CO₂ from clearing vegetation (forest, grassland → cropland)
  * Soil carbon losses

Direct N₂O emission factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model uses the 2019 Refinement to the IPCC Guidelines for National Greenhouse Gas Inventories to parameterise direct N₂O emissions from managed soils. Table 11.1 (updated) is reproduced below to make the default emission factors and their uncertainty ranges readily accessible when configuring fertilizer-related pathways.

.. list-table:: Default emission factors to estimate direct N₂O emissions from managed soils (IPCC, 2019 Refinement - Table 11.1)
   :header-rows: 1
   :widths: 32 11 14 25 9 9

   * - Emission factor
     - Aggregated default value
     - Aggregated uncertainty range
     - Disaggregation
     - Default value
     - Uncertainty range
   * - EF\ :sub:`1` for N additions from synthetic fertilisers, organic amendments and crop residues, and N mineralised from mineral soil as a result of loss of soil carbon [kg N₂O-N (kg N)\ :sup:`-1`]
     - 0.010
     - 0.002 – 0.018
     - Synthetic fertiliser inputs in wet climates

       Other N inputs in wet climates

       All N inputs in dry climates
     - 0.016 (wet synthetic)

       0.006 (wet other)

       0.005 (dry)
     - 0.013 – 0.019

       0.001 – 0.011

       0.000 – 0.011
   * - EF\ :sub:`1FR` for flooded rice fields [kg N₂O-N (kg N)\ :sup:`-1`]
     - 0.004
     - 0.000 – 0.029
     - Continuous flooding

       Single and multiple drainage
     - 0.003

       0.005
     - 0.000 – 0.010

       0.000 – 0.016
   * - EF\ :sub:`3PRP,CPP` for cattle (dairy, non-dairy and buffalo), poultry and pigs [kg N₂O-N (kg N)\ :sup:`-1`]
     - 0.004
     - 0.000 – 0.014
     - Wet climates

       Dry climates
     - 0.006

       0.002
     - 0.000 – 0.027

       0.000 – 0.007
   * - EF\ :sub:`3PRP,SO` for sheep and “other animals” [kg N₂O-N (kg N)\ :sup:`-1`]
     - 0.003
     - 0.000 – 0.010
     - –
     - –
     - –

.. _enteric-fermentation:

Enteric Fermentation (CH₄)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ruminant livestock (cattle, sheep, goats, buffalo) produce methane during digestion through microbial fermentation in the rumen. The model calculates enteric CH₄ emissions using IPCC Tier 2 methodology based on feed-specific methane yields.

Methodology
^^^^^^^^^^^

Enteric methane emissions are calculated as:

.. math::

   \text{CH}_4 = \text{DMI} \times \text{MY}_\text{enteric}

where:
  * **DMI** is dry matter intake (kg feed/day or t feed/year)
  * **MY**\ :sub:`enteric` is the enteric methane yield (g CH₄ per kg DMI)

The methane yield depends primarily on feed digestibility and fiber content. Higher-quality feeds (grains, concentrates) produce less CH₄ per unit intake than low-quality forages because they ferment more efficiently with less methane as a byproduct.

IPCC Conversion Factors
^^^^^^^^^^^^^^^^^^^^^^^^

The model uses methane yields from the `2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories <https://www.ipcc.ch/report/2019-refinement-to-the-2006-ipcc-guidelines-for-national-greenhouse-gas-inventories/>`_ [1]_, Volume 4, Chapter 10, Table 10.12 (Methane Conversion Factors for Cattle and Buffalo).

Feed categories are mapped to IPCC dietary classes:

* **Roughage** (23.3 g CH₄/kg DMI): High-forage diets >75% forage, digestible energy (DE) ≤ 62%, typical of extensive grazing systems
* **Forage** (21.0 g CH₄/kg DMI): Mixed rations 15-75% forage with grain/silage, DE 62-71%, typical of semi-intensive dairy and beef
* **Grain** (13.6 g CH₄/kg DMI): Concentrate-based feedlot diets 0-15% forage, DE ≥ 72%, typical of intensive finishing systems
* **Protein** (13.6 g CH₄/kg DMI): High-protein concentrates, same as grain category

Monogastric animals (pigs, poultry) produce negligible enteric methane and are not included in this calculation.

Data Sources
^^^^^^^^^^^^

* **IPCC values**: ``data/ipcc_enteric_methane_yields.csv`` maps feed categories to MY values from IPCC (2019) Table 10.12
* **Feed properties**: ``processing/{name}/ruminant_feed_categories.csv`` generated from GLEAM 3.0 [2]_ Supplement S1, Table S.3.3 (Ruminant Nutrition Parameters)
* **Feed mapping**: ``data/gleam_feed_mapping.csv`` links model feed items to GLEAM feed categories

Implementation
^^^^^^^^^^^^^^

Enteric emissions are calculated in ``workflow/scripts/build_model.py`` within the ``add_feed_to_animal_product_links()`` function:

1. Feed items are categorized by digestibility into roughage/forage/grain/protein pools (``workflow/scripts/categorize_feeds.py``)
2. Each category is assigned an MY value from ``data/ipcc_enteric_methane_yields.csv``
3. For each animal production link, CH₄ emissions per tonne of feed intake are calculated and attached to ``bus2`` (methane bus)
4. Emissions scale linearly with feed consumption in the optimization

.. _manure-management:

Manure Management (CH₄)
~~~~~~~~~~~~~~~~~~~~~~~

All livestock produce methane emissions from manure storage, handling, and treatment. Unlike enteric fermentation, manure CH₄ affects both ruminants and monogastrics (pigs, poultry), with emissions varying significantly by management system.

Methodology
^^^^^^^^^^^

Manure methane emissions follow IPCC Tier 2 methodology based on volatile solids excretion and system-specific methane conversion factors:

.. math::

   \text{CH}_4\text{_manure} = \text{VS} \times B_0 \times \text{MCF} \times 0.67

where:
  * **VS** is volatile solids excretion (kg VS per kg feed DM intake)
  * **B**\ :sub:`0` is maximum methane producing capacity (m³ CH₄ per kg VS)
  * **MCF** is the methane conversion factor (fraction 0-1, varies by management system and climate)
  * **0.67** converts m³ CH₄ to kg CH₄

Volatile Solids Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Volatile solids represent the organic fraction of manure available for anaerobic decomposition. The model calculates VS using an adapted version of IPCC Equation 10.24:

.. math::

   \text{VS} = (1 - \text{digestibility} + \text{UE}) \times (1 - \text{ash}/100)

where:
  * **Digestibility** is the fraction of feed digested by the animal (from GLEAM feed properties)
  * **UE** is urinary energy excretion as a fraction of gross energy intake:

    * 0.04 for ruminants (cattle, sheep, goats)
    * 0.02 for pigs
    * 0.00 for poultry (minimal urinary losses)

  * **Ash** is the ash content of feed (% dry matter, from ``data/feed_ash_content.csv`` based on `feedtables.com <https://www.feedtables.com/>`_)

The formula accounts for:
  * Undigested feed (1 - digestibility)
  * Urinary excretion (UE)
  * Mineral content that doesn't decompose (ash fraction)

Maximum Methane Producing Capacity (B₀)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

B₀ represents the theoretical maximum CH₄ yield from complete anaerobic digestion of manure volatile solids. Values are animal-specific:

.. list-table:: B₀ values by animal product (IPCC 2019, Table 10.16)
   :header-rows: 1
   :widths: 40 30 30

   * - Animal Product
     - B₀ (m³ CH₄/kg VS)
     - Source
   * - Dairy cattle
     - 0.24
     - IPCC Table 10.16, high productivity
   * - Beef cattle
     - 0.18
     - IPCC Table 10.16, high productivity
   * - Pigs
     - 0.45
     - IPCC Table 10.16
   * - Poultry (broilers)
     - 0.36
     - IPCC Table 10.16
   * - Poultry (layers/eggs)
     - 0.39
     - IPCC Table 10.16

Data source: ``data/ipcc_manure_methane_producing_capacity.csv``

Methane Conversion Factors (MCF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MCF represents the fraction of B₀ actually realized under specific management conditions. It varies by:

* **Management system**: Liquid systems (lagoons, slurry pits) have high MCF (0.4-0.8), solid systems (composting, daily spread) have low MCF (0.001-0.05)
* **Climate zone**: Warmer climates increase anaerobic activity and MCF
* **Storage duration**: Longer storage increases MCF

The model uses MCF values from IPCC (2019) Table 10.17, which provides system-specific and climate-specific factors for 21 manure management systems.

**Current simplification**: MCF values are averaged across climate zones for each management system (``workflow/scripts/calculate_manure_emissions.py``). This will be refined when climate zone data is added to modeling regions, allowing for country-specific and region-specific emission factors.

Manure Management System Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Real-world manure CH₄ emissions reflect a weighted average across multiple management practices. The model uses global system distributions from `GLEAM 3.0 <https://foodandagricultureorganization.shinyapps.io/GLEAMV3_Public/>`_ [2]_ (Supplement S1, Tables 4.4 and 4.5):

* **Cattle**: Primarily pasture/paddock (low MCF ~0.005) with some confinement and liquid systems
* **Pigs**: Mix of solid storage, liquid slurry, and pit systems (higher MCF ~0.1-0.4)
* **Poultry**: Mostly litter-based and solid systems (moderate MCF ~0.01-0.04)

The weighted MCF is calculated as:

.. math::

   \text{MCF}_\text{weighted} = \sum_{i} f_i \times \text{MCF}_i

where **f**\ :sub:`i` is the fraction of manure managed in system *i* (from ``data/gleam_tables/manure_management_systems_fraction.csv``).

Data Sources
^^^^^^^^^^^^

* **B₀ values**: ``data/ipcc_manure_methane_producing_capacity.csv`` (IPCC 2019 Table 10.16)
* **MCF values**: ``data/ipcc_manure_methane_conversion_factors.csv`` (IPCC 2019 Table 10.17)
* **MMS distributions**: ``data/gleam_tables/manure_management_systems_fraction.csv`` (GLEAM 3.0 Supplement S1)
* **Ash content**: ``data/feed_ash_content.csv`` (from feedtables.com, matched to model feed entities)
* **Feed properties**: ``processing/{name}/ruminant_feed_categories.csv`` and ``processing/{name}/monogastric_feed_categories.csv`` (digestibility from GLEAM 3.0)

Implementation
^^^^^^^^^^^^^^

Manure emissions are calculated in ``workflow/scripts/calculate_manure_emissions.py`` and integrated into the model via ``workflow/scripts/build_model.py``:

1. **Preprocessing** (``calculate_manure_emissions.py``):

   * Calculate VS excretion for each feed category using digestibility and ash content
   * Average MCF across climate zones for each management system (temporary simplification)
   * Compute weighted MCF for each animal product using GLEAM MMS distributions
   * Calculate CH₄ emissions per kg feed intake: VS × B₀ × MCF\ :sub:`weighted` × 0.67
   * Generate ``processing/{name}/manure_ch4_emission_factors.csv`` with emissions by country, product, and feed category

2. **Model integration** (``build_model.py``):

   * Load manure emission factors from ``processing/{name}/manure_ch4_emission_factors.csv``
   * In ``add_feed_to_animal_product_links()``, combine enteric and manure CH₄:

     .. math::

        \text{CH}_4\text{/t feed} = \text{MY}_\text{enteric} + \text{MY}_\text{manure}

   * Attach total CH₄ to ``bus2`` (methane bus) for all animal production links
   * Emissions scale with feed consumption in the optimization

Example Calculation
^^^^^^^^^^^^^^^^^^^^

**Scenario**: Dairy cow fed on forage (ruminant_forage category)

**Parameters**:
  * Digestibility: 0.61 (from GLEAM)
  * Ash content: 7.15% (average for forage feeds)
  * Urinary fraction (UE): 0.04 (ruminants)
  * B₀ (dairy): 0.24 m³ CH₄/kg VS
  * Weighted MCF (dairy, global average): 0.034 (mostly pasture-based)

**Calculation**:

1. VS excretion:

   .. math::

      \text{VS} = (1 - 0.61 + 0.04) \times (1 - 7.15/100) = 0.43 \times 0.9285 = 0.399 \text{ kg VS/kg DMI}

2. Manure CH₄:

   .. math::

      \text{CH}_4 = 0.399 \times 0.24 \times 0.034 \times 0.67 = 0.00217 \text{ kg CH₄/kg DMI} = 2.17 \text{ g CH₄/kg DMI}

3. Total CH₄ (enteric + manure):

   .. math::

      \text{Total} = 21.0 \text{ (enteric)} + 2.17 \text{ (manure)} = 23.17 \text{ g CH₄/kg DMI}

This shows that for dairy cattle on forage diets, manure contributes ~9% of total CH₄ emissions, with enteric fermentation being dominant. For monogastrics (pigs, poultry), where enteric emissions are zero, manure is the sole CH₄ source.

Manure Nitrogen Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to methane emissions, the model tracks nitrogen flows from livestock manure, accounting for both the fertilizer value and N₂O emissions from manure application.

**Nitrogen Mass Balance**

Nitrogen excreted in manure is calculated from a simple mass balance:

.. math::

   N_\text{excretion} = N_\text{feed} - N_\text{product}

where:
  * **N**\ :sub:`feed` is the nitrogen content of feed (from GLEAM feed properties, g N/kg DM)
  * **N**\ :sub:`product` is the nitrogen content of the animal product, derived from protein content

Protein-to-Nitrogen Conversion
"""""""""""""""""""""""""""""""

Animal product nitrogen content is calculated from protein using the standard Jones factor of 6.25:

.. math::

   N_\text{product} = \frac{\text{Protein}}{6.25}

This factor reflects that proteins average ~16% nitrogen by mass (1/6.25 ≈ 0.16). While specific proteins vary (5.18-6.38), 6.25 is the `FAO-recommended general conversion factor <https://www.fao.org/4/y5022e/y5022e03.htm>`_ for mixed animal products [3]_.

Protein content is sourced from USDA FoodData Central (``data/nutrition.csv``).

Manure Nitrogen as Fertilizer
""""""""""""""""""""""""""""""

Not all excreted nitrogen becomes available as fertilizer due to volatilization and other losses during storage and handling. The model applies a configurable recovery fraction:

.. math::

   N_\text{fertilizer} = N_\text{excretion} \times f_\text{recovery}

where **f**\ :sub:`recovery` is configured via ``primary.fertilizer.manure_n_to_fertilizer`` (default: 0.75, representing 75% recovery and 25% losses).

This manure N is added to the global fertilizer pool (``n_fertilizer`` bus) where it competes with and substitutes for synthetic fertilizer, subject to the global fertilizer limit.

**Special Case: Grazing Systems**

For animals fed from the ``ruminant_grassland`` feed category (pasture grazing), manure is deposited directly on pasture and not collected. For these systems:

* **N**\ :sub:`fertilizer` = 0 (no manure collection)
* N₂O emissions are still calculated from the full excreted nitrogen (representing pasture deposition emissions)

This distinction reflects the practical reality that grazing manure cannot be redistributed to cropland, while maintaining accurate N₂O accounting for pasture emissions.

N₂O Emissions from Manure Application
""""""""""""""""""""""""""""""""""""""

Applied manure nitrogen produces direct N₂O emissions using the same IPCC Tier 1 methodology as synthetic fertilizer:

.. math::

   N_2O = N_\text{fertilizer} \times EF_\text{manure} \times \frac{44}{28}

where:
  * **EF**\ :sub:`manure` is the emission factor (kg N₂O-N per kg manure N applied)
  * **44/28** converts N₂O-N to N₂O (molecular weight ratio)

The default emission factor (``primary.fertilizer.manure_n2o_factor = 0.010``) matches the synthetic fertilizer factor and corresponds to the IPCC 2019 aggregated default from Table 11.1 (see Direct N₂O emission factors section above).

Configuration
"""""""""""""

Manure nitrogen management is configured under ``primary.fertilizer``:

.. code-block:: yaml

   primary:
     fertilizer:
       manure_n_to_fertilizer: 0.75  # Fraction of excreted N available as fertilizer
       manure_n2o_factor: 0.010      # kg N₂O-N per kg manure N applied

Implementation
""""""""""""""

Manure nitrogen is implemented in ``workflow/scripts/build_model.py`` within the ``add_feed_to_animal_product_links()`` function:

1. For each animal production link, calculate:

   * N excretion from feed N content (GLEAM) minus product N content (protein ÷ 6.25)
   * Manure N available as fertilizer:

     - For ``ruminant_grassland``: 0 (manure deposited on pasture)
     - For other feed categories: N excretion × recovery fraction

   * N₂O emissions:

     - For ``ruminant_grassland``: N excretion × emission factor × 44/28 (all excreted N)
     - For other feed categories: manure N × emission factor × 44/28 (collected N only)

2. Attach outputs to the link:

   * ``bus3``: ``n_fertilizer`` (manure N contributing to fertilizer pool)
   * ``bus4``: ``n2o`` (direct N₂O emissions)

This creates a closed nutrient cycle where livestock manure offsets synthetic fertilizer demand while incurring proportional N₂O emissions, with grazing systems correctly accounting for on-pasture deposition.

Example Calculation
"""""""""""""""""""

**Scenario**: Beef cattle on forage diet (ruminant_forage)

**Parameters**:
  * Feed N: 19.5 g N/kg DM (from GLEAM)
  * Product protein: 18.59 g/100g (meat-cattle, from USDA FoodData Central)
  * Product N: 18.59 ÷ 6.25 = 2.97 g N/100g = 29.7 g N/kg
  * Feed conversion efficiency: 0.15 (6.67 kg feed per kg product)
  * Recovery fraction: 0.75
  * N₂O emission factor: 0.010

**Calculation** (per tonne of feed DM):

1. N inputs and outputs:

   .. math::

      N_\text{feed} &= 19.5 \text{ g/kg} = 0.0195 \text{ t N/t feed} \\
      \text{Product output} &= 0.15 \text{ t product/t feed} \\
      N_\text{product} &= 29.7 \text{ g/kg} \times 0.15 \text{ t/t} = 0.00446 \text{ t N/t feed}

2. N excretion:

   .. math::

      N_\text{excretion} = 0.0195 - 0.00446 = 0.0150 \text{ t N/t feed}

3. Manure N fertilizer:

   .. math::

      N_\text{fertilizer} = 0.0150 \times 0.75 = 0.0113 \text{ t N/t feed}

4. N₂O emissions:

   .. math::

      N_2O = 0.0113 \times 0.010 \times \frac{44}{28} = 0.000178 \text{ t N}_2\text{O/t feed}

**Result**: Each tonne of feed produces 11.3 kg of manure N (contributing to the fertilizer pool) and 178 g of N₂O emissions.

**Grazing Example**: Beef cattle on pasture (ruminant_grassland)

Using the same parameters as above but with pasture grazing:

1-2. N excretion remains the same: 0.0150 t N/t feed

3. Manure N fertilizer (grazing):

   .. math::

      N_\text{fertilizer} = 0 \text{ (manure deposited on pasture, not collected)}

4. N₂O emissions (from pasture deposition):

   .. math::

      N_2O = 0.0150 \times 0.010 \times \frac{44}{28} = 0.000236 \text{ t N}_2\text{O/t feed}

**Result**: No manure N enters the fertilizer pool, but 236 g N₂O per tonne feed is emitted from pasture deposition (higher than confined systems since all excreted N remains on pasture).

Future Refinements
^^^^^^^^^^^^^^^^^^

Planned improvements to manure emissions modeling:

* **Climate zone differentiation**: Use actual climate zones for each region instead of averaging MCF across zones
* **Country-specific MMS distributions**: Currently all countries use global GLEAM averages
* **Indirect N₂O emissions**: Add indirect N₂O from volatilization and leaching of manure N (IPCC EF₄ and EF₅)
* **Manure management system emissions**: Differentiate N₂O emission factors by storage system (currently uses field-application factor for all)

.. [3] FAO (2003). *Food energy - methods of analysis and conversion factors*. FAO Food and Nutrition Paper 77. Report of a Technical Workshop, Rome, 3-6 December 2002. https://www.fao.org/4/y5022e/y5022e03.htm

Carbon Pricing
~~~~~~~~~~~~~~

All GHG emissions (CO₂, CH₄, N₂O) are priced at a configurable rate:

.. literalinclude:: ../config/default.yaml
   :language: yaml
   :start-after: # --- section: emissions ---
   :end-before: # --- section: crops ---

.. [1] IPCC (2019). *2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories*, Volume 4: Agriculture, Forestry and Other Land Use. https://www.ipcc.ch/report/2019-refinement-to-the-2006-ipcc-guidelines-for-national-greenhouse-gas-inventories/

.. [2] FAO (2022). *Global Livestock Environmental Assessment Model (GLEAM) 3.0*. Food and Agriculture Organization of the United Nations. https://www.fao.org/gleam/

Land Use Change
---------------

Land-use change (LUC) emissions capture the carbon consequences of converting land between natural vegetation, cropland, pasture, and spared (actively rewilded) states. The model derives annualized per-hectare coefficients for each resource class and water supply that quantify the net CO₂ flux associated with allocating an additional hectare to a specific land use.

Conceptual overview
~~~~~~~~~~~~~~~~~~~

For every grid cell on the common suitability grid, the workflow computes three main quantities:

* **Pulse emissions (:math:`P_{i,u}`)** – the one-off release (or uptake) that occurs when land transitions from its natural state to land use :math:`u` (cropland or pasture). We estimate above-ground biomass (AGB), below-ground biomass (BGB), and soil organic carbon (SOC) stocks for both the natural and agricultural equilibria, then convert the difference to CO₂ using the stoichiometric factor :math:`44/12`.
* **Annual regrowth (:math:`R_i`)** – the ongoing sequestration potential when land is spared or allowed to regrow. Regrowth is only applied where forest cover is present in the baseline land-cover map.
* **Managed flux (:math:`M_{i,u}`)** – ongoing emissions from managed systems (e.g., peat oxidation, continuous tillage). The current implementation sets :math:`M_{i,u} = 0` everywhere as a simplifying assumption.

The per-hectare land-use change factor (LEF) combines these components over the planning horizon :math:`H` (years) configured in ``config/default.yaml``:

.. math::

   \mathrm{LEF}_{i,u} = \frac{P_{i,u}}{H} + (R_i - M_{i,u})

LEFs are computed for three uses (``cropland``, ``pasture``, ``spared``). Cropland and pasture incur positive costs when they release carbon; spared land yields negative LEFs because regrowth produces a CO₂ sink. Area-weighted aggregation over resource classes produces region-level coefficients that the optimisation layer consumes.

Input datasets
~~~~~~~~~~~~~~

The LUC pipeline harmonises several global datasets to the common grid:

* Land cover fractions and forest masks from Copernicus ESA CCI land cover (:ref:`copernicus-land-cover`)
* Above-ground biomass from ESA Biomass CCI v6.0 (:ref:`esa-biomass-cci`)
* Soil organic carbon stocks (0–30 cm) from ISRIC SoilGrids 2.0 (:ref:`soilgrids-soc`), scaled to 1 m depth using IPCC Tier 1 factors
* Natural forest regrowth rates from Cook-Patton & Griscom (2020) (:ref:`cook-patton-regrowth`)
* IPCC Tier 1 below-ground biomass ratios, soil depletion factors, and agricultural equilibrium assumptions stored in ``data/luc_zone_parameters.csv``

These layers are reprojected, resampled, and combined by dedicated Snakemake rules to produce per-cell biomass/SOC stocks, forest masks, and regrowth rates ready for downstream processing. Figure :ref:`fig-luc-inputs` summarises the harmonised rasters on the common model grid.

.. _fig-luc-inputs:

.. figure:: https://github.com/Sustainable-Solutions-Lab/food-opt/releases/download/doc-figures/environment_luc_inputs.png
   :alt: Global maps showing forest fraction, biomass, soil carbon, and regrowth inputs
   :align: center
   :width: 95%

   Land-use change input layers harmonised to the modelling grid: forest fraction (Copernicus CCI), above-ground biomass (ESA Biomass CCI v6.0), soil organic carbon 0–30 cm (SoilGrids 2.0), and natural forest regrowth potential (Cook-Patton & Griscom, 2020).

Model integration
~~~~~~~~~~~~~~~~~

The land-use change workflow consists of two scripts:

1. ``prepare_luc_inputs.py`` aligns the raw rasters to the resource-class grid and stores intermediate masks and carbon pools under ``processing/{config}/luc/``.
2. ``build_luc_carbon_coefficients.py`` derives pulse emissions, annual LEFs, and aggregates them to ``luc_carbon_coefficients.csv``.

During model construction, ``build_model.py`` loads these coefficients, converts the LEFs to marginal CO₂ flows (MtCO₂ per Mha-year), and attaches them to:

* Crop production links (cropland LEFs)
* Grazing links that supply ruminant feed (pasture LEFs)
* Spared-land allocation links that credit regrowth sinks (filtered by current biomass—see below)

All flows connect to a single global ``co2`` bus. A flexible CO₂ store with marginal cost equal to the carbon price (``emissions.ghg_price``) accumulates the net balance, so positive flows are penalised while negative flows earn credits. This approach keeps LUC accounting endogenous to the optimisation problem while leveraging a consistent carbon price alongside other greenhouse-gas sources. The spatial pattern of the resulting LEFs is shown in :ref:`fig-luc-lef`.

Spared land filtering
~~~~~~~~~~~~~~~~~~~~~

Regrowth sequestration rates from Cook-Patton et al. (2020) represent **young regenerating forest** (0-30 years) on previously cleared or degraded land. They do not apply to mature forests, which have already accumulated most of their carbon stock and exhibit near-zero net sequestration.

To avoid incorrectly crediting sequestration on mature forest, the LEF calculation for spared land includes a conditional that sets the sequestration benefit to zero where **current above-ground biomass (AGB)** exceeds a configurable threshold (``luc.spared_land_agb_threshold_tc_per_ha``, default 20 tC/ha). Specifically:

.. math::

   \mathrm{LEF}_{\mathrm{spared}} = \begin{cases}
   -R & \text{if } \mathrm{AGB} \leq \text{threshold} \\
   0 & \text{if } \mathrm{AGB} > \text{threshold}
   \end{cases}

This ensures:

* Low-biomass areas (recently cleared or degraded land suitable for agriculture) receive negative LEFs (sequestration credits) if left unused
* High-biomass areas (mature tropical rainforest, boreal forest) receive zero spared-land LEF—their carbon value is already captured via high pulse emissions if converted, but they are not credited for additional regrowth

The threshold of 20 tC/ha is intermediate between typical agricultural land (0-10 tC/ha) and mature forest (50-200+ tC/ha). Areas above this threshold are assumed to represent established vegetation that would not exhibit the rapid early-successional regrowth rates quantified by Cook-Patton et al.

.. _fig-luc-lef:

.. figure:: https://github.com/Sustainable-Solutions-Lab/food-opt/releases/download/doc-figures/environment_luc_lef.png
   :alt: Global maps of cropland, pasture, and spared land LEFs
   :align: center
   :width: 95%

   Annualised land-use change emission factors (LEFs) used in the optimisation. Warm colors indicate positive emissions costs (CO₂ release), while cool colors represent sequestration credits.

Limitations and assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current implementation makes several simplifying assumptions that should be considered when interpreting results:

* **Climatic zones**: Zones (tropical, temperate, boreal) are assigned by latitude only (tropical: :math:`\lvert \phi \rvert < 23.5^\circ`, boreal: :math:`\lvert \phi \rvert \ge 50^\circ`, temperate: otherwise). This does not account for altitude effects (e.g., highland tropics behave more like temperate zones) or local climate variations. A future enhancement would use actual biome or Köppen-Geiger climate classifications.

* **Agricultural biomass stocks**: Cropland and pasture equilibrium above-ground biomass is assumed to be negligible (0 tC/ha) for annual crops. This is a conservative assumption appropriate for grain crops where biomass is harvested annually, but underestimates carbon storage in perennial crops (orchards, oil palm, coffee) and improved pastures. See ``data/luc_zone_parameters.csv`` for the zone-specific parameters.

* **Forest mask threshold**: Regrowth sequestration is only applied to cells with ≥20% forest cover in the baseline land cover map. This threshold can be adjusted via ``config['luc']['forest_fraction_threshold']`` (default: 0.2). The choice of 20% is intermediate between FAO's forest definition (≥10% tree cover) and stricter definitions (≥30%).

* **Soil organic carbon depth**: SOC stocks in the 0-30 cm layer (from SoilGrids) are scaled to 1 m depth using zone-specific factors from ``data/luc_zone_parameters.csv``. **TODO**: These factors require verification against IPCC 2006/2019 Guidelines Volume 4 Chapter 2 to ensure they match the intended Tier 1 methodology.

* **Managed flux**: Set to zero everywhere (:math:`M_{i,u} = 0`), meaning ongoing emissions from agricultural management (e.g., peat oxidation, tillage-induced decomposition) are not currently modeled. Future work could incorporate organic soil maps and management-specific emission factors.

Nitrogen Pollution
------------------

Fertilizer application causes nitrogen pollution via:

* **Leaching**: NO₃⁻ contaminating groundwater
* **Runoff**: Eutrophication of rivers/lakes
* **Volatilization**: NH₃ → N₂O emissions

Global Fertilizer Limit
~~~~~~~~~~~~~~~~~~~~~~~

To prevent excessive pollution:

.. literalinclude:: ../config/default.yaml
   :language: yaml
   :start-after: # --- section: primary ---
   :end-before: # --- section: emissions ---

This caps total nitrogen-phosphorus-potassium application globally, forcing efficient use.

Nitrogen Use Efficiency (NUE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Crop-specific fertilizer requirements (in ``data/crops.csv``) implicitly include NUE (currently mock data). More efficient crops (legumes, which fix nitrogen) require less fertilizer.
