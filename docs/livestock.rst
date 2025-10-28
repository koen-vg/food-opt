.. SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
..
.. SPDX-License-Identifier: CC-BY-4.0

Livestock & Grazing
===================

Overview
--------

The livestock module models animal product production (meat, dairy, eggs) through two distinct production systems:

* **Grazing-based**: Animals feed on managed grasslands
* **Feed-based**: Animals consume crops as concentrated feed

Animal Products
---------------

The model includes five major animal product categories configured in ``config/default.yaml``:

.. literalinclude:: ../config/default.yaml
   :language: yaml
   :start-after: # --- section: animal_products ---
   :end-before: # --- section: food_groups ---

Each product can be produced via either production system, with different feed requirements and efficiencies.

Production Systems
------------------

Grazing-Based Production
~~~~~~~~~~~~~~~~~~~~~~~~

**Concept**: Animals graze on managed grasslands, converting grass biomass to animal products.

**Inputs**:
  * Land (per region and resource class, similar to cropland)
  * Managed grassland yields from ISIMIP LPJmL model

**Process**:
  1. Grassland yields (t dry matter/ha/year) are computed per region and resource class
  2. Feed conversion ratios translate grass biomass → animal products
  3. Land allocation to grazing competes with cropland expansion

**Configuration**: Enable/disable grazing with ``grazing.enabled: true``

Feed-Based Production
~~~~~~~~~~~~~~~~~~~~~

**Concept**: Animals consume crops (grains, soybeans, etc.) as concentrated feed.

**Inputs**:
  * Crops from crop production buses
  * Feed conversion ratios (kg crop → kg animal product)

**Process**:
  1. Crops are allocated to animal feed (competing with direct human consumption)
  2. Feed conversion links transform crop inputs to animal products
  3. Multiple crops can contribute (e.g., maize + soybean for poultry)

.. _grassland-yields:

Grassland Yields
----------------

Grazing supply is determined by managed grassland yields from the ISIMIP LPJmL historical simulation.

Data Source
~~~~~~~~~~~

**Dataset**: ISIMIP2b managed grassland yields (historical)

**Resolution**: 0.5° × 0.5° gridded annual yields

**Variable**: Above-ground dry matter production (t/ha/year)

**Processing**: ``workflow/scripts/build_grassland_yields.py``

Aggregation follows the same resource class structure as crops:

1. Load grassland yield NetCDF
2. Aggregate by (region, resource_class) using area-weighted means
3. Output CSV with yields in t/ha/year

.. figure:: https://github.com/Sustainable-Solutions-Lab/food-opt/releases/download/doc-figures/grassland_yield.png
   :alt: Managed grassland yield potential
   :width: 100%
   :align: center

   Global distribution of managed grassland yield potential (tonnes dry matter per hectare per year) from ISIMIP LPJmL historical simulations

Feed Conversion
---------------

The model uses feed conversion ratios to link feed inputs to animal outputs, with explicit categorization by feed quality to enable accurate CH₄ emissions tracking.

Feed System Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

The feed system uses **nine distinct feed pools** that combine animal type with feed quality:

* **Ruminant pools**: ``ruminant_roughage``, ``ruminant_forage``, ``ruminant_grassland``, ``ruminant_grain``, ``ruminant_protein``
* **Monogastric pools**: ``monogastric_low_quality``, ``monogastric_grain``, ``monogastric_energy``, ``monogastric_protein``

This categorization enables the model to:

1. Differentiate methane emissions using GLEAM feed digestibility classes (roughage/forage vs. grain/protein)
2. Route crops, residues, and processing byproducts to appropriate feed pools based on nutritional properties
3. Model production system choices (e.g., roughage-dominated beef vs. high-grain finishing rations)
4. Distinguish between grazing (grassland) and confinement feeding systems for nitrogen management

Feed Properties (Generated from GLEAM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feed properties (digestibility, metabolizable energy, protein content) are automatically generated from GLEAM 3.0 data during workflow execution. The workflow produces two files in ``processing/{name}/``:

* ``ruminant_feed_properties.csv``: Properties for all feeds used by ruminants
* ``monogastric_feed_properties.csv``: Properties for all feeds used by monogastrics

Each file contains:

* ``feed_item``: Item name (e.g., "maize", "wheat bran")
* ``source_type``: Either "crop" or "food" (byproduct)
* ``digestibility``: Digestible fraction (0-1)
* ``ME_MJ_per_kg_DM``: Metabolizable energy (MJ per kg dry matter)
* ``CP_pct_DM``: Crude protein (% of dry matter)
* ``ash_pct_DM``: Ash content (% of dry matter)
* ``NDF_pct_DM``: Neutral detergent fiber (% of dry matter)

These properties are extracted from the GLEAM 3.0 supplement using ``data/gleam_feed_mapping.csv`` to map between model feed items and GLEAM feed categories.

**Feed quality categories** (assigned based on digestibility and protein content):

* **Ruminant feeds**:

  * **Roughage**: Low digestibility (<0.55), high-fiber forages (crop residues, straw)
  * **Forage**: Medium digestibility (0.55-0.70), improved forages (silage maize, alfalfa)
  * **Grassland**: Managed pasture grazing (special category for nitrogen management; manure deposited on pasture)
  * **Grain**: High digestibility (0.70-0.90), low protein (<20%), energy concentrates (maize, wheat)
  * **Protein**: High digestibility (>0.90), high protein (≥20%), protein concentrates (soybean meal)

* **Monogastric feeds**:

  * **Low quality**: Low metabolizable energy (<12 MJ/kg DM), bulky feeds and byproducts
  * **Grain**: Medium energy (12-14 MJ/kg DM), low protein (<20%), cereal grains
  * **Energy**: High energy (>14 MJ/kg DM), low protein (<20%), fats and high-energy feeds
  * **Protein**: High protein (≥20%), protein concentrates (soybean meal, fish meal)

Byproducts from food processing (with ``source_type=food``) are automatically excluded from human consumption and can only be used as animal feed.

Feed Conversion Efficiencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feed conversion efficiencies (tonnes product per tonne feed DM) are generated automatically from **Wirsenius (2000)** regional feed energy requirements combined with GLEAM 3.0 feed category energy values.

**Data source**: Wirsenius, S. (2000). *Human Use of Land and Organic Materials: Modeling the Turnover of Biomass in the Global Food System*. Table 3.9. https://publications.lib.chalmers.se/records/fulltext/827.pdf

**Generation workflow**:

1. **Regional feed energy requirements** from Wirsenius (2000) provide MJ per kg product output for eight world regions
2. **Energy conversion for ruminants**: Net energy (NE) requirements converted to metabolizable energy (ME) using NRC (2000) efficiency factors:

   * k_m = 0.60 (maintenance)
   * k_g = 0.40 (growth)
   * k_l = 0.60 (lactation)

3. **Feed category energy content** from GLEAM 3.0 provides ME (MJ per kg DM) for each feed quality category
4. **Efficiency calculation**: efficiency = ME_feed / ME_requirement (tonnes product per tonne feed DM)

**Output**: ``processing/{name}/feed_to_animal_products.csv`` with columns:

* ``product``: Product name (e.g., "cattle meat", "dairy")
* ``feed_category``: Feed pool (e.g., ``ruminant_forage``, ``ruminant_grain``, ``monogastric_grain``)
* ``efficiency``: Feed conversion efficiency (t product / t feed DM)
* ``region``: Region label (averaged over configured regions)
* ``notes``: Description with inverse feed requirement

**Configuration**: Specify which Wirsenius regions to average in ``config/default.yaml``:

.. code-block:: yaml

   animal_products:
     wirsenius_regions:
     - North America & Oceania
     - West Europe

Available regions: East Asia, East Europe, Latin America & Caribbean, North Africa & West Asia, North America & Oceania, South & Central Asia, Sub-Saharan Africa, West Europe

If ``wirsenius_regions`` is null or empty, a global average across all eight regions is used.

**Example efficiencies** (North America & Oceania + West Europe average):

* Cattle meat from forage: 0.039 t/t (25.8 t DM feed per tonne meat)
* Cattle meat from grain: 0.053 t/t (18.9 t DM feed per tonne meat)
* Dairy from forage: 0.570 t/t (1.8 t DM feed per tonne milk equivalent)
* Pig meat from grain: 0.142 t/t (7.0 t DM feed per tonne meat)
* Chicken meat from grain: 0.348 t/t (2.9 t DM feed per tonne meat)

This structure allows modeling different production systems for the same product (grass-fed vs. grain-finished beef, pasture vs. intensive dairy, etc.).

Regional Feed Energy Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feed requirements vary significantly by region due to differences in production systems, genetics, and environmental conditions. Wirsenius (2000) [#wirsenius2000]_ provides estimated feed energy requirements per unit of commodity output:

.. table:: Feed energy requirements per unit of animal product output (Wirsenius 2000, Table 3.9)
   :widths: auto

   +-------------------------------+--------+----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+
   | Commodity                     | Unit   | East     | East     | Latin     | North     | North     | South &   | Sub-      | West      |
   |                               |        | Asia     | Europe   | America   | Africa &  | America   | Central   | Saharan   | Europe    |
   |                               |        |          |          | & Carib.  | W. Asia   | & Oc.     | Asia      | Africa    |           |
   +===============================+========+==========+==========+===========+===========+===========+===========+===========+===========+
   | Cattle milk & cow carcass     | NE_l   | 8.2      | 8.2      | 11        | 12        | 5.3       | 11        | 23        | 5.6       |
   | (MJ per kg whole milk &       | NE_m   | 2.3      | 1.3      | 1.9       | 2.0       | 1.1       | 2.5       | 5.4       | 1.3       |
   | carcass as-is)                | NE_g   | 0.46     | 0.45     | 0.30      | 0.32      | 0.50      | 0.32      | 0.70      | 0.52      |
   +-------------------------------+--------+----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+
   | Dairy bulls & heifers carcass | NE_m   | 187      | 47       | 143       | 130       | 53        | 344       | 211       | 41        |
   | (MJ per kg carcass as-is)     | NE_g   | 22       | 14       | 24        | 21        | 16        | 19        | 20        | 16        |
   +-------------------------------+--------+----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+
   | Beef carcass                  | NE_m   | 288      | 141      | 236       | 262       | 109       | 479       | 352       | 103       |
   | (MJ per kg carcass as-is)     | NE_g   | 25       | 19       | 28        | 23        | 23        | 20        | 21        | 23        |
   +-------------------------------+--------+----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+
   | Pig carcass                   | ME     | 86       | 84       | 131       | 86        | 65        | 115       | 123       | 64        |
   | (MJ per kg carcass-side       |        |          |          |           |           |           |           |           |           |
   | as-is)                        |        |          |          |           |           |           |           |           |           |
   +-------------------------------+--------+----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+
   | Eggs & hen carcass            | ME     | 43       | 42       | 39        | 43        | 32        | 53        | 56        | 30        |
   | (MJ per kg whole egg &        |        |          |          |           |           |           |           |           |           |
   | carcass as-is)                |        |          |          |           |           |           |           |           |           |
   +-------------------------------+--------+----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+
   | Meat-type chicken carcass     | ME     | 60       | 56       | 51        | 61        | 42        | 72        | 77        | 38        |
   | (MJ per kg eviscerated        |        |          |          |           |           |           |           |           |           |
   | carcass as-is)                |        |          |          |           |           |           |           |           |           |
   +-------------------------------+--------+----------+----------+-----------+-----------+-----------+-----------+-----------+-----------+

**Energy types**:
  * **NE_l**: Net energy for lactation (dairy production)
  * **NE_m**: Net energy for maintenance (basic metabolism)
  * **NE_g**: Net energy for growth (body mass gain)
  * **ME**: Metabolizable energy (for monogastrics)

**Notes**:
  * Values calculated from productivity estimates in Wirsenius (2000) Table 3.8
  * Regional variation reflects differences in production systems, breed genetics, climate, and management practices
  * Sub-Saharan Africa shows significantly higher requirements due to less intensive production systems
  * North America and Western Europe have lowest requirements, reflecting highly optimized industrial systems

.. [#wirsenius2000] Wirsenius, S. (2000). *Human Use of Land and Organic Materials: Modeling the Turnover of Biomass in the Global Food System*. Chalmers University of Technology and Göteborg University, Sweden. ISBN 91-7197-886-0. https://publications.lib.chalmers.se/records/fulltext/827.pdf

Model Implementation
--------------------

In ``workflow/scripts/build_model.py``, livestock production is represented as multi-bus links:

Grazing Links
~~~~~~~~~~~~~

**Inputs**:
  * ``bus0``: Grassland (land bus for region/class)

**Outputs**:
  * ``bus1``: Ruminant grassland feed pool (``feed_ruminant_grassland``)
  * ``bus2``: CO₂ emissions from land-use change (if configured)

**Efficiency**: Grassland yield (t DM/ha)

Grassland is routed to its own dedicated ``feed_ruminant_grassland`` pool. This separate category enables special handling of nitrogen: manure from grazing animals is deposited on pasture (not collected), so it produces N₂O emissions but does not contribute to the collected manure fertilizer pool available for cropland.

Crop Residue Feed Supply
~~~~~~~~~~~~~~~~~~~~~~~~

Crop residues (e.g., straw, stover, pulse haulms) are now generated explicitly using the new Snakemake rule ``build_crop_residue_yields``:

* **Configuration**: Select residue crops via ``animal_products.residue_crops`` in ``config/default.yaml``. Only crops present in ``config.crops`` are processed.
* **Data sources**:
  - GLEAM Supplement S1 Table S.3.1 (slope/intercept) and Tables 3.3 / 3.6 (FUE factors)
  - GLEAM feed codes → model mapping in ``data/gleam_feed_mapping.csv``
* **Outputs**: Per-crop CSVs at ``processing/{name}/crop_residue_yields/{crop}.csv`` with net dry-matter residue yields (t/ha) by region, resource class, and water supply.
* **Integration**: ``build_model`` reads all residue CSVs, adds ``residue_{feed_item}_{country}`` buses, and attaches them as additional outputs on crop production links. Residues flow through the same feed supply logic as crops/foods and enter the appropriate feed pools.

Feed Supply Links
~~~~~~~~~~~~~~~~~

The ``add_feed_supply_links()`` function creates links from crops, crop residues, and food byproducts to the nine feed pools:

**Item-to-Feed-Pool Links**:
  * **Inputs**: Crop, residue, or food byproduct buses (``bus0``)
  * **Outputs**: One of nine feed pool buses (``bus1``)
    - Ruminants: ``feed_ruminant_roughage``, ``feed_ruminant_forage``, ``feed_ruminant_grassland``, ``feed_ruminant_grain``, ``feed_ruminant_protein``
    - Monogastrics: ``feed_monogastric_low_quality``, ``feed_monogastric_grain``, ``feed_monogastric_energy``, ``feed_monogastric_protein``
  * **Efficiency**: Category-specific digestibility (from ``processing/{name}/ruminant_feed_categories.csv`` / ``monogastric_feed_categories.csv``)
  * **Routing**: Each feed item is mapped via ``processing/{name}/ruminant_feed_mapping.csv`` and ``processing/{name}/monogastric_feed_mapping.csv`` (generated by ``categorize_feeds.py``) to the relevant pool(s)
  * Crops compete between human consumption, food processing, and animal feed use; residues and byproducts are exclusive to feed use

**Example flow**:
  * Wheat grain → ``feed_ruminant_grain`` + ``feed_monogastric_grain`` (digestibility from GLEAM)
  * Wheat straw (residue) → ``feed_ruminant_roughage`` (low digestibility)
  * Wheat bran (byproduct) → ``feed_ruminant_grain`` + ``feed_monogastric_low_quality``

Feed-to-Animal-Product Links
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``add_feed_to_animal_product_links()`` function converts feed pools to animal products with CH₄ emissions:

**Feed-Pool-to-Product Links**:
  * **Inputs**: Feed pool bus (``bus0``, e.g., ``feed_ruminant_forage``)
  * **Outputs**:

    * Animal product bus (``bus1``, e.g., ``food_cattle_meat``)
    * CH₄ emissions bus (``bus2``) - all animal products

  * **Efficiency**: Feed conversion ratio (tonnes product per tonne feed DM)
  * **CH₄ calculation**: Combines enteric fermentation (ruminants) and manure management (all animals)

    .. math::

       \text{CH}_4\text{/t feed} = \text{MY}_\text{enteric} + \text{MY}_\text{manure}

    where methane yields (MY) are in kg CH₄ per kg dry matter intake.

**Example**: Grass-fed beef from forage feed with enteric MY 23.3 g/kg and manure MY 2.2 g/kg:
  * Total CH₄ = 23.3 + 2.2 = 25.5 g CH₄ per kg feed DM
  * For 1 tonne feed → 0.0255 t CH₄ emissions

See :ref:`livestock-emissions` for detailed methodology and data sources.

.. _livestock-emissions:

Emissions from Livestock
-------------------------

Livestock production generates significant greenhouse gas emissions from two primary sources:

* **Enteric fermentation (CH₄)**: Ruminants produce methane through digestive fermentation
* **Manure management (CH₄, N₂O)**: All livestock produce emissions from manure storage and handling

For detailed methodology, data sources, and IPCC calculations, see :doc:`environment` (sections on :ref:`enteric-fermentation` and :ref:`manure-management`).

Enteric Fermentation (CH₄)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ruminants (cattle, sheep) produce methane through digestive fermentation. The model uses IPCC Tier 2 methodology based on methane yields (MY) per unit dry matter intake (DMI).

Summary
^^^^^^^

* Enteric fermentation produces CH₄ in ruminants during digestion
* Methane yield (MY) varies by feed quality (roughage > forage > grain > protein)
* Model uses IPCC Tier 2 methodology with feed-specific emission factors
* See :ref:`enteric-fermentation` for full details

Data Sources
^^^^^^^^^^^^

* ``data/ipcc_enteric_methane_yields.csv``: IPCC methane yields by feed category
* ``processing/{name}/ruminant_feed_categories.csv``: Feed categories with MY values (generated from GLEAM 3.0 data)

Manure Management (CH₄, N₂O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All livestock produce emissions from manure storage, handling, and application:

* **CH₄**: From anaerobic decomposition (especially liquid systems like lagoons)
* **N₂O**: From nitrogen in manure (direct and indirect emissions)

Manure CH₄ emissions are calculated for all animal products (ruminants and monogastrics) and combined with enteric emissions in the model. See :ref:`manure-management` for full methodology.

Configuration Parameters
------------------------

.. literalinclude:: ../config/default.yaml
   :language: yaml
   :start-after: # --- section: animal_products ---
   :end-before: # --- section: food_groups ---

Disabling grazing (``enabled: false``) forces all animal products to come from feed-based systems or imports, useful for exploring intensification scenarios.

Workflow Rules
--------------

**build_grassland_yields**
  * **Input**: ISIMIP grassland yield NetCDF, resource classes, regions
  * **Output**: ``processing/{name}/grassland_yields.csv``
  * **Script**: ``workflow/scripts/build_grassland_yields.py``

Livestock production is then integrated into the ``build_model`` rule using the grassland yields and feed conversion CSVs.
