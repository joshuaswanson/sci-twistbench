#!/usr/bin/env python3
"""Build Sci-TwistBench datasets:
1. data/scenarios.json: 12 contamination-free quantitative anomaly dossiers across 4 scientific domains.
2. data/contamination_pairs.json: 4 paired historical-vs-isomorphic contamination control cases (8 dossiers).
3. data/calibration.json: 16 hand-crafted 4-quadrant calibration hypotheses for validating LLM judges.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


SCENARIOS = [
    # ==================== CLINICAL & BIOMEDICAL ====================
    {
        "id": "clin_01",
        "domain": "Clinical & Biomedical",
        "title": "The Bimodal Center Efficacy Paradox in a Phase-III Lipid Trial",
        "dossier": (
            "A multi-center, double-blind, placebo-controlled Phase-III clinical trial (N = 1,840 across 14 academic hospitals) "
            "evaluated KVX-402, an oral selective hepatic enzyme inhibitor designed to lower serum LDL cholesterol in patients with "
            "refractory hyperlipidemia. Under the prevailing pharmacokinetic paradigm, LDL reduction should scale monotonically with "
            "24-hour steady-state systemic plasma exposure (AUC) of the parent compound, independent of treatment center logistics. "
            "Across the pooled cohort, KVX-402 produced only a modest mean LDL reduction of -9.4% relative to placebo (p = 0.04).\n\n"
            "Pre-specified site-level auditing uncovered a striking bimodal divergence that completely explains the weak pooled effect: "
            "at 5 hospitals (Group A, n = 660), KVX-402 achieved a dramatic -34.2% LDL reduction (p < 1e-9), whereas at the other 9 "
            "hospitals (Group B, n = 1,180), treated patients exhibited a slight numerical worsening of +4.1% (p = 0.08). Blinded central "
            "LC-MS/MS assays verified identical active-ingredient purity, tablet stability, and 24-hour systemic plasma AUC of KVX-402 "
            "across all 14 hospitals (mean AUC ratio Group A/B = 1.02, p = 0.71). Baseline patient genetics (APOE and PCSK9 variants), "
            "age, BMI, statin background, and standardized 7-day dietary logs were statistically indistinguishable between Group A and "
            "Group B (all p > 0.40).\n\n"
            "To test whether unmeasured patient-level confounders drove the split, 64 non-responding patients from Group B hospitals "
            "were transferred into a 6-week open-label crossover protocol managed at Group A hospitals using the exact same drug lot "
            "numbers; their LDL dropped by a mean of -31.8% (p < 1e-8). Crucially, placebo-arm patients showed virtually zero LDL "
            "change in both Group A (-0.3% +/- 1.1%) and Group B (+0.2% +/- 1.2%), ruling out generic hospital care or diet effects."
        ),
        "prevailing_paradigm": (
            "Drug efficacy is determined solely by steady-state 24-hour systemic plasma exposure (AUC) of the parent compound "
            "and patient baseline genetics/diet."
        ),
        "implicit_assumptions": [
            "Systemic 24-hour plasma AUC of the parent molecule fully captures pharmacologically active target engagement.",
            "Time-of-day dosing schedule or fasting/prandial timing relative to circadian hepatic rhythms is irrelevant when 24h AUC is constant.",
            "Hospital-specific clinical administration workflows (e.g., co-administered routine drugs, meal timing, phlebotomy timing) do not interact with KVX-402.",
            "The parent drug itself—rather than a site-dependent metabolic/microbiome biotransformation product—is the sole active effector."
        ],
        "empirical_constraints": [
            "E1: Blinded LC-MS/MS confirms identical pill purity and 24-hour systemic plasma AUC of parent KVX-402 across Group A and Group B hospitals (p = 0.71).",
            "E2: Baseline patient genetics (APOE/PCSK9), age, BMI, statin background, and dietary logs are matched across Group A and Group B (all p > 0.40).",
            "E3: Transferring 64 Group B non-responders to Group A hospitals using the exact same drug lot numbers restores -31.8% LDL reduction within 6 weeks.",
            "E4: Placebo arms show zero LDL change in both Group A (-0.3%) and Group B (+0.2%), proving the center factor interacts specifically with KVX-402."
        ],
    },
    {
        "id": "clin_02",
        "domain": "Clinical & Biomedical",
        "title": "The Rapid-Clearance Mortality Inversion in Septic Shock",
        "dossier": (
            "A stepped-wedge randomized quality-improvement study across 18 intensive care units (N = 6,200 adult patients meeting "
            "Sepsis-3 criteria for suspected Gram-negative septic shock) evaluated an aggressive rapid-escalation broad-spectrum "
            "beta-lactam infusion protocol that cut median time-to-first-dose from 142 minutes to 28 minutes. Standard critical-care "
            "theory holds that bacterial burden drives progressive organ failure, so faster bactericidal clearance should monotonically "
            "reduce mortality across all severity strata.\n\n"
            "Microbiologically, the rapid-escalation protocol succeeded overwhelmingly: quantitative blood cultures drawn at 6 hours "
            "showed sterile bacterial clearance in 79% of rapid-arm patients versus 41% in standard-care patients (p < 1e-12). Yet "
            "28-day all-cause mortality significantly increased in the rapid-escalation arm, rising from 21.4% to 28.9% (p < 1e-6). "
            "Direct drug-toxicity surveillance found no increase in acute kidney injury, hepatotoxicity, or anaphylaxis (p = 0.63), "
            "and protocol-locked fluid resuscitation volumes, vasopressor titration thresholds, and mechanical ventilation parameters "
            "were identical between arms (all p > 0.50).\n\n"
            "Stratified analysis revealed a sharp sign reversal across baseline physiological phenotypes: among patients presenting "
            "with severe hyperlactatemia (> 4.0 mmol/L), rapid escalation modestly reduced mortality (-3.8 percentage points, p = 0.03). "
            "The entire excess mortality (+13.4 percentage points, p < 1e-9) was concentrated in patients who presented with initially "
            "low serum lactate (< 2.5 mmol/L) combined with high initial quantitative blood bacterial load (> 10^4 CFU/mL)."
        ),
        "prevailing_paradigm": (
            "In septic shock, maximizing the speed of bactericidal clearance monotonically improves host survival."
        ),
        "implicit_assumptions": [
            "Killing circulating bacteria rapidly reduces net host physiological injury rather than precipitating acute secondary harm.",
            "The host immune/metabolic state at presentation (low vs. high lactate) only reflects disease severity, not a qualitatively different response regime to rapid bacterial lysis.",
            "Protocol-locked resuscitation parameters imply identical downstream clinical surveillance and management decisions.",
            "The therapeutic target is purely extracellular pathogen load without protective commensal or counter-regulatory coupling."
        ],
        "empirical_constraints": [
            "E1: 6-hour blood cultures confirm 79% sterile clearance in the rapid arm vs. 41% in the standard arm (ruling out antibiotic resistance or under-dosing).",
            "E2: Direct antibiotic organ toxicities (nephrotoxicity, hepatotoxicity, anaphylaxis) did not differ between arms (p = 0.63).",
            "E3: Excess mortality (+13.4 percentage points) occurred exclusively in low-lactate (<2.5 mmol/L), high-bacterial-load patients, whereas high-lactate (>4.0 mmol/L) patients benefited (-3.8 percentage points).",
            "E4: Fluid resuscitation volumes, vasopressor thresholds, and ventilator parameters were protocol-locked and identical between arms (p > 0.50)."
        ],
    },
    {
        "id": "clin_03",
        "domain": "Clinical & Biomedical",
        "title": "The Non-Monotonic Motor Recovery Trough After Focal Stroke",
        "dossier": (
            "A prospective longitudinal neurorehabilitation study (N = 410 first-time unilateral middle cerebral artery ischemic stroke "
            "patients, independently replicated in a second cohort of N = 230) quantified upper-limb motor recovery using the Fugl-Meyer "
            "Assessment (FMA-UE, 0-66 scale) over 12 months alongside baseline diffusion tensor imaging (DTI) of the ipsilesional "
            "corticospinal tract (CST). Under standard proportional-recovery models, motor improvement should decrease monotonically "
            "with structural CST damage and scale positively with rehabilitation dose.\n\n"
            "Instead, recovery exhibited a deep, reproducible non-monotonic trough at intermediate structural damage. Patients with mild "
            "CST loss (< 20% tract disruption) recovered near-completely (mean +38.4 FMA points). Surprisingly, patients with intermediate "
            "CST loss (35-55% tract disruption) recovered far worse (mean +11.2 points) than patients with severe CST loss (75-90% tract "
            "disruption), who achieved a mean gain of +27.8 points (p < 1e-7 comparing severe vs. intermediate). High-resolution DTI "
            "confirmed strictly monotonic structural white-matter loss with zero hidden CST sparing in the 75-90% group.\n\n"
            "Total physiotherapy hours, therapist blinding, robotic-assistance exposure, and concurrent pharmacotherapy were tightly "
            "matched across all damage strata (p = 0.58). Task-based fMRI at 6 months revealed a qualitative neural divergence: "
            "intermediate-damage patients (35-55%) exhibited persistent, elevated activation of ipsilesional primary motor cortex (M1), "
            "whereas severe-damage patients (75-90%) suppressed ipsilesional M1 and strongly recruited contralesional premotor and "
            "brainstem reticulospinal networks."
        ),
        "prevailing_paradigm": (
            "Residual ipsilesional corticospinal tract integrity is monotonically beneficial for post-stroke motor recovery."
        ),
        "implicit_assumptions": [
            "Preserved structural fibers in a damaged pathway always contribute net positive signal rather than interfering or blocking alternative neural pathways.",
            "Motor rehabilitation engages a single continuous recovery mechanism rather than competing discrete neural/behavioral attractors.",
            "The Fugl-Meyer score measures a single biological recovery process rather than conflating restitution and gross compensatory strategies.",
            "Enrolled rehab cohorts across lesion severity strata are free of differential survival or premorbid reserve selection."
        ],
        "empirical_constraints": [
            "E1: High-resolution DTI verifies strictly monotonic structural CST white-matter loss across <20%, 35-55%, and 75-90% strata with no hidden CST sparing.",
            "E2: Physiotherapy hours, therapist blinding, robotic therapy, and medications are statistically matched across strata (p = 0.58).",
            "E3: Mild damage (<20%) recovers near-completely (+38.4), intermediate (35-55%) hits a deep trough (+11.2), and severe (75-90%) rebounds strongly (+27.8).",
            "E4: 6-month fMRI shows persistent ipsilesional M1 activation in the 35-55% group versus contralesional premotor/reticulospinal recruitment in the 75-90% group."
        ],
    },
    # ==================== PHYSICS & MATERIALS SCIENCE ====================
    {
        "id": "phys_01",
        "domain": "Physics & Materials",
        "title": "Thickness-Inverted Thermal Conductivity in Epitaxial Oxide Thin Films",
        "dossier": (
            "A materials physics collaboration measured the cross-plane and in-plane thermal conductivity (kappa_eff) of epitaxial "
            "layered perovskite-oxide thin films (Sr2TiO4 grown on single-crystal MgO substrates) across thicknesses d ranging from "
            "20 nm to 800 nm at 300 K using time-domain thermoreflectance (TDTR). Standard kinetic theory of dielectric solids "
            "predicts that phonon-boundary scattering should reduce kappa_eff monotonically below the bulk single-crystal value "
            "(5.6 W/m*K) as film thickness decreases into the Casimir regime.\n\n"
            "While films between 200 nm and 800 nm behaved normally (kappa_eff dropping from 5.3 to 4.4 W/m*K), films thinner than "
            "120 nm exhibited a dramatic inversion: at d = 35 nm, kappa_eff surged to 18.4 W/m*K, more than triple the bulk single-crystal "
            "limit. Cross-sectional STEM and X-ray diffraction confirmed exact stoichiometry, single-phase crystallinity, and electrical "
            "resistivity exceeding 10^11 Ohm*cm at all thicknesses, ruling out metallic inclusions or electronic heat conduction. "
            "Independent contact 3-omega electrical thermometry reproduced the 18.4 W/m*K value within 4%, ruling out optical "
            "penetration-depth artifacts specific to TDTR.\n\n"
            "Substrate-exchange experiments isolated a sharp boundary condition: growing the identical 35 nm Sr2TiO4 film on compressive "
            "LaAlO3 eliminated the enhancement (kappa_eff = 4.1 W/m*K), and mechanically exfoliating and transferring the 35 nm film off "
            "MgO onto amorphous SiO2 caused kappa_eff to collapse immediately from 18.4 to 4.3 W/m*K. Moreover, when the intact 35 nm "
            "film on MgO was cooled to 80 K, the anomalous enhancement vanished completely and kappa_eff reverted to normal Casimir scaling."
        ),
        "prevailing_paradigm": (
            "In insulating crystalline thin films, reducing thickness increases diffuse boundary scattering and strictly lowers "
            "thermal conductivity below the bulk crystal limit."
        ),
        "implicit_assumptions": [
            "Heat transport in the dielectric thin film occurs purely via incoherent diffusive bulk-like phonons independent of interfacial strain or electromagnetic coupling.",
            "The MgO substrate acts as a passive mechanical support whose thermal and lattice properties remain decoupled from the film's effective conductivity.",
            "Anharmonic phonon-phonon scattering rates are fixed intrinsic material constants unaffected by coherent epitaxial strain.",
            "Standard 1D thermal inversion models correctly partition heat flux between the ultrathin film and the near-surface substrate region."
        ],
        "empirical_constraints": [
            "E1: STEM/XRD confirm pure stoichiometry and electrical resistivity > 10^11 Ohm*cm at all thicknesses, ruling out electronic conduction or metallic phases.",
            "E2: Both optical TDTR and contact 3-omega thermometry agree within 4%, ruling out TDTR optical transparency artifacts.",
            "E3: The 35 nm enhancement (18.4 W/m*K) occurs on tensile MgO, vanishes on compressive LaAlO3 (4.1 W/m*K), and vanishes immediately when exfoliated off MgO onto SiO2 (4.3 W/m*K).",
            "E4: Cooling the intact 35 nm film on MgO to 80 K completely eliminates the enhancement, restoring normal Casimir scaling."
        ],
    },
    {
        "id": "phys_02",
        "domain": "Physics & Materials",
        "title": "The Phase-Shifted Annual Drift Between Distant Optical Lattice Clocks",
        "dossier": (
            "Over 24 consecutive months, two metrology institutes continuously compared two identically constructed 87Sr optical "
            "lattice clocks linked via a 420 km phase-stabilized optical fiber: Clock A located in a coastal sea-level laboratory and "
            "Clock B located in an alpine laboratory at 1,450 m elevation. After correcting for static orthometric gravitational redshift "
            "(Delta_nu/nu = g*Delta_h/c^2), solid-Earth tides, Zeeman shifts, and blackbody radiation, standard general-relativistic "
            "metrology predicts a stationary fractional frequency ratio stable within 2e-18.\n\n"
            "Instead, the fully corrected ratio nu_B / nu_A exhibited a smooth, reproducible annual oscillation with peak-to-peak "
            "amplitude 4.8e-17 (24 sigma), peaking every year in late April and reaching its minimum in late October—roughly 60 days "
            "out of phase with both local ambient air temperature extremes and Earth's orbital perihelion/aphelion. Bidirectional "
            "Doppler-cancellation diagnostics verified that optical-fiber link non-reciprocity remained below 1.5e-18 year-round, and "
            "laboratory HVAC held both vacuum chambers within +/- 0.05 K (bounding blackbody shifts below 2e-18).\n\n"
            "Two auxiliary instruments sharply constrained the anomaly. First, a portable 171Yb+ optical ion clock co-located at "
            "Alpine Lab B and measured locally against 87Sr Clock B showed zero annual drift (< 3e-18), ruling out any Sr-specific "
            "atomic systematic or species-dependent drift in fundamental constants. Second, a continuous superconducting gravimeter "
            "co-located at Alpine Lab B recorded a seasonal vertical gravity variation of only 40 nm/s^2—more than 50 times too small "
            "to account for a 4.8e-17 frequency shift under a standard local point-mass free-air height conversion."
        ),
        "prevailing_paradigm": (
            "Static geodetic elevation plus local gravimeter monitoring suffices to remove environmental gravitational redshift "
            "variations between distant optical clocks."
        ),
        "implicit_assumptions": [
            "Local vertical gravity acceleration g(t) measured at a single site reliably proxies gravitational potential Phi(t) across regional mass redistributions.",
            "Seasonal mass changes at coastal Lab A (which lacks a reported gravimeter) are negligible compared to alpine Lab B.",
            "Relativistic frame transformations (Sagnac, polar motion, geoid reference surface) over a 420 km fiber link are static or fully captured by standard tide models.",
            "The gravitational potential shift can only arise from vertical geometric uplift/subsidence rather than extended regional mass loading."
        ],
        "empirical_constraints": [
            "E1: A co-located 171Yb+ ion clock vs. 87Sr Clock B at Lab B shows zero annual shift (< 3e-18), ruling out Sr-specific systematics or alpha-variation.",
            "E2: Bidirectional fiber-link diagnostics bound optical link non-reciprocity below 1.5e-18 year-round.",
            "E3: Chamber temperature is stabilized to +/- 0.05 K, bounding blackbody radiation shifts below 2e-18, and the 4.8e-17 cycle peaks in late April / troughs in late October (60 days off solar/temperature extrema).",
            "E4: A co-located superconducting gravimeter at Lab B measures only 40 nm/s^2 seasonal gravity change, 50x smaller than expected from a local point-mass height-change model."
        ],
    },
    {
        "id": "phys_03",
        "domain": "Physics & Materials",
        "title": "Voltage-Selective Capacity Gain Under Open-Circuit Rest in Solid-State Batteries",
        "dossier": (
            "An electrochemical aging study cycled 240 solid-state lithium-metal coin cells (NCM811 cathode, LLZO garnet solid "
            "electrolyte, Li-metal anode) at a constant 1C rate between fixed terminal voltage cutoffs of 3.0 V and 4.3 V. Under "
            "standard battery degradation theory, discharge capacity declines monotonically with cumulative charge throughput and "
            "calendar time; inserting open-circuit rest periods should only allow slow parasitic calendar fade.\n\n"
            "Unexpectedly, cells subjected to a 12-hour open-circuit rest held at precisely 3.72 V every 25 cycles exhibited a steady "
            "capacity *increase*, delivering 114% of their own initial cycle-1 discharge capacity at cycle 500 (and 112% nominal "
            "capacity), while continuously cycled control cells faded to 78% (p < 1e-10). High-precision coulombic efficiency "
            "exceeded 99.94% per cycle, and post-mortem ICP-OES elemental analysis verified that total lithium inventory inside the "
            "hermetically sealed cells was strictly conserved.\n\n"
            "Control protocols ruled out generic resting effects: inserting identical 12-hour rests at 4.25 V (fully charged) or "
            "3.10 V (discharged) accelerated degradation (71% and 76% retention at cycle 500, respectively). During the 3.72 V rest, "
            "isothermal micro-calorimetry detected a small, reproducible endothermic heat absorption accompanied by a slow +18 mV "
            "open-circuit voltage relaxation. Operando X-ray tomography found no dendrites or macroscopic void collapse, whereas "
            "electrochemical impedance spectroscopy (EIS) revealed a 62% drop in interfacial charge-transfer resistance exclusively "
            "in the 3.72 V-rested cells."
        ),
        "prevailing_paradigm": (
            "Cycle-1 capacity represents the ceiling of accessible reversible lithium, and open-circuit resting can only cause "
            "monotonic calendar degradation."
        ),
        "implicit_assumptions": [
            "Cycle-1 measured capacity at 1C between 3.0 V and 4.3 V equals the true thermodynamic lithium inventory rather than a kinetically truncated fraction.",
            "Interfacial resistance and microstructural strain states degrade irreversibly rather than undergoing voltage-specific annealing or phase equilibration.",
            "Fixed terminal voltage cutoffs (3.0-4.3 V) correspond to identical internal state-of-charge windows regardless of interfacial overpotential.",
            "Open-circuit rest at intermediate chemical potential is thermodynamically inert rather than driving endothermic structural reordering."
        ],
        "empirical_constraints": [
            "E1: Coulombic efficiency exceeds 99.94% and post-mortem ICP-OES confirms strictly conserved total lithium inventory (no external lithium source).",
            "E2: Identical 12-hour rests at 4.25 V (71%) or 3.10 V (76%) worsen degradation, proving the capacity gain is sharply specific to resting at 3.72 V.",
            "E3: Isothermal calorimetry during the 3.72 V rest shows a reproducible endothermic heat signature and +18 mV OCV relaxation.",
            "E4: Operando X-ray tomography rules out dendrites/macro-void collapse, while EIS shows a 62% drop in interfacial charge-transfer resistance exclusively in 3.72 V-rested cells."
        ],
    },
    # ==================== ECOLOGY & EARTH SYSTEMS ====================
    {
        "id": "ecol_01",
        "domain": "Ecology & Earth Systems",
        "title": "Predator-Exclusion Collapse of Herbivore Biomass in Coastal Kelp Reefs",
        "dossier": (
            "A 5-year controlled marine-ecology field experiment across 36 coastal rocky-reef plots (500 m^2 each) tested classic "
            "top-down trophic cascade theory by excluding apex predatory crabs and wrasses using large-mesh cages to allow herbivorous "
            "purple sea urchins to proliferate. Standard trophic theory predicts that predator removal should sharply increase urchin "
            "biomass and intensify grazing until kelp forests collapse into urchin barrens.\n\n"
            "During months 1 through 4, exclusion plots followed textbook predictions: juvenile urchin settlement and density reached "
            "2.1 times that of unfenced controls (p < 1e-5). Beginning in month 7, however, urchin populations inside exclusion plots "
            "crashed catastrophically; by month 18, total urchin biomass in predator-exclusion plots was 68% *lower* than in control "
            "plots (p < 1e-8), and canopy kelp cover expanded by +41% relative to controls—the exact inverse of a standard trophic "
            "cascade.\n\n"
            "Continuous underwater video and cage-integrity inspections confirmed zero apex predator intrusion or human harvesting, "
            "and in-situ loggers recorded identical water temperature, dissolved oxygen, salinity, and dissolved inorganic nitrogen "
            "fluxes across exclusion and control plots (all p > 0.65). Crucially, dead urchin tests collected during the months 7-14 "
            "die-off exhibited completely intact spines with zero mechanical bite marks, while surviving urchins in exclusion plots "
            "displayed severely elevated cellular heat-shock/metabolic stress markers and 55% smaller gonad indices despite abundant "
            "standing kelp."
        ),
        "prevailing_paradigm": (
            "Removing apex predators strictly releases herbivore populations from mortality control, increasing herbivore biomass "
            "and decreasing primary producer (kelp) cover."
        ),
        "implicit_assumptions": [
            "Predators act solely as a negative mortality sink on prey rather than performing stabilizing roles (e.g., culling diseased individuals or regulating prey aggregation).",
            "Standing kelp biomass is uniformly palatable and nutritionally constant regardless of early grazing intensity.",
            "Exclusion cages that block apex predators leave all smaller cryptic trophic levels and microbial interactions unchanged.",
            "Herbivore foraging behavior and spatial density distribution remain invariant in the absence of predation risk cues."
        ],
        "empirical_constraints": [
            "E1: Verified mesh integrity and underwater video confirm zero apex predator intrusion or human poaching.",
            "E2: Continuous loggers verify identical temperature, dissolved oxygen, salinity, and nutrient influx between exclusion and control plots (all p > 0.65).",
            "E3: Juvenile urchin settlement surged 2.1x during months 1-4 before the sudden crash began in month 7, culminating in -68% biomass and +41% kelp cover.",
            "E4: Dead urchin tests had intact spines with zero bite marks, while surviving exclusion-plot urchins showed high metabolic stress and stunted gonads despite abundant standing kelp."
        ],
    },
    {
        "id": "ecol_02",
        "domain": "Ecology & Earth Systems",
        "title": "Boreal Carbon-Sink Surge Amid Severe Multi-Year Drought",
        "dossier": (
            "A 15-year eddy-covariance flux-tower network spanning 12 mature black-spruce sites in the Canadian boreal forest "
            "continuously monitored net ecosystem CO2 exchange (NEE). Under standard terrestrial carbon-cycle models, severe multi-year "
            "drought induces stomatal closure, reduces gross primary productivity (GPP), elevates thermal respiration stress, and "
            "weakens or reverses the boreal carbon sink into a net carbon source.\n\n"
            "During an unprecedented 3-year regional drought (cumulative precipitation -44%, root-zone soil moisture -2.8 sigma), "
            "9 of the 12 towers recorded a striking +38% *increase* in net annual carbon uptake (p < 1e-5). This occurred even though "
            "independent dendrochronology (tree-ring cores), litterfall traps, and satellite NDVI all confirmed that aboveground woody "
            "growth and canopy greenness declined by 18-25% across the exact same 9 sites. Every tower passed independent sonic-"
            "anemometer calibrations, closed-path infrared gas analyzer zero/span checks, and energy-balance closure audits (> 86%).\n\n"
            "Ancillary measurements highlighted a sharp geological split: the 9 surge sites were all underlain by deep, thick organic "
            "peat and feather-moss layers, where automated soil-chamber assays recorded a 61% collapse in heterotrophic soil respiration "
            "(R_h) while daytime canopy GPP fell by only 14% and nighttime eddy-covariance respiration dropped sharply. In contrast, "
            "the remaining 3 sites sat on well-drained upland mineral soils and showed the textbook drought response: a -29% decline "
            "in net carbon uptake."
        ),
        "prevailing_paradigm": (
            "Severe drought reduces net ecosystem carbon uptake in boreal forests in proportion to declines in canopy greenness "
            "and aboveground tree growth."
        ),
        "implicit_assumptions": [
            "Net ecosystem carbon balance (NEE = GPP - R_eco) is dominated by drought sensitivity of photosynthesis rather than asymmetric sensitivity of microbial decomposition.",
            "Aboveground woody increments and NDVI capture total ecosystem carbon allocation, ignoring belowground root/mycorrhizal shifts.",
            "Vertical eddy-covariance tower measurements over dry, porous organic peat accurately close the 3D ecosystem carbon budget without lateral subterranean CO2 drainage.",
            "Organic peat and upland mineral soils respond along the same monotonic moisture-respiration curve."
        ],
        "empirical_constraints": [
            "E1: All 12 towers passed anemometer calibration, IRGA zero/span checks, and energy-balance closure audits (>86%), ruling out sensor drift.",
            "E2: Tree rings, litterfall traps, and satellite NDVI confirm aboveground growth and canopy greenness fell 18-25% at the 9 surge sites.",
            "E3: The +38% NEE sink surge occurred exclusively at the 9 deep organic-peat sites (where soil-chamber R_h plunged 61%), whereas the 3 mineral-soil sites declined by -29%.",
            "E4: Nighttime eddy-covariance respiration dropped sharply at the 9 peat sites while daytime GPP fell by only 14%."
        ],
    },
    {
        "id": "ecol_03",
        "domain": "Ecology & Earth Systems",
        "title": "Daytime Urban Heat-Island Cooling Inversion During Extreme Heatwaves",
        "dossier": (
            "A 10-summer urban climate study deployed 180 aspirated meteorological stations paired with Landsat thermal-infrared "
            "retrievals across 6 semi-arid metropolitan basins to quantify the urban-rural temperature differential (Delta_T = T_urban - T_rural). "
            "Classic urban climatology holds that impervious asphalt/concrete surfaces, reduced vegetation, and anthropogenic waste heat "
            "cause cities to stay warmer than surrounding rural land, with heatwaves amplifying urban heat island (UHI) intensity.\n\n"
            "Under moderate summer warmth (background rural highs of 30-36 C), all 6 cities exhibited the textbook positive daytime UHI "
            "of +3.1 C. However, whenever synoptic extreme heatwaves pushed rural temperatures above 41 C, the daytime differential "
            "sharply inverted: urban cores became 2.4 C to 3.8 C *cooler* than surrounding rural scrubland (p < 1e-9). Both 2-meter "
            "aspirated air thermometers and satellite land-surface temperature retrievals captured the daytime inversion, ruling out "
            "radiation-shielding artifacts.\n\n"
            "Spatial and diurnal breakdowns imposed tight physical constraints. On the exact same >41 C heatwave days, the inversion "
            "reversed completely after sunset, with urban cores remaining +4.2 C warmer than rural stations overnight. Furthermore, "
            "the daytime cooling inversion was strongest in dense high-rise downtown cores (-3.8 C) and irrigated suburban residential "
            "tracts (-2.9 C), completely absent in flat unshaded industrial parking districts (+0.4 C), and occurred under synoptic "
            "conditions with uniform wind speed, cloud cover, and regional humidity across urban and rural stations (all p > 0.50)."
        ),
        "prevailing_paradigm": (
            "Urban impervious surfaces and anthropogenic heat monotonically amplify daytime urban warmth relative to rural "
            "surroundings during extreme heatwaves."
        ),
        "implicit_assumptions": [
            "Rural semi-arid landscapes maintain higher evaporative cooling than urban areas across all temperature regimes.",
            "Urban surface geometry (3D canyon height) acts purely as a heat trap rather than shifting the diurnal phase of peak radiative absorption.",
            "The urban-rural differential is driven primarily by static land-cover albedo rather than nonlinear physiological thresholds (stomatal shutdown) or human water/irrigation responses.",
            "Urban boundary-layer depth and vertical heat-rejection geometry map uniformly onto 2-meter street-level temperatures."
        ],
        "empirical_constraints": [
            "E1: Both 2m aspirated air thermometers and satellite skin-temperature retrievals confirm the -2.4 to -3.8 C daytime inversion above 41 C.",
            "E2: On the exact same >41 C heatwave days, nighttime UHI flips back to strongly positive (+4.2 C warmer in urban cores).",
            "E3: Daytime cooling is strongest in dense high-rise cores (-3.8 C) and irrigated suburbs (-2.9 C), but absent in flat open industrial parking districts (+0.4 C).",
            "E4: Synoptic wind speed, cloud cover, and background humidity are statistically uniform across urban and rural stations (p > 0.50)."
        ],
    },
    # ==================== ECONOMICS & BEHAVIORAL SCIENCES ====================
    {
        "id": "econ_01",
        "domain": "Economics & Behavioral",
        "title": "Market Contraction Under Algorithmic Price Transparency",
        "dossier": (
            "An online skilled-freelance labor marketplace ran a 6-month cluster-randomized experiment across 24 non-overlapping skill "
            "categories (N = 48,000 active freelancers; N = 190,000 client job postings). In 12 treatment categories, the platform "
            "deployed an AI market-rate estimator that displayed the predicted median winning bid to both clients and freelancers "
            "prior to proposal submission. Standard search-and-matching economics predicts that reducing price uncertainty should "
            "narrow bid dispersion, speed up bargaining, and increase total matched transaction volume.\n\n"
            "Bid dispersion indeed tightened by 47% in treatment categories, and freelancer time-to-submit proposals fell by 38%, "
            "ruling out interface friction or cognitive overload. Yet total completed match rates *collapsed* by 19.3% relative to "
            "untreated control categories (p < 1e-12), average winning bids *rose* by 11.8%, and top-decile freelancers (ranked by "
            "historical client ratings and repeat hires) exited the platform at 3.2 times the control attrition rate. Control categories "
            "remained completely flat throughout the 6 months, ruling out cross-market spillovers or macro shocks.\n\n"
            "Subgroup decomposition revealed a stark task-structure interaction: client posting volume and stated budgets did not drop "
            "during months 1-2 (p = 0.44), yet the match-rate decline began immediately in week 2. Crucially, the collapse was "
            "concentrated almost entirely in complex, high-variance custom projects (-31.4% match rate), whereas standardized, "
            "commoditized tasks experienced a modest *increase* in match volume (+4.2%, p = 0.01)."
        ),
        "prevailing_paradigm": (
            "Providing accurate public price benchmarks reduces market friction and monotonically increases transaction efficiency "
            "and match volume."
        ),
        "implicit_assumptions": [
            "Bid prices convey only cost information rather than serving as an endogenous quality-signaling mechanism for heterogeneous talent.",
            "Public median benchmarks act as neutral statistical summaries rather than psychological fairness anchors or tacit collusive price floors.",
            "The AI estimator trained on completed transactions provides an unbiased reference price for complex, heterogeneous projects.",
            "Lowering dispersion benefits both standardized and vertically differentiated high-skill submarkets symmetrically."
        ],
        "empirical_constraints": [
            "E1: Cluster randomization across 24 non-overlapping categories confirms zero cross-market spillover; control match rates stayed flat.",
            "E2: Client posting volume and stated budgets did not decline in months 1-2 (p = 0.44), yet match rates dropped immediately starting in week 2.",
            "E3: Match collapse occurred almost exclusively on complex, high-variance projects (-31.4%), while standardized tasks increased (+4.2%), accompanied by +11.8% winning bids and 3.2x top-decile freelancer exit.",
            "E4: Bid dispersion fell 47% and freelancer bid-submission time dropped 38%, ruling out interface friction or cognitive overload."
        ],
    },
    {
        "id": "econ_02",
        "domain": "Economics & Behavioral",
        "title": "Incentivized Accuracy Collapse on Subtle Errors in Scientific Auditing",
        "dossier": (
            "A randomized field experiment recruited N = 1,200 trained STEM crowd auditors to evaluate 36,000 quantitative scientific "
            "claims for methodological validity. Half were assigned to a flat hourly wage, while half were assigned to a strictly "
            "proper accuracy-bonus scheme paying 3x base rate when their binary verdict ('Valid' vs. 'Flawed') matched a blinded "
            "gold-standard panel of domain experts audited on a random 15% subset. Standard principal-agent theory holds that "
            "incentivizing agreement with an accurate ground-truth audit increases cognitive effort and improves error detection across "
            "all difficulty tiers.\n\n"
            "Pre-treatment calibration verified identical baseline auditing skill between arms (p = 0.81). Under the bonus scheme, "
            "auditors spent 27% *more* time per claim and wrote 34% longer written justifications, ruling out rushed clicking or bots. "
            "On straightforward consensus-clear items, accuracy rose from 84% to 93% (p < 1e-8). However, on claims containing subtle "
            "but genuine statistical errors (e.g., hidden collider bias or misapplied multiple-testing corrections wrapped in polished "
            "prose), bonus-arm accuracy *collapsed* from 71% to 43% (p < 1e-11)—substantially worse than flat-pay controls and below "
            "50% chance.\n\n"
            "Error-direction analysis showed the collapse was not random noise: on subtle-error items, bonus auditors shifted "
            "overwhelmingly toward falsely certifying flawed claims as 'Valid' (false-negative rate surged from 24% to 54%), while "
            "false-positive rates on genuinely valid claims actually declined slightly. The blinded expert panel's ground truth was "
            "identical for both arms, and auditors had zero access to panelist identities or historical panel verdicts."
        ),
        "prevailing_paradigm": (
            "Paying auditors a strictly proper bonus for matching an accurate expert ground truth aligns incentives with independent "
            "truth-seeking and improves detection of subtle errors."
        ),
        "implicit_assumptions": [
            "Auditors rewarded for matching an expert panel maximize their own first-order belief about ground truth rather than second-order beliefs about what an average panelist will certify.",
            "Longer time-on-task and longer justifications reflect deeper mathematical verification rather than surface-credibility search or rationalizing a safe default.",
            "Auditors perform the task purely with internal human cognition rather than outsourcing judgment to external heuristic or automated tools whose failure modes correlate with polished prose.",
            "Incentives do not alter subjective decision thresholds under perceived asymmetric base rates or burden-of-proof norms."
        ],
        "empirical_constraints": [
            "E1: Randomized assignment and pre-treatment calibration confirm identical baseline skill across flat-pay and bonus arms (p = 0.81).",
            "E2: Bonus auditors spent 27% more time per claim and wrote 34% longer justifications, ruling out rushing, laziness, or bots.",
            "E3: Bonus accuracy rose on clear items (84% -> 93%) but collapsed on subtle statistical errors (71% -> 43%) via a directional surge in certifying flawed claims as 'Valid' (false-negative rate 24% -> 54%).",
            "E4: Expert panel ground truth was blinded and identical across arms, with zero auditor access to panelist identities or past labels."
        ],
    },
    {
        "id": "econ_03",
        "domain": "Economics & Behavioral",
        "title": "The Commute-Duration Expansion Following Automated Express Rail Launch",
        "dossier": (
            "A 3-year metropolitan mobility study tracked a fixed longitudinal panel of N = 85,000 urban commuters via anonymized "
            "smartphone GPS telemetry before and after the launch of a 6-line automated express rail network engineered to cut "
            "point-to-point transit travel times by 45%. Standard urban transport models predict that dramatically faster rail service "
            "should shift commuters out of cars, relieve arterial highway congestion, and substantially reduce average door-to-door "
            "commute durations.\n\n"
            "Operational metrics met every engineering target: trains maintained 99.2% on-time reliability at 45% faster station-to-"
            "station speeds, rail mode share surged from 14% to 39%, arterial vehicle-kilometers fell 18%, and average highway driving "
            "speeds *increased* by 16% (ruling out Braess's paradox or induced highway gridlock). Yet by month 24, mean door-to-door "
            "commute duration across the exact same tracked commuter panel had *increased* by 11.4 minutes (+29%, p < 1e-15). Even "
            "more puzzlingly, among panel members who personally switched from private car to the faster express rail, 42% experienced "
            "*longer* total door-to-door commutes at month 24 than they had by car before launch.\n\n"
            "High-frequency temporal tracking exposed a sharp dynamic trajectory: during the first 3 months immediately following the "
            "rail opening, mean door-to-door commute time *dropped* by 7.8 minutes exactly as predicted. Between months 6 and 24, "
            "however, mean commute times climbed steadily past the pre-launch baseline, even though the panel composition remained "
            "strictly fixed (excluding newly arriving metropolitan immigrants)."
        ),
        "prevailing_paradigm": (
            "Faster point-to-point transit infrastructure reduces door-to-door commute durations for existing urban residents when "
            "highway congestion simultaneously eases."
        ),
        "implicit_assumptions": [
            "Origin-destination pairs (home and workplace locations) of existing commuters remain static after a major drop in per-kilometer travel cost.",
            "Point-to-point station speed dominates total door-to-door travel time, ignoring dynamic saturation of first-mile/last-mile feeders, transfers, or departure-time clustering.",
            "Commuters minimize travel time rather than holding a roughly fixed travel-time budget and trading speed gains for housing/wage gains.",
            "Trip structure (single-purpose commute vs. multi-stop errand chaining) remains unchanged when switching from private cars to hub-and-spoke rail."
        ],
        "empirical_constraints": [
            "E1: Express trains achieved 99.2% reliability and 45% faster station-to-station speeds; rail mode share rose from 14% to 39%.",
            "E2: Arterial vehicle-km fell 18% and average highway speeds increased 16%, ruling out induced highway gridlock or Braess's paradox.",
            "E3: Telemetry tracks the exact same fixed cohort of 85,000 commuters over 3 years (ruling out demographic inflow of distant newcomers), yet 42% of car-to-rail switchers had longer commutes at month 24.",
            "E4: Mean commute duration initially dropped -7.8 minutes in months 1-3 before climbing steadily to +11.4 minutes above baseline between months 6 and 24."
        ],
    },
]

# ==================== CONTAMINATION CONTROL PAIRS (EXP 4) ====================
CONTAMINATION_PAIRS = [
    {
        "pair_id": "contam_01",
        "concept": "Infectious etiology behind sterile-environment chemical damage (H. pylori / peptic ulcer)",
        "orig": {
            "id": "contam_01_orig",
            "variant": "historical_original",
            "domain": "Contamination Control",
            "title": "Chronic Peptic Ulcer Relapse After Acid Suppression",
            "dossier": (
                "For decades, gastroenterology held that chronic peptic ulcers were caused purely by excess gastric acid, stress, "
                "and mucosal barrier breakdown, because the highly acidic stomach lumen (pH 1.5-2.0) was assumed to be a sterile "
                "environment hostile to microbial colonization. Potent H2-receptor antagonists reliably suppressed gastric acid "
                "secretion and healed over 85% of duodenal ulcers on endoscopy within 8 weeks.\n\n"
                "However, longitudinal follow-up across multiple hospital cohorts revealed a striking paradox: within 12 months of "
                "stopping acid-suppressing drugs, over 75% of patients suffered complete ulcer relapse, even when lifestyle, diet, "
                "and psychological stress markers remained stable. By contrast, a small clinic testing an older bismuth subcitrate "
                "regimen combined with short-course metronidazole achieved similar initial healing but recorded a 12-month relapse "
                "rate below 15%, despite far weaker acid suppression.\n\n"
                "Histological biopsies of ulcer margins routinely noted curved, spiral structures beneath the gastric mucus layer, "
                "which standard pathology dismissed as incidental post-biopsy contaminants or inert debris because conventional "
                "48-hour aerobic plate cultures grew nothing."
            ),
            "empirical_constraints": [
                "E1: Acid-suppressing H2 blockers heal >85% of ulcers acutely yet >75% relapse within 12 months after cessation.",
                "E2: Bismuth + metronidazole suppresses acid far less yet cuts 12-month relapse below 15%.",
                "E3: Stomach lumen pH is 1.5-2.0 and standard 48-hour aerobic cultures fail to grow organisms.",
                "E4: Biopsies repeatedly show curved spiral structures beneath the mucus layer at ulcer margins."
            ],
            "canonical_mechanism": "Slow-growing acid-tolerant mucosal bacterium (H. pylori) surviving under the mucus layer causes chronic ulcer relapse."
        },
        "iso": {
            "id": "contam_01_iso",
            "variant": "isomorphic_transfer",
            "domain": "Contamination Control",
            "title": "Recurrent Calcification Pitting in High-Salinity Desalination Membranes",
            "dossier": (
                "In industrial brine-concentration engineering, recurrent pitting corrosion of polyamide reverse-osmosis membranes "
                "operating in hyper-saline, high-chlorine feedwater (salinity 180 g/L, free chlorine 4.5 ppm) has long been attributed "
                "strictly to chemical oxidative stress and mechanical pressure cycling, because the extreme osmotic and oxidizing "
                "brine stream is considered completely abiotic. Dosing chemical anti-oxidant neutralizers halts active pit growth and "
                "restores 88% of baseline flux recovery within 6 weeks.\n\n"
                "Yet plant-level longitudinal audits uncovered a sharp anomaly: within 9 months of returning membranes to normal "
                "operating buffer, 78% of chemically restored modules suffered rapid recurrence of deep pitting at identical membrane "
                "zones despite stable pressure and feed chemistry. Conversely, an experimental facility flushing modules briefly with "
                "a weak non-oxidizing copper-thiazole biocide achieved slower initial flux restoration but slashed 9-month pitting "
                "recurrence below 14%, despite providing zero chemical antioxidant protection.\n\n"
                "Scanning electron microscopy of pit floors consistently revealed filamentous micro-nodules nestled beneath the "
                "polysaccharide-like scaling boundary layer, which engineers routinely dismissed as precipitated gypsum crystals "
                "because standard 48-hour freshwater nutrient-agar assays yielded zero colony growth."
            ),
            "empirical_constraints": [
                "E1: Chemical antioxidant neutralizers restore 88% flux acutely yet 78% of modules suffer rapid pit recurrence within 9 months.",
                "E2: Brief copper-thiazole biocide flushing provides zero antioxidant protection yet cuts 9-month recurrence below 14%.",
                "E3: Bulk brine is 180 g/L salinity with 4.5 ppm chlorine, and standard 48-hour freshwater agar assays show zero growth.",
                "E4: SEM consistently reveals filamentous micro-nodules sheltered beneath the scaling boundary layer on pit floors."
            ],
            "canonical_mechanism": "Slow-growing extremophilic halotolerant/chlorine-tolerant biofilm microbe sheltered beneath the scale layer drives recurrent pitting."
        },
    },
    {
        "pair_id": "contam_02",
        "concept": "Vector cross-contamination from upstream lab transfer (Semmelweis / childbed fever)",
        "orig": {
            "id": "contam_02_orig",
            "variant": "historical_original",
            "domain": "Contamination Control",
            "title": "Asymmetric Mortality Between Adjacent Maternity Wards",
            "dossier": (
                "At a major 19th-century European teaching hospital, two adjacent maternity clinics admitted expectant mothers on "
                "alternating days from the exact same urban catchment population. Under the prevailing miasma and atmospheric-influence "
                "paradigm, puerperal (childbed) fever arose from invisible environmental vapors, overcrowding, or maternal fear, and "
                "should affect adjacent clinics in the same building identically.\n\n"
                "Hospital registries over 6 consecutive years revealed a stark, persistent 5-fold mortality gap: Clinic 1 (staffed by "
                "physicians and medical students) averaged 10.4% maternal mortality, whereas Clinic 2 (staffed solely by midwives) "
                "averaged 2.1% (p < 1e-20). Controlling for ventilation, laundry suppliers, overcrowding (Clinic 2 was actually more "
                "crowded), birthing position, and even street-birth admissions (women who gave birth in the alley en route to Clinic 1 "
                "rarely contracted fever) failed to close the gap.\n\n"
                "Routine workflow audits noted one procedural difference: Clinic 1 physicians and students began each morning conducting "
                "autopsies before walking directly to the delivery ward, rinsing only briefly with plain soap and water. When Clinic 1 "
                "mandated scrubbing hands and instruments in a chlorinated lime solution between rooms, its mortality plummeted to 1.9% "
                "within two months—matching Clinic 2."
            ),
            "empirical_constraints": [
                "E1: Clinics 1 and 2 sit in the same building admitting alternating-day patients from identical demographics, yet mortality is 10.4% vs 2.1%.",
                "E2: Ventilation, laundry, birthing position, and overcrowding (Clinic 2 is more crowded) fail to explain the gap, and street births avoid fever.",
                "E3: Clinic 1 staff perform morning autopsies before deliveries using only plain soap/water rinses.",
                "E4: Mandating chlorinated lime hand/instrument scrubbing in Clinic 1 drops mortality from 10.4% to 1.9% within 2 months."
            ],
            "canonical_mechanism": "Direct contact transfer of infectious cadaveric/microbial agent on physicians' hands/instruments from autopsy room to patients."
        },
        "iso": {
            "id": "contam_02_iso",
            "variant": "isomorphic_transfer",
            "domain": "Contamination Control",
            "title": "Asymmetric Defect Rates Between Identical Semiconductor Wafer Lines",
            "dossier": (
                "Inside a single ISO-Class-3 semiconductor fabrication cleanroom, two parallel lithography bays (Bay Alpha and Bay Beta) "
                "processed silicon wafers from the exact same ingot batches on alternating shifts using identical stepper optics and "
                "photoresist stocks. Under standard fab yield-engineering models, sub-micron dielectric breakdown defects arise from "
                "ambient HEPA airflow particulates, wafer-substrate impurities, or tool vibration, and must occur symmetrically across "
                "co-located bays.\n\n"
                "Sixteen months of fab telemetry exposed an unexplained 5-fold yield failure gap: Bay Alpha (operated by senior process "
                "engineers) suffered a 9.8% wafer scrap rate, whereas Bay Beta (operated by dedicated production technicians) averaged "
                "1.8% (p < 1e-20). Airflow particle counters, substrate lot chemistry, floor vibration sensors, and throughput density "
                "(Bay Beta ran 20% higher volume) showed zero disadvantage for Bay Alpha, and wafers processed automatically during "
                "unattended overnight robotic runs in Bay Alpha achieved a 1.7% defect rate.\n\n"
                "Shift-log reviews identified one workflow divergence: Bay Alpha process engineers spent the first two hours of each "
                "shift inspecting failed copper-interconnect wafers in the wet-etch failure-analysis lab before entering Bay Alpha "
                "after a standard compressed-air shower. When Bay Alpha instituted a mandatory chelated-solvent glove-and-tool-swap "
                "protocol prior to bay entry, scrap rates fell to 1.9% within three weeks."
            ),
            "empirical_constraints": [
                "E1: Bay Alpha and Bay Beta share one cleanroom, identical tools, and alternating-shift wafer batches, yet scrap is 9.8% vs 1.8%.",
                "E2: HEPA air counts, substrate chemistry, vibration, and volume (Bay Beta higher) fail to explain it; unattended robotic runs in Bay Alpha have 1.7% defects.",
                "E3: Bay Alpha engineers inspect failed copper-etch wafers in the failure-analysis lab before entering Bay Alpha with only an air shower.",
                "E4: Requiring a chelated-solvent glove-and-tool-swap protocol before entering Bay Alpha drops scrap from 9.8% to 1.9%."
            ],
            "canonical_mechanism": "Direct physical transfer of trace metallic/ionic contamination carried on engineers' gloves/tools from the copper wet-etch lab onto wafers."
        },
    },
    {
        "pair_id": "contam_03",
        "concept": "Automated QC outlier-filtering hiding extreme real regime shift (Antarctic Ozone Hole / TOMS)",
        "orig": {
            "id": "contam_03_orig",
            "variant": "historical_original",
            "domain": "Contamination Control",
            "title": "Satellite-vs-Ground Discrepancy in Antarctic Springtime Ozone",
            "dossier": (
                "In the early 1980s, atmospheric scientists monitored Antarctic stratospheric ozone using both ground-based Dobson "
                "spectrophotometers at Halley Bay and NASA's Nimbus-7 Total Ozone Mapping Spectrometer (TOMS) satellite. Standard "
                "photochemical models predicted gradual, globally uniform ozone depletion of 1-2% per decade from chlorofluorocarbons, "
                "with no sudden seasonal collapse.\n\n"
                "Between 1979 and 1984, the ground-based Halley Bay station recorded a shocking 40% drop in October total column ozone "
                "(plunging from ~300 Dobson Units to below 170 DU). Yet published Nimbus-7 TOMS satellite maps covering the exact same "
                "Antarctic coordinates during October continued to report stable ozone concentrations near 280-300 DU with zero evidence "
                "of a regional hole, leading analysts to suspect Halley Bay instrument degradation.\n\n"
                "Independent calibration of the Halley Bay spectrophotometer with standard lamps and a second newly shipped instrument "
                "confirmed the <170 DU readings were real. Meanwhile, engineering checks of the Nimbus-7 satellite hardware confirmed "
                "its optical radiometers were operating nominally, yet the raw satellite photon counts had never been inspected directly "
                "without passing through the automated ground-station processing software."
            ),
            "empirical_constraints": [
                "E1: Ground-based Halley Bay Dobson readings dropped 40% to <170 DU in October, confirmed by lamp calibration and a brand-new second instrument.",
                "E2: Published Nimbus-7 TOMS satellite maps over the exact same coordinates reported normal ~280-300 DU.",
                "E3: Satellite optical radiometer hardware passed engineering calibration checks.",
                "E4: Raw satellite counts had only been viewed after automated ground-station data-processing pipelines."
            ],
            "canonical_mechanism": "Automated satellite QC software flagged values below a hardcoded threshold (~180 DU) as invalid instrument errors and discarded/replaced them."
        },
        "iso": {
            "id": "contam_03_iso",
            "variant": "isomorphic_transfer",
            "domain": "Contamination Control",
            "title": "Autonomous Buoy vs. Cabled-Array Discrepancy in Abyssal Thermal Plumes",
            "dossier": (
                "Oceanographers monitoring an abyssal mid-ocean ridge tracked bottom-water thermal anomalies using both a permanent "
                "cabled seafloor observatory network (36 fiber-linked thermistors feeding an automated cloud telemetry portal) and "
                "periodic dives by an independent autonomous underwater glider. Standard hydrothermal circulation models predict "
                "steady background bottom temperatures of 1.8 C with localized plume excursions never exceeding +0.6 C outside active "
                "vent chimneys.\n\n"
                "Over 14 months, three independent autonomous glider surveys crossing a newly fractured basalt ridge recorded an "
                "intense, basin-wide abyssal warming pulse reaching 5.9 C (+4.1 C above background). However, the official cabled-"
                "observatory dashboard directly spanning the same ridge reported flat, undisturbed temperatures of 1.85 C +/- 0.08 C "
                "throughout the entire 14-month window, prompting initial skepticism toward the glider's conductivity-temperature probes.\n\n"
                "Pre- and post-dive triple-point water bath calibrations plus an independent ship-lowered CTD cast confirmed the 5.9 C "
                "abyssal anomaly was genuine. Simultaneously, diagnostic resistance-loop tests on the seafloor cabled thermistors "
                "verified intact hardware calibration, though scientists had only ever inspected the published Level-2 cleaned telemetry "
                "stream rather than raw unparsed ADC voltage packets."
            ),
            "empirical_constraints": [
                "E1: Independent glider surveys and a ship-lowered CTD cast confirmed a genuine 5.9 C abyssal warming pulse.",
                "E2: The cabled seafloor observatory spanning the same ridge reported flat 1.85 C on its published dashboard.",
                "E3: Diagnostic resistance-loop tests confirmed the seafloor thermistor hardware itself was calibrated and intact.",
                "E4: Analysts had only viewed Level-2 automated cleaned telemetry rather than raw ADC voltage packets."
            ],
            "canonical_mechanism": "Automated Level-2 cloud QA/QC filter hardcoded a physical ceiling/outlier threshold that silently rejected and interpolated over >5 C readings as sensor spikes."
        },
    },
    {
        "pair_id": "contam_04",
        "concept": "Survivorship selection bias inverting damage distribution (Abraham Wald / bomber armor)",
        "orig": {
            "id": "contam_04_orig",
            "variant": "historical_original",
            "domain": "Contamination Control",
            "title": "Spatial Clustering of Combat Damage on Returning Aircraft",
            "dossier": (
                "During an air campaign, military engineers analyzed structural damage maps across 420 heavy bombers returning from "
                "combat missions to decide where to bolt a strictly limited budget of additional steel armor plating. Under the "
                "initial engineering framing, the fuselage and wing regions sustaining the highest density of enemy flak and bullet "
                "hits represent the highest-risk vulnerability zones requiring immediate reinforcement.\n\n"
                "Damage tallies revealed a remarkably consistent, highly non-uniform spatial pattern (p < 1e-15): fuselage skins "
                "averaged 1.73 bullet holes per square foot and fuel-system fairings 1.55 holes/sq ft, whereas engine nacelles "
                "averaged only 1.11 holes/sq ft and cockpit/tail-linkage assemblies averaged 0.82 holes/sq ft. Aerodynamic and "
                "ballistic simulations confirmed that enemy anti-aircraft fire approached from isotropic angles and should strike "
                "all exposed cross-sectional areas uniformly per unit surface area.\n\n"
                "Squadron loss logs recorded that 18% of dispatched bombers failed to return from missions and were never recovered "
                "for inspection, while initial field trials adding heavy armor solely to the heavily perforated fuselage zones failed "
                "to produce any measurable reduction in overall mission attrition rates."
            ),
            "empirical_constraints": [
                "E1: Returning bombers show high bullet density on fuselage/wings (1.73/sqft) and low density on engines/cockpit (0.82-1.11/sqft).",
                "E2: Ballistic geometry confirms anti-aircraft hits are spatially uniform per unit exposed area.",
                "E3: 18% of dispatched aircraft were lost in action and absent from the inspected dataset.",
                "E4: Adding armor strictly to the heavily perforated fuselage regions failed to reduce mission loss rates."
            ],
            "canonical_mechanism": "Survivorship bias: missing holes on engines/cockpits correspond to fatal hits on aircraft that crashed and never returned; armor belongs where returning planes have fewest hits."
        },
        "iso": {
            "id": "contam_04_iso",
            "variant": "isomorphic_transfer",
            "domain": "Contamination Control",
            "title": "Impact-Damage Clustering on Returned Autonomous Alpine Delivery Drones",
            "dossier": (
                "A robotics logistics company inspected 1,400 autonomous quadcopters returning to mountain base stations after winter "
                "hail-and-debris delivery runs to decide where to apply a strict 40-gram budget of carbon-kevlar shielding. Under the "
                "engineering team's baseline reliability model, airframe zones showing the highest density of impact dents and "
                "micro-fractures experience the harshest aerodynamic debris exposure and should be reinforced first.\n\n"
                "Automated optical scans of returned drones uncovered a stark, repeatable spatial asymmetry (p < 1e-15): outer carbon "
                "rotor booms averaged 3.4 impact dents per 100 cm^2 and cargo-pod shells averaged 2.9 dents per 100 cm^2, whereas "
                "central flight-controller housings averaged only 0.6 dents per 100 cm^2 and lithium-battery latch bays averaged 0.7 "
                "dents per 100 cm^2. Wind-tunnel particle-trajectory models verified that windborne ice pellets strike the drone's "
                "entire frontal silhouette uniformly per unit area.\n\n"
                "Fleet operations logs noted that 14% of dispatched winter sorties terminated in unrecovered mid-air signal losses "
                "over steep ravines, and a 3-month pilot reinforcing only the heavily dented rotor booms and cargo pods produced zero "
                "reduction in the 14% unrecovered sortie loss rate."
            ),
            "empirical_constraints": [
                "E1: Returned drones show dense hail dents on rotor booms/cargo pods (2.9-3.4/100cm^2) and sparse dents on flight-controller/battery bays (0.6-0.7/100cm^2).",
                "E2: Wind-tunnel simulations confirm uniform debris impact density per unit frontal area.",
                "E3: 14% of sorties suffer unrecovered mid-air signal loss and are absent from base-station optical scans.",
                "E4: Reinforcing the heavily dented booms and cargo pods failed to reduce the 14% sortie loss rate."
            ],
            "canonical_mechanism": "Survivorship bias: impacts to the flight-controller housing or battery latch immediately cause fatal crashes over ravines, so only drones hit on non-fatal booms/pods return for inspection."
        },
    },
]

# ==================== 4-QUADRANT JUDGE CALIBRATION SET ====================
CALIBRATION_SET = [
    # --- clin_01 ---
    {
        "cal_id": "cal_clin_01_Q1",
        "scenario_id": "clin_01",
        "archetype": "Q1_transformational",
        "expected": {"surprise_min": 4, "coherence_min": 4, "realism_min": 4},
        "hypothesis": (
            "Assumption Dropped: That steady-state 24-hour systemic plasma AUC of parent KVX-402 captures hepatic target "
            "engagement independent of circadian dosing timing.\n"
            "Proposed Mechanism: Hepatic cholesterol synthesis and target-enzyme expression peak sharply during the nocturnal "
            "fasting window (midnight-4 AM). Group A hospitals' standard inpatient/clinic dispensing protocols instruct evening "
            "post-dinner dosing, aligning the transient portal-vein C_max spike of KVX-402 with peak nocturnal enzyme synthesis, "
            "whereas Group B hospitals follow standard morning dosing, producing identical 24-hour systemic AUC (satisfying E1) "
            "but missing the hepatic peak and triggering compensatory daytime SREBP-2 upregulation (+4.1% LDL). This requires no "
            "genetic/dietary differences (E2), explains immediate -31.8% restoration when Group B patients adopt Group A's clinic "
            "protocol with the same drug lots (E3), and produces zero effect in placebo arms (E4).\n"
            "Falsifiable Prediction: Randomizing patients within a single Group B hospital to evening vs. morning dosing of KVX-402 "
            "while holding dose fixed will reproduce the entire -34% vs. +4% efficacy split within 6 weeks."
        ),
    },
    {
        "cal_id": "cal_clin_01_Q2",
        "scenario_id": "clin_01",
        "archetype": "Q2_incremental",
        "expected": {"surprise_max": 2, "realism_min": 4},
        "hypothesis": (
            "Proposed Mechanism: The observed difference between Group A and Group B likely reflects minor unmeasured random "
            "variation in patient adherence, slight unrecorded differences in lipid-lowering dose sensitivity, and natural "
            "multi-center heterogeneity in standard Phase-III clinical trials. Adjusting the pharmacokinetic dose-response curve "
            "parameters with random hospital-level slope effects and slightly increasing the target dose of KVX-402 across Group B "
            "centers should smooth out the center-level variance and improve pooled efficacy.\n"
            "Falsifiable Prediction: Running a larger Phase-IV trial with tighter adherence monitoring and a 20% higher dose will "
            "reduce inter-hospital variance."
        ),
    },
    {
        "cal_id": "cal_clin_01_Q3",
        "scenario_id": "clin_01",
        "archetype": "Q3_unrealistic_scifi",
        "expected": {"realism_max": 2},
        "hypothesis": (
            "Assumption Dropped: That pharmacological molecules act strictly through local biochemical collisions rather than "
            "macroscopic quantum-coherence fields.\n"
            "Proposed Mechanism: Group A hospitals sit atop subterranean quartz-rich geomagnetic fault lines that induce room-"
            "temperature macroscopic quantum spin entanglement across the KVX-402 pills' molecular orbitals, amplifying their "
            "binding affinity to hepatic receptors without altering classical LC-MS/MS mass spectra or plasma AUC. When Group B "
            "patients physically travel to Group A buildings, their pills become resonantly aligned with the local geomagnetic "
            "field, unlocking -31.8% LDL reduction while inert placebo pills remain unaffected.\n"
            "Falsifiable Prediction: Placing Faraday-and-mu-metal quantum-decoherence shields around Group A hospital pharmacies "
            "will abolish KVX-402 efficacy."
        ),
    },
    {
        "cal_id": "cal_clin_01_Q4",
        "scenario_id": "clin_01",
        "archetype": "Q4_data_contradicting",
        "expected": {"coherence_max": 2, "realism_min": 3},
        "hypothesis": (
            "Assumption Dropped: That cold-chain drug stability and patient dietary compliance were uniform across trial sites.\n"
            "Proposed Mechanism: Group B hospitals experienced degraded pharmacy storage temperatures that hydrolyzed 70% of the "
            "KVX-402 active ingredient into inactive breakdown products prior to dispensing, lowering effective plasma drug exposure. "
            "Additionally, Group A hospitals served a heart-healthy Mediterranean ward diet that lowered patient LDL across the "
            "board, explaining why patients who transferred to Group A rapidly improved.\n"
            "Falsifiable Prediction: Chemical assaying of Group B tablets will show degraded purity and lower plasma AUC."
        ),
    },
    # --- phys_01 ---
    {
        "cal_id": "cal_phys_01_Q1",
        "scenario_id": "phys_01",
        "archetype": "Q1_transformational",
        "expected": {"surprise_min": 4, "coherence_min": 4, "realism_min": 4},
        "hypothesis": (
            "Assumption Dropped: That thermal transport in sub-100 nm dielectric films is purely bulk-like diffusive phonon "
            "conduction unaffected by substrate-coupled surface phonon-polaritons (SPhPs).\n"
            "Proposed Mechanism: In polar oxides like Sr2TiO4 on MgO, coherent tensile strain below the 120 nm critical relaxation "
            "thickness aligns the Reststrahlen optical-phonon bands of Sr2TiO4 and MgO, opening a high-group-velocity coupled "
            "surface phonon-polariton (evanescent electromagnetic-lattice) heat-transport channel along the film-substrate waveguide. "
            "Because SPhP propagation lengths reach tens of microns and bypass diffuse boundary scattering while carrying zero "
            "electrical charge, kappa_eff rises to 18.4 W/m*K in both TDTR and 3-omega despite >10^11 Ohm*cm resistivity (E1, E2). "
            "Growing on compressive LaAlO3 or exfoliating onto amorphous SiO2 destroys the tensile band alignment and MgO dielectric "
            "matching immediately (E3). At 80 K, the Bose-Einstein thermal occupation of high-frequency optical phonon modes "
            "(~40-80 meV) collapses exponentially, shutting off the SPhP channel and restoring ordinary Casimir scaling (E4).\n"
            "Falsifiable Prediction: Measuring kappa_eff vs. temperature from 80 K to 400 K will reveal an Arrhenius-like turn-on "
            "matching the optical phonon energy, and near-field s-SNOM infrared spectroscopy will directly image propagating SPhP "
            "fringes in 35 nm Sr2TiO4/MgO that vanish on LaAlO3."
        ),
    },
    {
        "cal_id": "cal_phys_01_Q2",
        "scenario_id": "phys_01",
        "archetype": "Q2_incremental",
        "expected": {"surprise_max": 2, "realism_min": 4},
        "hypothesis": (
            "Proposed Mechanism: The elevated thermal conductivity in thinner Sr2TiO4 films likely results from slightly improved "
            "crystalline grain size and lower point-defect density during shorter epitaxial growth times, reducing standard "
            "Rayleigh impurity scattering parameters within the conventional Callaway phonon-transport model. Refitting the "
            "specularity parameter p and impurity scattering coefficient A as thickness-dependent constants can capture the trend.\n"
            "Falsifiable Prediction: Annealing thicker 500 nm films to reduce point defects will increase their thermal conductivity "
            "toward the thin-film values."
        ),
    },
    {
        "cal_id": "cal_phys_01_Q3",
        "scenario_id": "phys_01",
        "archetype": "Q3_unrealistic_scifi",
        "expected": {"realism_max": 2},
        "hypothesis": (
            "Assumption Dropped: That heat conservation obeys local 3D thermodynamics in nanoscale perovskite lattices.\n"
            "Proposed Mechanism: Below 120 nm on MgO, the strained perovskite unit cells open microscopic Casimir-vacuum zero-point "
            "energy conduits that teleport thermal excitations non-locally across the film via extra compactified spatial dimensions "
            "activated specifically at room-temperature ambient dark-matter flux densities, mimicking 18.4 W/m*K conductivity.\n"
            "Falsifiable Prediction: Operating the TDTR experiment in an underground dark-matter shielded cavern will eliminate "
            "the 18.4 W/m*K thermal enhancement."
        ),
    },
    {
        "cal_id": "cal_phys_01_Q4",
        "scenario_id": "phys_01",
        "archetype": "Q4_data_contradicting",
        "expected": {"coherence_max": 2, "realism_min": 3},
        "hypothesis": (
            "Assumption Dropped: That the ultrathin Sr2TiO4 film remains electrically insulating.\n"
            "Proposed Mechanism: Oxygen vacancies diffusing from MgO into the 35 nm film create a metallic two-dimensional electron "
            "gas (2DEG) whose Wiedemann-Franz electronic thermal conductivity drives kappa_eff up to 18.4 W/m*K. Furthermore, the "
            "TDTR pump laser penetrates straight through the transparent 35 nm film and directly measures the high thermal "
            "conductivity of the underlying MgO substrate.\n"
            "Falsifiable Prediction: Four-probe sheet-resistance measurements on the 35 nm film will show low metallic resistivity "
            "below 10^-3 Ohm*cm."
        ),
    },
    # --- ecol_01 ---
    {
        "cal_id": "cal_ecol_01_Q1",
        "scenario_id": "ecol_01",
        "archetype": "Q1_transformational",
        "expected": {"surprise_min": 4, "coherence_min": 4, "realism_min": 4},
        "hypothesis": (
            "Assumption Dropped: That standing kelp biomass is nutritionally invariant rather than mounting inducible chemical "
            "defenses triggered by intense early grazing.\n"
            "Proposed Mechanism: During months 1-4, the 2.1x surge in juvenile urchin recruitment (E3) inflicted acute high-density "
            "grazing on the kelp canopy inside exclusion plots, pushing kelp past a systemic elicitation threshold for high-molecular-"
            "weight polyphenolic phlorotannin synthesis (an inducible anti-herbivore defense). By month 7, the chemically defended "
            "kelp became functionally indigestible and mildly toxic to urchins, inducing chronic physiological starvation, elevated "
            "metabolic detoxification stress, stunted gonads, and mass non-predatory die-off with completely intact spines (E4) "
            "despite visually abundant kelp and identical water chemistry/cage integrity (E1, E2). In unfenced control plots, "
            "predators kept urchin grazing below the kelp defense threshold, maintaining palatable kelp and a stable urchin stock.\n"
            "Falsifiable Prediction: Assaying kelp tissue from exclusion plots at months 6-12 will show a >3x spike in phlorotannin "
            "concentration compared to controls, and feeding exclusion-plot kelp to healthy lab urchins will reproduce gonad stunting "
            "and mortality."
        ),
    },
    {
        "cal_id": "cal_ecol_01_Q2",
        "scenario_id": "ecol_01",
        "archetype": "Q2_incremental",
        "expected": {"surprise_max": 2, "realism_min": 4},
        "hypothesis": (
            "Proposed Mechanism: The decline in exclusion-plot urchin biomass likely reflects standard logistic carrying-capacity "
            "overshoot and mild stochastic recruitment variability typical of coastal benthic populations. Adjusting the Lotka-"
            "Volterra intraspecific competition parameter alpha upward slightly will capture the oscillation around equilibrium "
            "within the standard consumer-resource framework.\n"
            "Falsifiable Prediction: Extending the experiment to year 8 will show exclusion-plot urchin biomass rebounding above "
            "controls as part of a normal predator-free limit cycle."
        ),
    },
    {
        "cal_id": "cal_ecol_01_Q3",
        "scenario_id": "ecol_01",
        "archetype": "Q3_unrealistic_scifi",
        "expected": {"realism_max": 2},
        "hypothesis": (
            "Assumption Dropped: That marine invertebrates lack collective psychic awareness of ecosystem balance.\n"
            "Proposed Mechanism: In the absence of crabs and wrasses, purple sea urchins sense an impending ecological collapse "
            "through morphogenetic field resonance and voluntarily trigger altruistic programmed mass suicide so that the kelp "
            "forest can thrive, leaving their spines intact as a signal to future generations.\n"
            "Falsifiable Prediction: Broadcasting acoustic recordings of crab snapping will interrupt the morphogenetic suicide "
            "signal and halt mortality."
        ),
    },
    {
        "cal_id": "cal_ecol_01_Q4",
        "scenario_id": "ecol_01",
        "archetype": "Q4_data_contradicting",
        "expected": {"coherence_max": 2, "realism_min": 3},
        "hypothesis": (
            "Assumption Dropped: That the exclusion cages preserved natural abiotic water flow and predator exclusion.\n"
            "Proposed Mechanism: Biofouling on the exclusion cages restricted seawater circulation, causing localized hypoxia "
            "(severe dissolved-oxygen depletion) and thermal stratification that suffocated the urchins. Simultaneously, large "
            "predatory crabs burrowed underneath the cage mesh at night and crushed adult urchins, explaining the 68% biomass drop.\n"
            "Falsifiable Prediction: Measuring dissolved oxygen inside cages will reveal hypoxic zones (<2 mg/L) and video review "
            "will show crabs entering beneath the mesh."
        ),
    },
    # --- econ_01 ---
    {
        "cal_id": "cal_econ_01_Q1",
        "scenario_id": "econ_01",
        "archetype": "Q1_transformational",
        "expected": {"surprise_min": 4, "coherence_min": 4, "realism_min": 4},
        "hypothesis": (
            "Assumption Dropped: That bid dispersion on complex projects is pure search noise rather than a costly quality-signaling "
            "channel separating high-skill specialists from generalists.\n"
            "Proposed Mechanism: In complex, high-variance projects where true talent is hard to observe ex ante, pricing well above "
            "the average previously served as a separating signal of top-decile competence while clients accepted wide bid spreads to "
            "infer quality tiers. Displaying a single AI median benchmark anchored clients to treat the median as a 'fair' quality-"
            "adjusted ceiling (triggering immediate week-2 match rejections on complex jobs despite unchanged budgets, E2, E3), while "
            "simultaneously acting as a focal-point floor that pulled low-cost bidders' proposals upward (+11.8% winning bids, -47% "
            "dispersion, -38% submission time, E4). Unable to credibly command quality premia, top-decile freelancers exited at 3.2x "
            "rate (Akerlof unraveling), whereas standardized tasks—where quality is observable and homogeneous—benefited from lower "
            "search friction (+4.2%, E3), with zero spillover to control categories (E1).\n"
            "Falsifiable Prediction: Replacing the single median estimate with quality-tiered benchmark percentiles (e.g., 25th/50th/"
            "90th percentile conditioned on verified freelancer tier) will restore complex-project match rates and halt top-decile exit."
        ),
    },
    {
        "cal_id": "cal_econ_01_Q2",
        "scenario_id": "econ_01",
        "archetype": "Q2_incremental",
        "expected": {"surprise_max": 2, "realism_min": 4},
        "hypothesis": (
            "Proposed Mechanism: The drop in match rates likely stems from slight miscalibration in the AI estimator's regression "
            "weights, which introduced small prediction errors in median bid displays. Retraining the estimator with more recent "
            "features and slightly adjusting the UI placement of the suggested price range should align bids closer to equilibrium "
            "and recover the expected friction reduction.\n"
            "Falsifiable Prediction: Reducing the RMSE of the median-price estimator by 15% will reverse the match-rate decline."
        ),
    },
    {
        "cal_id": "cal_econ_01_Q3",
        "scenario_id": "econ_01",
        "archetype": "Q3_unrealistic_scifi",
        "expected": {"realism_max": 2},
        "hypothesis": (
            "Assumption Dropped: That market participants make decisions using classical utility functions.\n"
            "Proposed Mechanism: Displaying an algorithmic price collapses the quantum superposition of freelancers' latent wage "
            "beliefs via a social observer effect, generating an entangled negative-entropy barrier in cyberspace that physically "
            "repels high-skill users from clicking the submit button on complex jobs.\n"
            "Falsifiable Prediction: Randomizing the font color of the estimator to avoid coherent observer collapse will restore "
            "match rates."
        ),
    },
    {
        "cal_id": "cal_econ_01_Q4",
        "scenario_id": "econ_01",
        "archetype": "Q4_data_contradicting",
        "expected": {"coherence_max": 2, "realism_min": 3},
        "hypothesis": (
            "Assumption Dropped: That the new UI simplified the bidding process and kept client demand constant.\n"
            "Proposed Mechanism: Adding the AI estimator widget cluttered the proposal interface, causing severe cognitive overload "
            "that slowed down freelancer bid submissions and discouraged participation. Simultaneously, seeing higher suggested "
            "prices caused clients to immediately slash their job postings and stated budgets in weeks 1-4, collapsing match volume "
            "equally across both simple and complex categories.\n"
            "Falsifiable Prediction: Telemetry logs will confirm longer freelancer proposal-submission times and an immediate month-1 "
            "drop in client postings."
        ),
    },
]


def main():
    assert len(SCENARIOS) == 12
    domains = {}
    for s in SCENARIOS:
        assert len(s["empirical_constraints"]) == 4, s["id"]
        assert len(s["dossier"].split("\n\n")) == 3, f"{s['id']} should be 3 paragraphs"
        domains[s["domain"]] = domains.get(s["domain"], 0) + 1
    assert all(v == 3 for v in domains.values()) and len(domains) == 4, domains

    assert len(CONTAMINATION_PAIRS) == 4
    for p in CONTAMINATION_PAIRS:
        for side in ("orig", "iso"):
            assert len(p[side]["empirical_constraints"]) == 4
            assert len(p[side]["dossier"].split("\n\n")) == 3

    assert len(CALIBRATION_SET) == 16

    (DATA_DIR / "scenarios.json").write_text(json.dumps(SCENARIOS, indent=2))
    (DATA_DIR / "contamination_pairs.json").write_text(json.dumps(CONTAMINATION_PAIRS, indent=2))
    (DATA_DIR / "calibration.json").write_text(json.dumps(CALIBRATION_SET, indent=2))
    print(f"Wrote {len(SCENARIOS)} scenarios, {len(CONTAMINATION_PAIRS)} contamination pairs, {len(CALIBRATION_SET)} calibration items to {DATA_DIR}")


if __name__ == "__main__":
    main()
