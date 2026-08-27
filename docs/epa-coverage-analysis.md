# EPA SWMM coverage-gap analysis (July, 2026)


**[INFERENCE] Do not characterize the visible deficit as mainly error/reporting.** 

Error handling and validation are clearly under-tested, but the strongest low-coverage evidence 
also contains supported hydraulic regimes, routing modes, geometry variants, and numerical boundary 
behavior. 

A global numerical majority cannot be claimed from this report alone: classifying all 3,384 unhit 
outcomes would require a full manual review, and the filtered report does not expose such a 
classification.

## Lowest branch-rate files

The first ten rows are the lowest-rate files in the filtered solver report through `hotstart.c`; 
the final three are additional model/control files with large branch denominators.

| File | Covered / total branches | Unhit | Why it matters here |
| --- | ---: | ---: | --- |
| [street.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.street.c.4be413661a01fd6955ec49a470fe9ff4.html) | 28 / 60 (46.7%) | 32 | Street cross-section parsing and routing mode |
| [roadway.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.roadway.c.34c60e2b5ab0de8af4ec66d4ba6cf764.html) | 20 / 38 (52.6%) | 18 | Roadway-weir hydraulic calculation |
| [iface.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.iface.c.49fdcd3fa15ba8a6ec33317d973622bd.html) | 102 / 185 (55.1%) | 83 | Interface-file I/O and quality-data mode |
| [exfil.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.exfil.c.e733ee3e3413c8952b5c1bf4fcf75c58.html) | 29 / 52 (55.8%) | 23 | Storage exfiltration shapes and bank loss |
| [datetime.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.datetime.c.b4947e4004da882e847c98433b00dcde.html) | 61 / 108 (56.5%) | 47 | Date format and boundary handling |
| [inlet.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.inlet.c.8a303809c0bfec5b0704e3b19dc393e8.html) | 260 / 459 (56.6%) | 199 | Inlet capture, specialized inlet types, routing modes |
| [table.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.table.c.06b5bae912b5217569529a3d1fe1eeba.html) | 130 / 226 (57.5%) | 96 | Curves, time series, file-backed tables |
| [rain.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.rain.c.93cac3b1447ddaf26c0d0567be3190cb.html) | 189 / 319 (59.2%) | 130 | External rainfall formats and errors |
| [lidproc.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.lidproc.c.fa8e52dec9cc856cc60073640f8bbd41.html) | 217 / 366 (59.3%) | 149 | LID states, overflow, drains, and controls |
| [hotstart.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.hotstart.c.7d6481369abe409f3b9d53a043d00fea.html) | 100 / 168 (59.5%) | 68 | Compatibility and file-error handling |
| [controls.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.controls.c.7d2cc1cc59e6790be8af9f19ff91f990.html) | 315 / 523 (60.2%) | 208 | Rule-language and control semantics |
| [gwater.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.gwater.c.9b6792d3f2ac7298d69e91338e3d232d.html) | 123 / 198 (62.1%) | 75 | Groundwater modes and physical limits |
| [xsect.c](http://swmm-rs.github.io/swmm-bench/epa-swmm-coverage.xsect.c.61c30db9c77960303643c05d808c68f1.html) | 320 / 511 (62.6%) | 191 | Core cross-section geometry and lookup logic |

## What the missed branches are

| Class | Direct report/source evidence | Reading of the evidence |
| --- | --- | --- |
| **Error / reporting / validation** | **[OBSERVED]** `street_readParams` has unhit `error_setInpError` alternatives at report lines 80, 84, 90, 95, 100, 106, and 112. `hotstart.c` has unhit open, format, and object-count mismatch reporting paths at 118–176. `inlet_validate` has an unhit warning/removal path at 527–545. | **[INFERENCE]** These are real coverage gaps, especially for malformed input and file compatibility; they are not the whole story. |
| **Specialized supported features** | **[OBSERVED]** `xsect_setParams` is **16/24** at line 231: unexecuted cases include `EGGSHAPED`, `GOTHIC`, `CATENARY`, `SEMIELLIPTICAL`, `BASKETHANDLE`, and `SEMICIRCULAR` (295–363), with other switch outcomes also absent. `exfil_initState` is **2/4** at line 93 and has unexecuted cylindrical, conical, and pyramidal storage paths (144–151). `inlet.c` misses DROP_CURB, DROP_GRATE, and SLOTTED handling (1337–1359) and generic-grate behavior (1456–1457). | **[INFERENCE]** The low rates reflect substantial untested model-option breadth, not merely rejected input. |
| **Normal hydraulic/numerical decisions** | **[OBSERVED]** All three `roadway.c` functions execute, yet the file is only **20/38**. Its missed outcomes include SI conversion, fixed versus variable discharge coefficient, reverse direction, high head/width ratio, downstream submergence, and interpolation endpoints (110–193). In `xsect.c`, `invLookup` is called **832,898** times but has only **38.9%** branch coverage; its unhit outcomes are decreasing-table, end-point, zero-slope, and clamp cases (1543–1565). | **[INFERENCE]** This is the strongest evidence against an error/reporting-only explanation: these are computational regime choices in ordinary solver functions. |
| **Other: defensive, allocation, and compatibility paths** | **[OBSERVED]** `street_create` and `exfil.c` include unhit allocation-null guards; `iface.c`, `rain.c`, and `hotstart.c` combine external-file formats/versions with file-failure paths. | **[INFERENCE]** These should be tested separately from both numerical behavior and user-input diagnostics. |

## Prioritized test targets

1. **Cross-section option and boundary matrix (`xsect.c`, 191 unhit outcomes).** Exercise the unhit shape dispatches, shallow/near-full geometry, and inverse-table decreasing/end-point/clamp cases. The report's 16/24 dispatch and 38.9% `invLookup` coverage make this the largest concentrated core-logic target.
2. **Inlet routing and inlet-type matrix (`inlet.c`, 199).** Add a non-Dynamic-Wave, on-sag inlet scenario that reaches the limit pass at 573–639; pair it with DROP_CURB, DROP_GRATE, SLOTTED, generic-grate, and depressed-gutter cases.
3. **LID and groundwater state boundaries (`lidproc.c`, 149; `gwater.c`, 75).** Cover full soil/storage, overflow, underdrain coefficient/control-curve, clogging, fixed-depth, saturation-clamp, and alternate groundwater-flow cases.
4. **Roadway-weir regimes (`roadway.c`, 18).** A compact matrix can cover SI conversion, fixed/variable coefficient, paved/gravel, low/high head ratio, submergence, reverse direction, and interpolation bounds. This is a low-cost, purely non-reporting target.
5. **Validation and persistence compatibility.** Add malformed street/inlet/table input, invalid inlet placement, failed/legacy hotstart, and external-file error cases after the behavior tests above. These defend diagnostics but should not displace high-value hydraulic paths.
