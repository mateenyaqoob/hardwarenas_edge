# Labs: hardware-aware NAS (4-5 hours total; Labs 1-3 need no GPU or data)

Setup: `pip install -e ".[viz,dev]"` then `pytest -q` (all tests must pass before you start).

## Lab 1 - Hardware metrics and latency models (45 min)
1. Print `arch_metrics` for the all-`k3_e1` and all-`k5_e6` architectures. Which constraint (flash, SRAM, MACs)
   grows fastest? Why does peak SRAM depend on the *largest* layer pair, not the sum?
2. Run `python -m hwnas.cli additivity --device D` for D in `mcu`, `rpi4`, `edge_gpu`. Explain the ordering of the
   LUT error. (Hint: `DeviceProfile.overlap` and `cache_bytes`.)
3. In `latency.py`, `LatencyPredictor` is a ridge regression on `[LUT sum, one-hot arch]`. Why can it fix the error
   almost perfectly here but would not on real hardware? What would you add (noise, interactions between
   neighbouring layers)?
4. **Deliverable:** one paragraph recommending LUT vs predictor vs on-device measurement for (a) an MCU,
   (b) a phone NPU, (c) a Jetson.

## Lab 2 - Constrained search (60 min)
1. `python examples/convergence.py`. Why does evolution win? Re-run with `--latency-ms 0.65` and `1.4`. When does the
   gap close, and why?
2. Add a *flash* limit (`--flash-kb 150`) and a *SRAM* limit (`--sram-kb 64`) to `search`. Which constraint binds?
3. Implement the MnasNet soft reward `mnas_reward(acc, lat, T, w=-0.07)` as the fitness in `evolutionary_search`
   (currently accuracy among feasible). Compare soft vs hard constraint at the same T. How often does the soft
   version return an architecture that violates T?

## Lab 3 - Pareto fronts and device specificity (45 min)
1. `python -m hwnas.cli pareto --device rpi4` and `--device edge_gpu`. Compare front shapes.
2. `python -m hwnas.cli devices`. Describe, per device, what the search chose (depth, kernel size, expansion) and
   explain it from the device profile. Then **cross-deploy**: evaluate each device's winner on the other two devices
   with `sim_latency`. How much latency do you lose?
3. Change one number in `devices.py` (e.g. make depthwise efficient on `edge_gpu`) and predict, then verify, how the
   chosen architecture changes.

## Lab 4 - Real measurements and differentiable search (90 min, PyTorch)
1. `python -m hwnas.cli measure-lut --out luts/host_cpu.json` (use `--iters 100`). Compare the table's *ranking*
   of operators to the simulated `rpi4` LUT (Kendall tau). Which operator is the biggest surprise?
2. Search with the measured LUT: `diff-search --lut luts/host_cpu.json --lat-loss fbnet`, then `hinge` with
   `--latency-ms` set to 60% of the all-`k5_e6` latency. Plot `expected_lat_ms` per epoch from the output JSON.
3. `train found.json` and report accuracy, MACs, and *measured* host latency. Does the LUT estimate match?
4. Explain the role of the warm-up epochs and of temperature annealing in `differentiable_search`.

## Lab 5 - Stretch projects (pick one)
* **Real accuracy for evolution:** replace `SyntheticAccuracy` by (a) a 2-epoch proxy training on a CIFAR-10 subset,
  (b) `proxies.synflow`. Measure rank correlation with full training on 20 random architectures.
* **Once-for-All-lite:** train a weight-sharing super-net with random sub-network sampling, fit an accuracy
  predictor on sampled sub-nets, then run `evolutionary_search` with it.
* **Energy-aware search:** add an energy model and make NSGA-II optimise (accuracy, latency, energy).
* **MCU mode:** put the whole space at 96x96 input, add an SRAM constraint of 256 KB and a flash constraint of
  1 MB, and report how the Pareto front changes.

### Marking guide (suggested)
Correct use of constraints (20%), quality of explanations tied to the device model (30%), experiments with seeds and
uncertainty (20%), honesty about synthetic vs real quantities (15%), code quality and tests for any new code (15%).
