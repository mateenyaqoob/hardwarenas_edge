# Design notes

**One encoding, many searchers.** An architecture is a tuple of operator indices (one per layer). Random,
evolutionary, NSGA-II and the differentiable super-net all use it, so their results are comparable.

**Cheap checks before expensive evaluations.** `Evaluator.cheap()` (latency, SRAM, flash) runs before
`Evaluator.evaluate()` (accuracy). The search budget counts accuracy evaluations only - the same economics as
real NAS, where a latency LUT lookup costs microseconds and an accuracy estimate costs GPU-minutes.

**Why the simulator and the LUT disagree.** `build_lut` times each block in isolation (cold input, full launch
overhead). `sim_latency` models the whole network: activations that fit on chip are not re-read, and
asynchronous runtimes hide part of each block's launch overhead (`DeviceProfile.overlap`). This reproduces
the additivity error students meet on real hardware, in a controlled setting.

**Peak SRAM.** Peak live activations of a block = max over its kernels of (input + output [+ residual input]).
For MCUs this, not MACs, is usually the binding constraint (MCUNet).

**Differentiable latency.** `SuperNet.expected_latency` = base + sum of (Gumbel-softmax weights x LUT row).
The FBNet loss `CE * alpha * log(latency_us)^beta` and its hinge/linear alternatives are in `diff_search.py`;
`tests/test_torch.py` checks that the gradient pushes probability mass toward cheaper operators.

## Known limitations (good project ideas)

* Accuracy for the evolutionary/NSGA-II path is synthetic. Plug in a real estimator via
  `Evaluator(space, acc_fn, lat_fn)`; `acc_fn` is any callable `arch -> float`.
* No weight-sharing super-net with progressive shrinking (Once-for-All) - Lab 5 asks you to build a small one.
* Int8 quantisation is modelled only through `dtype_bytes`; real int8 kernels change *relative* operator costs.
* Energy is not modelled; adding `energy = sum(parts) * power` as a second constraint is a 10-line change.
* The differentiable search runs every candidate op in every layer, so it is memory hungry
  (ProxylessNAS's path binarisation fixes this).
