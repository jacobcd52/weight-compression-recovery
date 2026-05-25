# weight-compression-recovery

**Threat model:** weight exfiltration. If an attacker steals a small number of bytes
from a trained model's weights *and* has access to the training data, how cheaply can
they retrain back to full performance? This project measures the Pareto frontier of
**compression ratio** vs. **fraction of original training cost needed to recover**.

Model + data: **ResNet-20** (CIFAR variant, ~270k params) on **CIFAR-10**.

### Key definitions

- **Compression ratio** = `compressed_bytes / baseline_fp32_bytes`, counting only the
  conv/linear weight tensors that get compressed. Honest byte accounting (index/codebook
  overhead included). BatchNorm, biases, and 1-D tensors stay dense and are excluded from
  both numerator and denominator. Lower ratio = more compressed.
- **Recovery threshold** = test accuracy ≥ `baseline_test_acc − 0.5pp` on the full 10,000
  image CIFAR-10 test split (no subsetting).
- **Retraining budget cap** = a run is killed and marked **DNR** (did not recover) if it
  does not hit the threshold within **10%** of baseline training steps.
- **Recovery fraction** = `recovery_steps / baseline_steps` (∈ (0, 0.10]; DNR plotted at 0.10).
- **Retraining mode** = dense-from-init: reconstruct the best dense approximation from the
  compressed bits, use it as the initialization, and train with the same optimizer/schedule.
  No sparsity-mask preservation.

See **[BRIEF.md](BRIEF.md)** for the full experiment specification, and **[STATUS.md](STATUS.md)**
for live progress. Findings land in **[RESULTS.md](RESULTS.md)** and `figures/pareto.png`.
