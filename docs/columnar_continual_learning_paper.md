# Columnar Predictive-Causal Networks for Continual Learning:
# Experimental Lessons from Split-MNIST and a Path Beyond It

## Abstract

This paper synthesizes two local theory documents, `columnar_bayesian_causal_coding_v3.pdf` and `v_20_b_vs_v_18.pdf`, with the recent `experimental/columnar` additions in the `FabricPC` repository. The resulting picture is a practical research program for continual learning based on columnar neural architectures that combine predictive coding, causal coding, replay-backed support selection, and conservative anti-overwrite mechanisms. The strongest current empirical evidence remains in the Split-MNIST regime, where archived five-task runs in this repository reach final average accuracies of `0.9227`, `0.9240`, and `0.9181` with corresponding average forgetting of `0.0874`, `0.0817`, and `0.0951`, closely matching the earlier V18-era result summarized in the PDF (`0.841` mean seen-task accuracy, `0.089` forgetting) while improving final retained accuracy. More recent FabricPC smoke and parity runs show that the V20.2b-style control stack is now operational: replay, audit-guided causal scoring, TransWeave bookkeeping, and task-start transition autotuning have all been ported into modular code, and the causal guidance path can become active (`mix_gate` rising to `0.2711` and `0.3500` in a three-task causal smoke run). The main conclusion is that the columnar causal-coding hypothesis is now supported as an engineering direction, but not yet validated as a general continual-learning solution. The next decisive step is to pair the current control regime with a genuinely columnar host architecture and to evaluate it on harder benchmarks, multi-seed protocols, and richer temporal domains.

## 1. Introduction

The central claim in `columnar_bayesian_causal_coding_v3.pdf` is that continual learning becomes more tractable when modularity is imposed at two levels. At the coarse scale, Bayesian routing among columns determines which modules are active and when reuse should give way to forking. At the fine scale, each column contains a sparse predictive-coding micrograph whose shell structure supports pruning, inhibition, and consolidation. The companion note `v_20_b_vs_v_18.pdf` translates this architectural thesis into a concrete experimental lineage: V18 established a stable teacher-first scaffold for Split-MNIST, while V20.2b sought to let replay-backed online proposals matter without sacrificing retention.

The current `FabricPC` port is important because it moves these ideas out of monolithic notebooks and into a reusable predictive-coding library with a modular continual-learning layer. The recent `experimental/columnar` branch extends that port with batched support and causal logic, a notebook-parity regression harness, explicit V18-like and V20.2b-like profiles, and an opt-in transition-autotuning mechanism for shell promotion and demotion. This makes it possible to ask a more serious question than whether the original notebooks were suggestive: what, exactly, has been demonstrated, and what is still missing before columnar predictive-causal networks can claim broad continual-learning relevance?

## 2. Repository-Specific Architectural Synthesis

Across the two PDFs and the current code, the same design pattern appears repeatedly.

First, support selection is treated as a structured control problem rather than as ordinary weight adaptation. In the repository this appears in `fabricpc/continual/support.py`, which maintains support banks, replay-backed support proposals, selector trust, and conservative replacement logic. The recent branch makes these operations more scalable by vectorizing column statistics, challenger scoring, and replay sampling.

Second, causal coding is implemented not as a global theorem only, but as a practical feature-and-ranking stack. In `fabricpc/continual/causal.py`, the system learns from audit rows, tracks predictor agreement, and exports quantities that can influence support choice when evidence is strong enough. The recent branch strengthens this path by batching predictor bookkeeping and by making parity metrics include mean causal mix-gate activity.

Third, shell structure is now linked to runtime control. The task-start autotuning logic in `fabricpc/continual/trainer.py` evaluates candidate demotion and promotion thresholds using short rollouts and transport-style diagnostics. This is a modest but important step toward the paper’s broader claim that microcolumn hygiene should be adjusted using task-local evidence rather than fixed once and left untouched.

Taken together, the port does not yet realize the full columnar Bayesian causal coding network described in the theory PDF, but it does implement a recognizable approximation of the control regime: replay may propose, causal estimates may rank, shell transitions may adapt, and exact or near-exact audits remain the final authority.

## 3. Experimental Evidence Available in This Repository

### 3.1 Baseline from the V18/V20 note

The PDF `v_20_b_vs_v_18.pdf` reports that the V18 system reached mean seen-task accuracy around `0.841` after five Split-MNIST tasks with mean forgetting around `0.089`. That result matters because it established the original teacher-first scaffold and showed that support diversification and local repair could work without immediate collapse.

### 3.2 Archived five-task FabricPC runs

The strongest directly inspectable continual-learning artifacts in the repository are the archived Split-MNIST result directories under `results/split_mnist/`. Reading the saved `accuracy_matrix.csv` files gives the following final accuracies after task 4:

| Run directory | Final average accuracy | Average forgetting | BWT |
|---|---:|---:|---:|
| `split_mnist_seed42_20260407_200236` | `0.9227` | `0.0874` | `-0.0874` |
| `split_mnist_seed42_20260407_200545` | `0.9240` | `0.0817` | `-0.0817` |
| `split_mnist_seed42_20260407_200748` | `0.9181` | `0.0951` | `-0.0951` |

Across these three archived runs, final average accuracy is `0.9216 +/- 0.0031` and average forgetting is `0.0881 +/- 0.0067` over seeds/reruns present locally. Two points stand out.

1. The forgetting level is nearly identical to the V18 number in the PDF, which suggests that the port preserved the anti-forgetting character of the earlier regime.
2. Final retained accuracy is materially higher than the PDF’s V18 summary, suggesting that the ported implementation and its support-control scaffolding are at least competitive with the earlier notebook lineage on this benchmark.

At the same time, the archived summaries show an important limitation: `causal_selector_mix_gate` remained `0.0` in these five-task runs. The system accumulated causal examples and some predictor correlation, but the gating threshold was never crossed strongly enough for causal guidance to visibly steer support decisions. These runs therefore support the value of the teacher-first scaffold more strongly than they support the causal-guidance layer itself.

### 3.3 Smoke and parity evidence for the later control stack

The repository documentation in `docs/fabricpc_port_summary.md` records two newer smoke experiments run against the current code:

1. `examples/split_mnist_continual.py --quick-smoke` completed five tasks with mean test accuracy about `0.9879` and average forgetting about `0.0748`, but causal guidance remained effectively inactive.
2. `examples/split_mnist_causal.py --quick-smoke --num-tasks 3` completed three tasks with final mean accuracy about `0.9556`, average forgetting about `0.0574`, and visibly active causal gating, with `mix_gate` rising from `0.0` on task 0 to `0.2711` and `0.3500` on tasks 1 and 2.

The parity harness in `benchmarks/notebook_parity_baselines.json` provides a consistent V20.2b-like three-task reference point: final mean accuracy `0.9543`, average forgetting `0.0590`, and mean causal mix gate `0.1992`. This is not a publication-grade benchmark, but it is useful evidence that the later replay-assisted, conservative-selection regime is now encoded in regression-tested repository machinery rather than left as notebook folklore.

## 4. Interpretation

The current evidence supports three claims.

First, the columnar continual-learning idea has survived translation from theory PDF to notebook lineage to modular code. The repository now contains working implementations of replay-backed support selection, exact-audit-style supervision, causal challenger ranking, and shell-level transport heuristics.

Second, the most robust empirical win so far is not that causal coding has clearly surpassed simpler methods, but that the teacher-first columnar control scaffold is stable and repeatable on Split-MNIST. The five-task archived runs show a narrow forgetting band centered almost exactly on the earlier V18 number.

Third, the causal and predictive extensions are promising but not yet decisive. When the gating path is inactive, the system behaves like a competent support-managed replay learner. When the gating path turns on in the newer causal smoke regime, the behavior is encouraging, but the evidence is still limited to a short three-task run on an easy benchmark. In other words, the code now demonstrates mechanism liveness more clearly than benchmark superiority.

## 5. Path Forward Beyond Split-MNIST

The repository’s own strategic notes in `docs/future_strategic_improvements.md` point in the right direction, but the path can be sharpened around the columnar causal-coding hypothesis itself.

### 5.1 Build a genuine columnar host architecture

The present examples still use a relatively simple feedforward graph. This undercuts the theory, because the support-selection stack is steering a host network that is weaker and less explicitly modular than the CBCCN proposal assumes. The next architecture should expose columns, shells, and composer pathways as first-class graph builders rather than as indirect conventions.

### 5.2 Make causal guidance predict retention, not only immediate gain

The current causal stack is closer to a guarded challenger ranker than to a full continual-learning critic. The next version should predict at least three quantities separately: current-task gain, old-task loss, and uncertainty. That would align the learned controller with the CCL view that low-commutator updates matter more than locally attractive but globally destructive swaps.

### 5.3 Move from replay as memory to replay as intervention design

Replay rows should no longer be sampled only for generic retention. They should be organized around support sensitivity, audit disagreement, and counterfactual discriminativeness. That would make replay a source of approximate interventions, bringing practice closer to the causal-coding motivation in the theory PDF.

### 5.4 Expand benchmarks along structure, not only difficulty

The next test suite should include:

- Permuted-MNIST, to separate task-identity reuse from class-specific feature reuse.
- Split Fashion-MNIST, to test whether the current support logic survives a less trivial visual domain.
- Split CIFAR-10 or CIFAR-100, to test whether shell-level hygiene still helps when representations are less linearly separable.
- A temporal or autoregressive benchmark, where predictive coding and causal routing can be tested in the setting they were originally meant to address.

The longer-term target should be a multiresolution sequence model in which columns or microcolumns correspond to resolution bands, matching the transformer-oriented discussion in `columnar_bayesian_causal_coding_v3.pdf`.

### 5.5 Raise the standard of experimental claims

The branch already adds parity regression and CI, but publication-level evidence will require:

- multi-seed reporting with confidence intervals;
- direct ablations for replay, causal ranking, TransWeave, and transition autotuning;
- support-diversity and swap-acceptance statistics, not accuracy alone;
- comparisons against naive fine-tuning, replay-only, and regularization baselines;
- explicit reporting of when `mix_gate` stays inactive, because that determines whether the causal subsystem actually contributed.

## 6. Continual Learning as Column Management

A useful conceptual reframing emerges from the local materials. Continual learning in this family of systems is not best understood as generic parameter protection. It is better understood as column management under uncertainty: selecting supports, estimating overlap, deciding when to reuse or repair, and protecting the internal causal cleanliness of the selected columns. Predictive coding supplies the local inference dynamics; causal coding supplies the modularity criterion; replay and audit make the control loop data-driven; shell transitions supply a mechanism for fast structural adaptation without global rewriting.

That reframing suggests why Split-MNIST is only a first step. On such a small benchmark, support management can look successful even when the host architecture is underexpressive and the causal gate is mostly dormant. Harder domains are needed not only because they are more difficult, but because they force the system to prove that its columnar abstractions carry real semantic load.

## 7. Conclusion

The local evidence supports a cautious positive conclusion. The `FabricPC` repository now contains a credible experimental substrate for columnar predictive-causal continual learning. Archived Split-MNIST runs show stable forgetting around `0.088`, consistent with the earlier V18 story and paired with improved retained accuracy around `0.922`. The newer branch further shows that replay-assisted causal support guidance, parity-tested control profiles, and shell-transition autotuning are no longer purely conceptual.

What has not yet been shown is that these ideas scale beyond a carefully managed toy domain. The decisive next phase is therefore clear: instantiate a truly columnar architecture, make causal guidance explicitly retention-aware, and evaluate it on harder multi-seed continual-learning and sequence benchmarks. If those steps succeed, the columnar predictive-causal framework will move from an interesting design philosophy to a serious candidate for general continual learning.

## References

1. `../columnar_bayesian_causal_coding_v3.pdf`
2. `../v_20_b_vs_v_18.pdf`
3. `docs/fabricpc_port_summary.md`
4. `docs/future_strategic_improvements.md`
5. `benchmarks/notebook_parity_baselines.json`
6. `results/split_mnist/split_mnist_seed42_20260407_200236/accuracy_matrix.csv`
7. `results/split_mnist/split_mnist_seed42_20260407_200545/accuracy_matrix.csv`
8. `results/split_mnist/split_mnist_seed42_20260407_200748/accuracy_matrix.csv`
