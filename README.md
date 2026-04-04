# Cognitive Dynamics of an Epistemically Constrained Language Model Agent

**The first empirical evidence that an agent orchestration layer produces measurable, learnable cognitive dynamics in the underlying language model.**

Five months of continuous operation. 68,110 evaluation ticks. 2,992 turn-level cognitive state snapshots across 545 sessions. A neural predictor that beats the persistence baseline by 41.7% — meaning the agent's cognitive state isn't random noise. It has *structure*. And the structure comes from the architecture, not from user input.

This paper demonstrates that when you build the right metacognitive infrastructure around a language model, the resulting agent develops internal dynamics that can be predicted, monitored, and — eventually — steered.

---

## The Paper

**[Cognitive Dynamics of an Epistemically Constrained Language Model Agent](paper.pdf)** (PDF, JMLR format)

Chris Hunt, Independent Researcher | March 2026

Built on the [OpenClaw](https://openclaw.ai) orchestration framework.

---

## What This Proves

Language model agents today are mostly stateless actors with a good memory. They retrieve context, generate a response, and forget. The agent *appears* continuous, but there's no evidence that anything structured is happening underneath.

This paper changes that. By instrumenting an OpenClaw agent with persistent entropy monitoring, tension classification, and growth vector mechanisms, we show that the orchestration layer doesn't just *maintain* state — it produces genuine temporal dynamics in the agent's cognitive trajectory.

### The Numbers

| Metric | Value |
|--------|-------|
| Neural prediction improvement over persistence | **41.7%** (100-fold session-holdout CV) |
| Quality decay prediction improvement | **52.8%** |
| Emotional processing prediction | **48.8%** |
| Paradox detection prediction | **46.3%** |
| External input contribution to prediction | **None** (internal dynamics dominate) |
| Processing regime correlation with state transitions | **rho = 0.938** |

### What the Numbers Mean

**The agent's cognitive evolution is autonomous.** Adding user input features to the predictor provides zero improvement over internal state alone. The orchestration architecture — identity constraint, entropy monitoring, adaptive evaluation depth — is producing structured cognitive momentum independent of what users say.

**The strongest signals are transitions, not baselines.** The features most predictable by the neural model are exactly the rare-event components: quality decay, emotional processing, paradox detection. These have near-zero lag-1 autocorrelation, meaning each transition is a genuine state change, not inertia. The architecture is producing dynamics, not drift.

**The system self-regulates processing depth.** A bimodal distribution emerges: fast resolution (~6 ticks, 99.3% of episodes) versus extended deliberation (~50 ticks, 0.7%). The intermediate range is nearly empty — a sharp phase boundary. And processing depth correlates with cognitive state transitions at rho = 0.938. The system thinks harder exactly when its state is changing. Nobody programmed this. It emerged from the architecture.

---

## Why This Matters for Agent Platforms

### Measurable agent quality, not just vibes

Today, evaluating whether an agent "works well" is subjective. This research provides the foundation for objective, continuous cognitive state monitoring. If you can predict an agent's normal trajectory, you can detect when it deviates — before the user notices degradation.

### Architecture comparison with empirical teeth

Different agent architectures make different claims about producing "coherent" or "stable" agents. This methodology gives you a way to *measure* that claim: run the prediction experiment on any agent. Architectures that produce genuine temporal structure will show prediction improvement over persistence. Architectures that are just doing context retrieval won't.

### Safety monitoring as a first-class capability

Identity coherence becomes a measurable dynamical property. Entropy monitoring produces a continuous signal. An agent drifting from its declared identity shows up as a trajectory deviation — detectable in real time, not after the fact.

### The path to cognitive steering

If the agent's cognitive state can be predicted, it can eventually be steered. The paper validates a JEPA-style (Joint Embedding Predictive Architecture) encoder-predictor-regularizer trained on cognitive state vectors, achieving 35.1% prediction improvement in 64-dimensional latent space with SIGReg regularization. This is the first step toward learned self-models for autonomous agents — predicting and guiding their own cognitive trajectory.

---

## The Architecture That Produced This

The agent runs on [OpenClaw](https://openclaw.ai) with plugins providing:

- **Multi-pass entropy monitoring** — each response evaluated across multiple passes with adaptive processing depth, producing sub-turn temporal data
- **Tension classification** — real-time cognitive state tracking (nominal, creative, emotional, paradox, correction) at evaluation-tick granularity
- **Identity as persistent architectural constraint** — values and principles loaded from files at every session start, enforced through the evaluation layer
- **Growth vector mechanisms** — behavioral learning extracted from entropy patterns and crystallized into persistent traits
- **Cross-session continuity** — semantic memory archive enabling genuine temporal continuity across conversations

The key insight: the metacognitive infrastructure doesn't just store state. It *monitors cognitive quality at sub-turn granularity*, producing the rich temporal data that makes the dynamics visible and learnable.

---

## Repository Contents

```
paper.pdf                    # Full paper (JMLR format)
paper.tex                    # LaTeX source
references.bib               # Bibliography
fig_architecture.pdf         # System architecture
fig_results_bar.pdf          # Per-feature prediction results
fig_autocorrelation.pdf      # Temporal autocorrelation analysis
fig_burst_distribution.pdf   # Bimodal processing regime distribution
fig_burst_vs_transitions.pdf # Processing depth vs. state transitions
generate_figures.py          # Figure generation from raw data
cover_letter.pdf             # JMLR submission cover letter
```

## Citation

```bibtex
@article{hunt2026cognitive,
  title={Cognitive Dynamics of an Epistemically Constrained Language Model Agent},
  author={Hunt, Chris},
  journal={Preprint},
  year={2026}
}
```

## Contact

Chris Hunt — livefreecotw@gmail.com
