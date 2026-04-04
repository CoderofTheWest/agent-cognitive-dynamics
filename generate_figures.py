#!/usr/bin/env python3
"""Generate all data figures for the temporal dynamics paper."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from datetime import datetime
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

ENTROPY_PATH = Path("/Users/clint/robot/storage/entropy-monitor.jsonl")
RESULTS_A = Path("/Users/clint/robot/clint_experiment_a_results.json")
OUT = Path("/Users/clint/robot/paper")


def load_entries():
    entries = []
    with open(ENTROPY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def parse_ts(ts):
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def get_bursts(entries):
    boundaries = [0]
    for i in range(1, len(entries)):
        t1 = parse_ts(entries[i-1]['timestamp'])
        t2 = parse_ts(entries[i]['timestamp'])
        if (t2 - t1).total_seconds() > 60:
            boundaries.append(i)
    bursts = []
    for i in range(len(boundaries)):
        s = boundaries[i]
        e = boundaries[i+1] if i+1 < len(boundaries) else len(entries)
        bursts.append((s, e))
    return bursts


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Experiment A results — bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_results_bar():
    with open(RESULTS_A) as f:
        results = json.load(f)

    pf = results['per_feature']
    # Sort by improvement over persistence
    items = [(name, data['a_vs_persist_pct'], data['autocorrelation'])
             for name, data in pf.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    names = [x[0] for x in items]
    improvements = [x[1] for x in items]
    autocorrs = [x[2] for x in items]

    # Color by AC regime
    colors = []
    for ac in autocorrs:
        if ac > 0.9:
            colors.append('#d62728')  # red — persistence wins
        elif ac > 0.3:
            colors.append('#2ca02c')  # green — moderate AC
        else:
            colors.append('#1f77b4')  # blue — low AC

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, improvements, color=colors, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace('_', r'\_') for n in names], fontsize=7, fontfamily='monospace')
    ax.set_xlabel('Improvement over persistence baseline (%)')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='AC(1) < 0.3'),
        Patch(facecolor='#2ca02c', label='0.3 < AC(1) < 0.9'),
        Patch(facecolor='#d62728', label='AC(1) > 0.9'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)
    ax.set_title('MLP Prediction Improvement over Persistence Baseline')

    plt.tight_layout()
    fig.savefig(OUT / 'fig_results_bar.pdf')
    plt.close()
    print("Saved fig_results_bar.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Autocorrelation plot — lag 1 through 10
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_autocorrelation(entries):
    bursts = get_bursts(entries)
    # Get final tick per burst
    turn_entries = [entries[e-1] for (s, e) in bursts]

    # Extract key features
    features_to_plot = {
        'entropy_total': [e['entropy']['total'] for e in turn_entries],
        'shannon': [e['entropy']['breakdown'].get('shannon', None) for e in turn_entries],
        'recursiveMeta': [e['entropy']['breakdown'].get('recursiveMeta', 0) for e in turn_entries],
        'tension_type': [e.get('context', {}).get('tensionType', '') for e in turn_entries],
        'entropy_debt': [e.get('intervention', {}).get('entropyDebt', 0) for e in turn_entries],
        'qualityDecay': [e['entropy']['breakdown'].get('qualityDecay', 0) for e in turn_entries],
        'emotional': [e['entropy']['breakdown'].get('emotional', 0) for e in turn_entries],
    }

    # Encode tension_type as int
    tension_types = sorted(set(features_to_plot['tension_type']))
    tmap = {t: i for i, t in enumerate(tension_types)}
    features_to_plot['tension_type'] = [tmap.get(t, 0) for t in features_to_plot['tension_type']]

    # Handle None in shannon
    shannon = features_to_plot['shannon']
    med = np.nanmedian([x for x in shannon if x is not None])
    features_to_plot['shannon'] = [x if x is not None else med for x in shannon]

    # Infer sessions for turn-level
    turn_timestamps = [parse_ts(e['timestamp']) for e in turn_entries]
    session_ids = [0]
    sid = 0
    for i in range(1, len(turn_timestamps)):
        gap = (turn_timestamps[i] - turn_timestamps[i-1]).total_seconds() / 60
        if gap > 30:
            sid += 1
        session_ids.append(sid)

    max_lag = 10
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    styles = {
        'entropy_total': ('-', '#1f77b4', 'entropy_total'),
        'shannon': ('-', '#ff7f0e', 'shannon'),
        'recursiveMeta': ('-', '#2ca02c', 'recursiveMeta'),
        'tension_type': ('--', '#d62728', 'tension_type'),
        'entropy_debt': ('--', '#9467bd', 'entropy_debt'),
        'qualityDecay': (':', '#8c564b', 'qualityDecay'),
        'emotional': (':', '#e377c2', 'emotional'),
    }

    for fname, vals in features_to_plot.items():
        vals = np.array(vals, dtype=float)
        sids = np.array(session_ids)
        ls, color, label = styles[fname]

        acs = []
        for lag in range(1, max_lag + 1):
            # Only within sessions
            x_vals, y_vals = [], []
            for i in range(len(vals) - lag):
                if sids[i] == sids[i + lag]:
                    x_vals.append(vals[i])
                    y_vals.append(vals[i + lag])
            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals)
            if len(x_vals) > 10 and np.std(x_vals) > 1e-8 and np.std(y_vals) > 1e-8:
                ac = np.corrcoef(x_vals, y_vals)[0, 1]
            else:
                ac = 0.0
            acs.append(ac)

        ax.plot(range(1, max_lag + 1), acs, ls, color=color, label=label,
                linewidth=1.2, markersize=3, marker='o')

    ax.set_xlabel('Lag (turns)')
    ax.set_ylabel('Autocorrelation')
    ax.set_xticks(range(1, max_lag + 1))
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
    ax.legend(fontsize=6.5, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    ax.set_ylim(-0.15, 1.05)
    ax.set_title('Turn-Level Autocorrelation by Lag')

    plt.tight_layout()
    fig.savefig(OUT / 'fig_autocorrelation.pdf')
    plt.close()
    print("Saved fig_autocorrelation.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Burst length distribution histogram
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_burst_histogram(entries):
    bursts = get_bursts(entries)
    lengths = np.array([e - s for s, e in bursts])

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bins = list(range(1, 12)) + list(range(12, 62, 2)) + list(range(62, 260, 5))
    counts, edges, patches = ax.hist(lengths, bins=bins, color='#1f77b4',
                                      edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Burst length (ticks per turn)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Processing Burst Lengths')

    peak_height = counts.max()

    # Annotate fast resolution — arrow pointing down to the ~6 tick peak
    ax.annotate('Fast resolution\n($\\sim$6 ticks, 59.5\\%)',
                xy=(6, counts[5]),  # point at the 5-6 bin peak
                xytext=(18, peak_height * 0.85),
                fontsize=7, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4e6f1', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=0.8))

    # Annotate extended deliberation — arrow pointing down to the ~50 tick peak
    # Find the bin around 50 ticks
    bin_50_idx = np.searchsorted(edges, 50) - 1
    ax.annotate('Extended deliberation\n($\\sim$50 ticks, 39.2\\%)',
                xy=(50, counts[min(bin_50_idx, len(counts)-1)]),
                xytext=(80, peak_height * 0.7),
                fontsize=7, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=0.8))

    # Annotate the empty intermediate range
    ax.annotate('Nearly empty\nintermediate range',
                xy=(15, 5),
                xytext=(35, peak_height * 0.4),
                fontsize=6, ha='center', color='#555555',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))

    plt.tight_layout()
    fig.savefig(OUT / 'fig_burst_distribution.pdf')
    plt.close()
    print("Saved fig_burst_distribution.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Burst length vs tension transitions scatter
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_burst_scatter(entries):
    bursts = get_bursts(entries)

    lengths = []
    transitions = []
    for s, e in bursts:
        n = e - s
        tensions = [entries[j].get('context', {}).get('tensionType', '') for j in range(s, e)]
        trans = sum(1 for t in range(1, len(tensions)) if tensions[t] != tensions[t-1])
        lengths.append(n)
        transitions.append(trans)

    lengths = np.array(lengths)
    transitions = np.array(transitions)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    # Color by regime
    short = lengths < 10
    long_mask = lengths >= 20
    mid = ~short & ~long_mask

    ax.scatter(lengths[short], transitions[short], s=4, alpha=0.3,
               color='#1f77b4', label=f'Fast (<10, n={short.sum()})', zorder=2)
    ax.scatter(lengths[mid], transitions[mid], s=8, alpha=0.5,
               color='#2ca02c', label=f'Intermediate (n={mid.sum()})', zorder=3)
    ax.scatter(lengths[long_mask], transitions[long_mask], s=4, alpha=0.3,
               color='#d62728', label=f'Extended ($\\geq$20, n={long_mask.sum()})', zorder=2)

    ax.set_xlabel('Burst length (ticks)')
    ax.set_ylabel('Tension type transitions')
    ax.set_title(f'Burst Length vs. Tension Transitions (Spearman $\\rho$ = 0.938)')
    ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    fig.savefig(OUT / 'fig_burst_vs_transitions.pdf')
    plt.close()
    print("Saved fig_burst_vs_transitions.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: System architecture diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.2))

    def draw_panel(ax, title):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7.5)
        ax.axis('off')
        ax.set_title(title, fontsize=9, fontweight='bold', pad=8)

    def box(ax, x, y, w, h, text, color='#d4e6f1', fontsize=7, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor='#2c3e50', linewidth=0.7)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, weight=weight)

    def arrow(ax, x1, y1, x2, y2, color='#2c3e50', style='->', lw=0.8):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    # ── Panel A: Research Server (monolithic) ──
    ax = axes[0]
    draw_panel(ax, '(a) Research Server (Oct 2025 \u2013 Feb 2026)')

    # Top: user/response
    box(ax, 0.3, 6.3, 1.8, 0.8, 'User Input', '#aed6f1', bold=True)
    box(ax, 7.5, 6.3, 2.2, 0.8, 'Response', '#aed6f1', bold=True)

    # Monolith block
    box(ax, 0.3, 2.5, 9.4, 3.5, '', '#f7f9f9', fontsize=6)
    ax.text(5.0, 5.75, 'server.js  (~9,900 lines)', ha='center',
            fontsize=7, style='italic', color='#7f8c8d')

    # Inside monolith
    box(ax, 0.6, 4.5, 2.8, 0.9, 'constructPrompt\nOptimized', '#d5f5e3', fontsize=6.5)
    box(ax, 4.0, 4.5, 2.2, 0.9, 'LLM call\n(stateless)', '#fadbd8', fontsize=6.5, bold=True)
    box(ax, 6.8, 4.5, 2.6, 0.9, 'Post-processing\n+ response', '#d5f5e3', fontsize=6.5)

    arrow(ax, 1.5, 6.3, 1.5, 5.4, lw=0.7)
    arrow(ax, 3.4, 4.95, 4.0, 4.95, lw=0.7)
    arrow(ax, 6.2, 4.95, 6.8, 4.95, lw=0.7)
    arrow(ax, 8.6, 5.4, 8.6, 6.3, lw=0.7)

    # Identity subsystem inside monolith
    box(ax, 0.6, 2.8, 2.5, 0.9, 'identityEvolution\nCodeAligned', '#fdebd0', fontsize=6)
    box(ax, 3.4, 2.8, 2.5, 0.9, 'identityIntegration\nCodeAligned', '#fdebd0', fontsize=6)
    box(ax, 6.2, 2.8, 3.2, 0.9, 'Entropy monitor\n+ tension + SEAL', '#fdebd0', fontsize=6)

    ax.text(5.0, 3.95, 'Identity + Metacognitive Layer', ha='center',
            fontsize=6.5, style='italic', color='#7f8c8d')

    # Feedback arrows
    arrow(ax, 7.8, 4.5, 7.8, 3.7, color='#e74c3c', lw=0.7)
    arrow(ax, 2.0, 3.7, 2.0, 4.5, color='#27ae60', lw=0.7)

    # Bottom: persistent files + log
    box(ax, 0.3, 0.8, 3.8, 0.9, 'Persistent Files\n(SOUL, HEARTBEAT,\nAGENTS, TOOLS, CHRIS)',
        '#e8daef', fontsize=6)
    box(ax, 4.5, 0.8, 5.2, 0.9, 'entropy-monitor.jsonl\n68,110 ticks \u2192 2,992 turns',
        '#d4efdf', fontsize=6)

    arrow(ax, 2.2, 2.8, 2.2, 1.7, color='#8e44ad', lw=0.6)
    arrow(ax, 7.8, 2.8, 7.1, 1.7, color='#27ae60', lw=0.6)

    # ── Panel B: OpenClaw (modular) ──
    ax = axes[1]
    draw_panel(ax, '(b) OpenClaw (Feb 2026 \u2013)')

    # Top: user/response
    box(ax, 0.2, 6.3, 1.6, 0.8, 'User\nInput', '#aed6f1', fontsize=6, bold=True)
    box(ax, 8.0, 6.3, 1.7, 0.8, 'Response', '#aed6f1', fontsize=6, bold=True)

    # Gateway
    box(ax, 2.5, 6.3, 4.8, 0.8, 'OpenClaw Gateway', '#d5f5e3', fontsize=6.5)
    arrow(ax, 1.8, 6.7, 2.5, 6.7, lw=0.7)
    arrow(ax, 7.3, 6.7, 8.0, 6.7, lw=0.7)

    # LLM
    box(ax, 3.0, 5.0, 3.8, 0.7, 'LLM (stateless)', '#fadbd8', fontsize=6, bold=True)
    arrow(ax, 4.9, 6.3, 4.9, 5.7, lw=0.7)
    arrow(ax, 4.9, 5.0, 4.9, 4.5, color='#e74c3c', lw=0.7)

    # Core plugin pipeline — 2 rows of 3
    ax.text(5.0, 4.4, 'Core Identity Plugins', ha='center',
            fontsize=6, style='italic', color='#7f8c8d')
    core_row1 = [('stability', 0.3), ('metabolism', 3.4), ('contemplation', 6.5)]
    core_row2 = [('crystallization', 0.3), ('continuity', 3.4), ('truth', 6.5)]
    for name, x in core_row1:
        box(ax, x, 3.6, 2.8, 0.55, name, '#fdebd0', fontsize=5.5)
    for name, x in core_row2:
        box(ax, x, 2.9, 2.8, 0.55, name, '#fdebd0', fontsize=5.5)

    # Inject back up
    arrow(ax, 1.7, 4.15, 3.0, 5.3, color='#27ae60', lw=0.7)
    ax.text(1.5, 4.8, 'inject', fontsize=5, color='#27ae60', style='italic')

    # Supporting plugins row
    ax.text(5.0, 2.35, 'Supporting Plugins', ha='center',
            fontsize=6, style='italic', color='#7f8c8d')
    plugins2 = [('graph', 0.3), ('threads', 2.3), ('nightshift', 4.3),
                ('embodiment', 6.3), ('coaching', 8.3)]
    for name, x in plugins2:
        box(ax, x, 1.6, 1.7, 0.55, name, '#fef9e7', fontsize=5)

    # Bottom: persistent files + event buses
    box(ax, 0.2, 0.3, 4.6, 0.65, 'Workspace Files\n(SOUL, HEARTBEAT, growth-vectors)',
        '#e8daef', fontsize=5)
    box(ax, 5.1, 0.3, 4.6, 0.65, 'Event Buses\n(inter-plugin communication)',
        '#d4e6f1', fontsize=5)

    arrow(ax, 2.5, 1.8, 2.5, 0.95, color='#8e44ad', lw=0.6)
    arrow(ax, 7.4, 1.8, 7.4, 0.95, color='#2980b9', lw=0.6)

    plt.tight_layout()
    fig.savefig(OUT / 'fig_architecture.pdf')
    plt.close()
    print("Saved fig_architecture.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Loading data...")
    entries = load_entries()

    print("Generating figures...")
    fig1_architecture()
    fig2_results_bar()
    fig3_autocorrelation(entries)
    fig4_burst_histogram(entries)
    fig5_burst_scatter(entries)
    print("Done.")
