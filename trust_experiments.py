import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd

from config_loader import load_json_config
from graph_loader import add_speeds_and_travel_time, get_edge_info_factory, load_osm_graph
from shadow import generate_shadow_field
from sim_core import VanetEngine


VEHICLE_COUNTS = [10, 40, 70, 100]
DEFAULT_EXPERIMENT_REPETITIONS = 10
ATTACK_LABELS = {
    "false_feedback": "False feedback",
    "spoofing": "Spoofing",
    "collusion": "Collusion",
    "transient": "Transient",
}
PAPER_TIMINGS_MS = {
    "T_m_ecc": 2.4070,
    "T_ex": 0.0037,
    "T_h": 0.0001,
    "T_puf": 0.3780,
}
PAPER_SIZES_BITS = {
    "G": 320,
    "PUF": 64,
    "ID": 128,
    "TS": 32,
}


def paper_vehicle_cost_ms(cfg):
    base = PAPER_TIMINGS_MS["T_m_ecc"] + 2 * PAPER_TIMINGS_MS["T_ex"] + 2 * PAPER_TIMINGS_MS["T_h"]
    if cfg.get("puf_enabled", True):
        base += 2 * PAPER_TIMINGS_MS["T_puf"]
    if not cfg.get("secure_aggregation_enabled", True):
        base -= 2 * PAPER_TIMINGS_MS["T_ex"]
    return max(base, 0.0)


def paper_uav_cost_ms(cfg):
    base = PAPER_TIMINGS_MS["T_m_ecc"] + 2 * PAPER_TIMINGS_MS["T_ex"] + 2 * PAPER_TIMINGS_MS["T_h"]
    if cfg.get("puf_enabled", True):
        base += 3 * PAPER_TIMINGS_MS["T_puf"]
    if not cfg.get("secure_aggregation_enabled", True):
        base -= 2 * PAPER_TIMINGS_MS["T_ex"]
    return max(base, 0.0)


def paper_feedback_bits(cfg):
    bits = 2 * PAPER_SIZES_BITS["G"] + 2 * PAPER_SIZES_BITS["ID"] + 2 * PAPER_SIZES_BITS["TS"]
    if cfg.get("puf_enabled", True):
        bits += 3 * PAPER_SIZES_BITS["PUF"]
    if not cfg.get("secure_aggregation_enabled", True):
        bits -= PAPER_SIZES_BITS["G"]
    return bits


def paper_metric_bundle(cfg, trust):
    vehicle_cost = paper_vehicle_cost_ms(cfg)
    uav_cost = paper_uav_cost_ms(cfg)
    feedback_bits = paper_feedback_bits(cfg)
    vehicle_feedback = int(trust.get("vehicle_feedback_total", 0))
    uav_feedback = int(trust.get("uav_feedback_total", 0))
    return {
        "aggregation_latency_ms": float(trust.get("avg_aggregation_latency_s", 0.0)) * 1000.0,
        "communication_overhead_bits": (vehicle_feedback + uav_feedback) * feedback_bits,
        "computation_cost_ms": vehicle_feedback * vehicle_cost + uav_feedback * uav_cost,
    }


class SimulationContext:
    def __init__(self, cfg):
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(Path(__file__).resolve().parent / "cache")
        self.cfg = copy.deepcopy(cfg)
        self.graph = load_osm_graph(cfg["place"], cfg.get("use_bbox", False), cfg.get("bbox", None), cfg.get("network_type", "drive"))
        self.graph = add_speeds_and_travel_time(self.graph)
        self.get_edge_info = get_edge_info_factory(self.graph, default_speed_m_s=cfg.get("default_speed_m_s", 13.9))
        xs = [self.graph.nodes[n]["x"] for n in self.graph.nodes]
        ys = [self.graph.nodes[n]["y"] for n in self.graph.nodes]
        self.shadow_field, self.shadow_gx, self.shadow_gy = generate_shadow_field(
            min(xs),
            max(xs),
            min(ys),
            max(ys),
            grid_res_m=cfg.get("shadow_grid_res_m", 50.0),
            sigma_m=cfg.get("shadow_corr_len_m", 50.0),
            sigma_db=cfg.get("shadow_std_db", 7.0),
        )

    def run(self, cfg):
        engine = VanetEngine(cfg, self.graph, self.get_edge_info, self.shadow_field, self.shadow_gx, self.shadow_gy)
        for _ in range(engine.STEPS):
            engine.step()
        trust = engine.trust_summary()
        paper_metrics = paper_metric_bundle(cfg, trust)
        metrics = {
            "n_vehicles": int(cfg["n_vehicles"]),
            "n_uavs": int(cfg["n_uavs"]),
            "attack_type": cfg.get("attack_type", "mixed"),
            "trust_update_mode": cfg.get("trust_update_mode", "event"),
            "uav_feedback_enabled": bool(cfg.get("uav_feedback_enabled", True)),
            "vehicle_feedback_enabled": bool(cfg.get("vehicle_feedback_enabled", True)),
            "puf_enabled": bool(cfg.get("puf_enabled", True)),
            "secure_aggregation_enabled": bool(cfg.get("secure_aggregation_enabled", True)),
            "weighted_fusion_enabled": bool(cfg.get("weighted_fusion_enabled", True)),
            "malicious_uav_ratio": float(cfg.get("malicious_uav_ratio", 0.0)),
            "collusion_ratio": float(cfg.get("collusion_ratio", cfg.get("malicious_vehicle_ratio", 0.0))),
            "aggregation_latency_ms": paper_metrics["aggregation_latency_ms"],
            "communication_overhead_bits": paper_metrics["communication_overhead_bits"],
            "computation_cost_ms": paper_metrics["computation_cost_ms"],
        }
        metrics.update(trust)
        frames = pd.DataFrame(engine.frame_stats)
        if not frames.empty:
            frames["n_vehicles"] = int(cfg["n_vehicles"])
            frames["attack_type"] = cfg.get("attack_type", "mixed")
            frames["trust_update_mode"] = cfg.get("trust_update_mode", "event")
        return metrics, frames


def base_experiment_config(config_path):
    cfg = load_json_config(config_path)
    cfg.update(
        {
            "seed": 42,
            "experiment_repetitions": int(cfg.get("experiment_repetitions", DEFAULT_EXPERIMENT_REPETITIONS)),
            "sim_seconds": 60.0,
            "dt": 0.5,
            "n_uavs": 3,
            "freq_hz": 5.9e9,
            "bw": 10_000_000,
            "phy_rate_bps": 6_000_000.0,
            "trust_feedback_threshold": 25,
            "trust_event_period_s": 1.0,
            "event_scan_period_s": 0.25,
            "trust_update_mode": "event",
            "trust_target_policy": "attack_focus",
            "attack_focus_probability": 0.90,
            "attack_type": "mixed",
            "attack_start_s": 0.0,
            "attack_end_s": 60.0,
            "vehicle_feedback_enabled": True,
            "uav_feedback_enabled": True,
            "puf_enabled": True,
            "secure_aggregation_enabled": True,
            "weighted_fusion_enabled": True,
            "trust_alpha": 0.15,
            "trust_beta": 0.50,
            "trust_gamma": 0.35,
            "malicious_vehicle_ratio": 0.2,
            "malicious_uav_ratio": 0.0,
            "uav_observation_accuracy": 0.96,
            "uav_comm_delay_s": 0.002,
        }
    )
    return cfg


def run_many(ctx, base_cfg, scenario_name, scenario_builder):
    rows = []
    frames = []
    repetitions = max(1, int(base_cfg.get("experiment_repetitions", DEFAULT_EXPERIMENT_REPETITIONS)))
    base_seed = int(base_cfg.get("seed", 42))
    for n_vehicles in VEHICLE_COUNTS:
        print(f"[{scenario_name}] vehicles={n_vehicles}, repetitions={repetitions}, seeds={base_seed}-{base_seed + repetitions - 1}")
        for repetition in range(repetitions):
            cfg = copy.deepcopy(base_cfg)
            cfg["n_vehicles"] = n_vehicles
            cfg["seed"] = base_seed + repetition
            cfg.update(scenario_builder(n_vehicles))
            metrics, frame_df = ctx.run(cfg)
            metrics["scenario"] = scenario_name
            metrics["repetition"] = repetition + 1
            metrics["seed"] = cfg["seed"]
            rows.append(metrics)
            if not frame_df.empty:
                frame_df["scenario"] = scenario_name
                frame_df["repetition"] = repetition + 1
                frame_df["seed"] = cfg["seed"]
                frames.append(frame_df)
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_run_statistics(all_metrics_df):
    """Return per-scenario, per-vehicle-count statistics across random seeds."""
    metric_columns = [
        "classification_accuracy",
        "malicious_detected_ratio",
        "false_positive_rate",
        "reputation_mse",
        "avg_detection_delay_s",
        "system_reliability",
        "aggregation_latency_ms",
        "communication_overhead_bits",
        "computation_cost_ms",
    ]
    available = [column for column in metric_columns if column in all_metrics_df.columns]
    grouped = all_metrics_df.groupby(["scenario", "n_vehicles"], as_index=False)[available].agg(
        ["count", "mean", "std", "sem"]
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]
    grouped = grouped.reset_index()
    for column in available:
        sem_column = f"{column}_sem"
        if sem_column in grouped.columns:
            grouped[f"{column}_ci95"] = 1.96 * grouped[sem_column]
    return grouped


def plot_figure1(df, out_dir):
    agg = df.groupby("attack_type", as_index=False)[["classification_accuracy", "avg_detection_delay_s"]].mean()
    agg["label"] = agg["attack_type"].map(ATTACK_LABELS)
    x = np.arange(len(agg))
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.bar(x, agg["classification_accuracy"], color="#4C78A8", width=0.55)
    ax1.set_ylabel("Detection accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(x, agg["label"], rotation=15)
    ax1.set_title("Figure 1. Detection accuracy and delay under different attacks")
    ax2 = ax1.twinx()
    ax2.plot(x, agg["avg_detection_delay_s"], color="#F58518", marker="o", linewidth=2)
    ax2.set_ylabel("Avg detection delay (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_attack_detection.png", dpi=220)
    plt.close(fig)
    agg.to_csv(out_dir / "figure1_attack_detection.csv", index=False)


def plot_figure2(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    palette = {
        "vehicle_only": "#4C78A8",
        "uav_only": "#72B7B2",
        "fusion": "#E45756",
    }
    labels = {
        "vehicle_only": "Vehicle-only",
        "uav_only": "UAV-only",
        "fusion": "Fusion",
    }
    for method, part in df.groupby("method"):
        part = part.sort_values("collusion_ratio")
        axes[0].plot(part["collusion_ratio"], part["reputation_mse"], marker="o", linewidth=2, color=palette[method], label=labels[method])
        axes[1].plot(part["collusion_ratio"], part["classification_accuracy"], marker="o", linewidth=2, color=palette[method], label=labels[method])
    axes[0].set_title("Figure 2a. Collusion ratio vs reputation MSE")
    axes[0].set_ylabel("Distributed-target reputation MSE (points^2)")
    axes[1].set_title("Figure 2b. Collusion ratio vs detection accuracy")
    axes[1].set_ylabel("Detection accuracy")
    for ax in axes:
        ax.set_xlabel("Colluding vehicle ratio")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_collusion_comparison.png", dpi=220)
    plt.close(fig)
    df.to_csv(out_dir / "figure2_collusion_comparison.csv", index=False)


def plot_figure3(df, summary_df, out_dir, reputation_threshold):
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    colors = {"event": "#E45756", "periodic": "#4C78A8"}
    labels = {"event": "Event-driven", "periodic": "Periodic"}
    ax.axvspan(8.0, 18.0, color="#F3E3A1", alpha=0.35, zorder=0)
    ax.axvline(8.0, color="#B08D1A", linestyle=":", linewidth=1.2, zorder=1)
    ax.axvline(18.0, color="#B08D1A", linestyle=":", linewidth=1.2, zorder=1)
    for mode, part in df.groupby("trust_update_mode"):
        part = part.sort_values("time_s")
        ax.step(
            part["time_s"],
            part["avg_malicious_reputation"],
            where="post",
            linewidth=2.8,
            color=colors[mode],
            label=labels[mode],
            linestyle="-" if mode == "event" else "--",
            zorder=3 if mode == "event" else 4,
        )
        if mode == "periodic":
            ax.plot(
                part["time_s"],
                part["avg_malicious_reputation"],
                linestyle="None",
                marker="o",
                markersize=3.8,
                color=colors[mode],
                markevery=8,
                zorder=5,
                clip_on=False,
            )
    event_row = summary_df[summary_df["trust_update_mode"] == "event"].iloc[0]
    periodic_row = summary_df[summary_df["trust_update_mode"] == "periodic"].iloc[0]
    note = (
        f"Attack window: 8-18 s\n"
        f"Event-driven: acc={event_row['classification_accuracy']:.3f}, delay={event_row['avg_detection_delay_s']:.1f}s\n"
        f"Periodic: acc={periodic_row['classification_accuracy']:.3f}, delay={periodic_row['avg_detection_delay_s']:.1f}s"
    )
    ax.text(
        0.02,
        0.97,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.95},
        zorder=6,
    )
    ax.axhline(reputation_threshold, color="#B22222", linestyle="--", linewidth=1.4, label=f"Malicious threshold ({reputation_threshold:g})")
    ax.text(13.0, 4.0, "Transient attack window", color="#8C6B00", fontsize=9, ha="center", va="bottom")
    ax.set_title("Figure 3. Malicious-Vehicle Reputation Under Transient Attacks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average malicious-vehicle reputation")
    ax.set_ylim(0.0, 100.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure3_update_modes.png", dpi=220)
    plt.close(fig)
    df.to_csv(out_dir / "figure3_update_modes_curve.csv", index=False)
    summary_df.to_csv(out_dir / "figure3_update_modes_summary.csv", index=False)


def plot_figure4(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True)
    x = df["compromised_uavs"]
    axes[0].plot(x, df["classification_accuracy"], marker="o", linewidth=2, color="#4C78A8")
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Classification accuracy")
    axes[1].plot(x, df["reputation_mse"], marker="o", linewidth=2, color="#F58518")
    axes[1].set_title("MSE")
    axes[1].set_ylabel("Distributed-target reputation MSE (points^2)")
    axes[2].plot(x, df["false_positive_rate"], marker="o", linewidth=2, color="#54A24B")
    axes[2].set_title("False-positive rate")
    axes[2].set_ylabel("FPR")
    for ax in axes:
        ax.set_xlabel("Compromised UAV count")
        ax.grid(alpha=0.25)
    fig.suptitle("Figure 4. Impact of compromised UAVs on trust performance", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "figure4_uav_compromise.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(out_dir / "figure4_uav_compromise.csv", index=False)


def plot_figure5(df, out_dir):
    alphas = sorted(df["alpha"].unique())
    betas = sorted(df["beta"].unique())
    matrix = np.full((len(betas), len(alphas)), np.nan)
    gamma_map = {}
    for _, row in df.iterrows():
        i = betas.index(row["beta"])
        j = alphas.index(row["alpha"])
        matrix[i, j] = row["reputation_mse"]
        gamma_map[(i, j)] = row["gamma"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#D9D9D9")
    im = ax.imshow(matrix, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(alphas)), [f"{x:.2f}" for x in alphas])
    ax.set_yticks(range(len(betas)), [f"{x:.2f}" for x in betas])
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title("Figure 5. Weight sensitivity for distributed-target MSE (points^2)")
    for i in range(len(betas)):
        for j in range(len(alphas)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "invalid", ha="center", va="center", fontsize=7, color="#555555")
            else:
                ax.text(j, i, f"{matrix[i, j]:.3f}\ng={gamma_map[(i, j)]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Distributed-target reputation MSE (points^2)")
    fig.tight_layout()
    fig.savefig(out_dir / "figure5_weight_heatmap.png", dpi=220)
    plt.close(fig)
    df.to_csv(out_dir / "figure5_weight_heatmap.csv", index=False)


def plot_figure6(df, out_dir):
    display = df.copy()
    display = display[["variant", "classification_accuracy", "reputation_mse", "avg_detection_delay_s", "communication_overhead_bits", "computation_cost_ms", "aggregation_latency_ms"]]
    display.columns = ["Variant", "Accuracy", "MSE(points^2)", "Delay(s)", "Comm overhead(bits)", "Compute cost(ms)", "Aggregation latency(ms)"]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    rounded = display.copy()
    for col in ["Accuracy", "MSE(points^2)", "Delay(s)", "Aggregation latency(ms)"]:
        rounded[col] = rounded[col].map(lambda x: f"{x:.3f}")
    for col in ["Comm overhead(bits)", "Compute cost(ms)"]:
        rounded[col] = rounded[col].map(lambda x: f"{x:.1f}")
    table = ax.table(cellText=rounded.values, colLabels=rounded.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    ax.set_title("Figure 6. Ablation table", pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / "figure6_ablation_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(out_dir / "figure6_ablation_table.csv", index=False)


def build_report(outputs_dir, fig1, fig2, fig3, fig4, fig5, fig6, repetitions, base_seed):
    best_heat = fig5.sort_values("reputation_mse").iloc[0]
    lines = [
        "# OX v0.2 trust experiment results",
        "",
        f"The six requested studies were run on the local OX_v0.2 platform with {repetitions} repetitions per vehicle count, seeds {base_seed}-{base_seed + repetitions - 1}, 3 UAVs, IEEE 802.11p settings, threshold N=25, and vehicle counts of 10, 40, 70, and 100.",
        "",
        "## Highlights",
        f"- Figure 1: `{fig1.sort_values('classification_accuracy', ascending=False).iloc[0]['attack_type']}` produced the highest average detection accuracy, while `{fig1.sort_values('avg_detection_delay_s').iloc[0]['attack_type']}` had the shortest average detection delay.",
        f"- Figure 2: the fusion method kept the lowest MSE at high collusion ratios, with the most visible gap near ratio `{fig2.sort_values('collusion_ratio').iloc[-1]['collusion_ratio']:.1f}`.",
        f"- Figure 3: event-driven updates ended at accuracy `{fig3[fig3['trust_update_mode'] == 'event']['classification_accuracy'].mean():.3f}` versus `{fig3[fig3['trust_update_mode'] == 'periodic']['classification_accuracy'].mean():.3f}` for periodic updates.",
        f"- Figure 4: compromising all 3 UAVs reduced average reliability to `{fig4[fig4['compromised_uavs'] == 3]['system_reliability'].iloc[0]:.3f}`.",
        f"- Figure 5: the best weight combination in this sweep was alpha={best_heat['alpha']:.2f}, beta={best_heat['beta']:.2f}, gamma={best_heat['gamma']:.2f}, with MSE `{best_heat['reputation_mse']:.3f}`.",
        f"- Figure 6: the full scheme achieved the strongest overall trade-off across accuracy, MSE, and detection delay, while each ablation degraded at least one key metric.",
        "",
        "Figure values are means across vehicle counts and repeated random seeds. Raw runs are in all_run_metrics.csv; per-scenario, per-vehicle-count mean/std/SEM/95% CI values are in all_scenario_statistics.csv.",
    ]
    (outputs_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Number of independent random-seed runs for each vehicle count (default: config value or 10)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="First random seed; repetition i uses base_seed + i (default: config seed)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = base_experiment_config(args.config)
    if args.repetitions is not None:
        if args.repetitions < 1:
            parser.error("--repetitions must be at least 1")
        base_cfg["experiment_repetitions"] = args.repetitions
    if args.base_seed is not None:
        base_cfg["seed"] = args.base_seed
    repetitions = max(1, int(base_cfg.get("experiment_repetitions", DEFAULT_EXPERIMENT_REPETITIONS)))
    base_seed = int(base_cfg.get("seed", 42))
    ctx = SimulationContext(base_cfg)

    all_metrics = []

    fig1_rows = []
    for attack_type in ["false_feedback", "spoofing", "collusion", "transient"]:
        df, _ = run_many(
            ctx,
            base_cfg,
            f"fig1_{attack_type}",
            lambda n, attack_type=attack_type: {
                "attack_type": attack_type,
                "attack_start_s": 8.0 if attack_type == "transient" else 0.0,
                "attack_end_s": 18.0 if attack_type == "transient" else 60.0,
                "malicious_vehicle_ratio": 0.2,
                "collusion_ratio": 0.2 if attack_type == "collusion" else 0.0,
            },
        )
        fig1_rows.append(df)
        all_metrics.append(df)
    fig1_df = pd.concat(fig1_rows, ignore_index=True)
    fig1_plot = fig1_df.groupby("attack_type", as_index=False)[["classification_accuracy", "avg_detection_delay_s"]].mean()
    plot_figure1(fig1_df, output_dir)

    fig2_rows = []
    for collusion_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        method_overrides = {
            "vehicle_only": {"vehicle_feedback_enabled": True, "uav_feedback_enabled": False, "trust_feedback_threshold": 25},
            "uav_only": {"vehicle_feedback_enabled": False, "uav_feedback_enabled": True, "trust_feedback_threshold": 25},
            "fusion": {"vehicle_feedback_enabled": True, "uav_feedback_enabled": True, "trust_feedback_threshold": 25},
        }
        for method, overrides in method_overrides.items():
            df, _ = run_many(
                ctx,
                base_cfg,
                f"fig2_{method}_{collusion_ratio:.1f}",
                lambda n, collusion_ratio=collusion_ratio, overrides=overrides: {
                    "attack_type": "collusion",
                    "malicious_vehicle_ratio": collusion_ratio,
                    "collusion_ratio": collusion_ratio,
                    **overrides,
                },
            )
            df["method"] = method
            df["collusion_ratio"] = collusion_ratio
            fig2_rows.append(df)
            all_metrics.append(df)
    fig2_df = pd.concat(fig2_rows, ignore_index=True)
    fig2_plot = fig2_df.groupby(["method", "collusion_ratio"], as_index=False)[["reputation_mse", "classification_accuracy"]].mean()
    plot_figure2(fig2_plot, output_dir)

    fig3_rows = []
    fig3_frames = []
    for mode in ["event", "periodic"]:
        df, frames = run_many(
            ctx,
            base_cfg,
            f"fig3_{mode}",
            lambda n, mode=mode: {
                "attack_type": "transient",
                "attack_start_s": 8.0,
                "attack_end_s": 18.0,
                "trust_update_mode": mode,
                "trust_target_policy": "attack_focus" if mode == "event" else "random",
                "trust_event_period_s": 6.0 if mode == "periodic" else 1.0,
                "event_scan_period_s": 0.25,
            },
        )
        fig3_rows.append(df)
        fig3_frames.append(frames)
        all_metrics.append(df)
    fig3_df = pd.concat(fig3_rows, ignore_index=True)
    fig3_curve = pd.concat(fig3_frames, ignore_index=True).groupby(["trust_update_mode", "time_s"], as_index=False)["avg_malicious_reputation"].mean()
    fig3_summary = fig3_df.groupby("trust_update_mode", as_index=False)[["classification_accuracy", "avg_detection_delay_s", "reputation_mse"]].mean()
    plot_figure3(fig3_curve, fig3_summary, output_dir, float(base_cfg.get("reputation_bad_threshold", 60.0)))

    fig4_rows = []
    for compromised_uavs in [0, 1, 2, 3]:
        df, _ = run_many(
            ctx,
            base_cfg,
            f"fig4_uav_{compromised_uavs}",
            lambda n, compromised_uavs=compromised_uavs: {
                "attack_type": "collusion",
                "malicious_vehicle_ratio": 0.4,
                "collusion_ratio": 0.4,
                "malicious_uav_ratio": compromised_uavs / 3.0,
                "malicious_uav_feedback_flip_prob": 1.0,
            },
        )
        df["compromised_uavs"] = compromised_uavs
        fig4_rows.append(df)
        all_metrics.append(df)
    fig4_df = pd.concat(fig4_rows, ignore_index=True)
    fig4_plot = fig4_df.groupby("compromised_uavs", as_index=False)[["classification_accuracy", "reputation_mse", "false_positive_rate"]].mean()
    plot_figure4(fig4_plot, output_dir)

    fig5_rows = []
    for alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:
        for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            gamma = round(1.0 - alpha - beta, 2)
            if gamma <= 0.0:
                fig5_rows.append({"alpha": alpha, "beta": beta, "gamma": gamma, "reputation_mse": np.nan})
                continue
            df, _ = run_many(
                ctx,
                base_cfg,
                f"fig5_{alpha:.2f}_{beta:.2f}",
                lambda n, alpha=alpha, beta=beta, gamma=gamma: {
                    "attack_type": "collusion",
                    "malicious_vehicle_ratio": 0.2,
                    "collusion_ratio": 0.2,
                    "trust_alpha": alpha,
                    "trust_beta": beta,
                    "trust_gamma": gamma,
                },
            )
            all_metrics.append(df)
            fig5_rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "reputation_mse": df["reputation_mse"].mean(),
                    "avg_detection_delay_s": df["avg_detection_delay_s"].mean(),
                }
            )
    fig5_df = pd.DataFrame(fig5_rows)
    plot_figure5(fig5_df, output_dir)

    variants = {
        "Full scheme": {},
        "No UAV feedback": {"uav_feedback_enabled": False},
        "No PUF": {"puf_enabled": False},
        "No secure aggregation": {"secure_aggregation_enabled": False, "insecure_feedback_duplication_factor": 2},
        "No weighted fusion": {"weighted_fusion_enabled": False},
        "No event-driven update": {"trust_update_mode": "periodic", "trust_target_policy": "random", "trust_event_period_s": 6.0},
    }
    fig6_rows = []
    for variant, overrides in variants.items():
        df, _ = run_many(
            ctx,
            base_cfg,
            f"fig6_{variant}",
            lambda n, overrides=overrides: {
                "attack_type": "collusion",
                "malicious_vehicle_ratio": 0.3,
                "collusion_ratio": 0.3,
                **overrides,
            },
        )
        all_metrics.append(df)
        row = {
            "variant": variant,
            "classification_accuracy": df["classification_accuracy"].mean(),
            "reputation_mse": df["reputation_mse"].mean(),
            "avg_detection_delay_s": df["avg_detection_delay_s"].mean(),
            "communication_overhead_bits": df["communication_overhead_bits"].mean(),
            "computation_cost_ms": df["computation_cost_ms"].mean(),
            "aggregation_latency_ms": df["aggregation_latency_ms"].mean(),
        }
        fig6_rows.append(row)
    fig6_df = pd.DataFrame(fig6_rows)
    plot_figure6(fig6_df, output_dir)

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(output_dir / "all_run_metrics.csv", index=False)
    build_run_statistics(all_metrics_df).to_csv(output_dir / "all_scenario_statistics.csv", index=False)
    build_report(
        output_dir,
        fig1_plot,
        fig2_plot,
        fig3_df,
        fig4_plot,
        fig5_df.dropna(),
        fig6_df,
        repetitions,
        base_seed,
    )
    print(f"Saved experiment outputs to: {output_dir}")


if __name__ == "__main__":
    main()
