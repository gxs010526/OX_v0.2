# main.py
import argparse
from config_loader import load_json_config
from graph_loader import load_osm_graph, add_speeds_and_travel_time, get_edge_info_factory
from shadow import generate_shadow_field
from sim_core import VanetEngine
from viz import run_animation
import csv
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.json")
    args = parser.parse_args()
    cfg = load_json_config(args.config)

    # load graph
    G = load_osm_graph(cfg['place'], cfg.get('use_bbox', False), cfg.get('bbox', None), cfg.get('network_type', 'drive'))
    G = add_speeds_and_travel_time(G)
    get_edge_info = get_edge_info_factory(G, default_speed_m_s=cfg.get('default_speed_m_s', 13.9))

    # shadow
    xs = [G.nodes[n]['x'] for n in G.nodes]; ys = [G.nodes[n]['y'] for n in G.nodes]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    shadow_field, shadow_gx, shadow_gy = generate_shadow_field(minx, maxx, miny, maxy,
                                                               grid_res_m=cfg.get('shadow_grid_res_m',50.0),
                                                               sigma_m=cfg.get('shadow_corr_len_m',50.0),
                                                               sigma_db=cfg.get('shadow_std_db',7.0))

    # engine
    engine = VanetEngine(cfg, G, get_edge_info, shadow_field, shadow_gx, shadow_gy)

    # run visualization + sim
    run_animation(engine, cfg.get('output_gif', 'vanet_sim.gif'), fps=cfg.get('fps', 10))

    # save frame CSV and summary (engine.frame_stats contains per-frame data)
    frame_csv = cfg.get('frame_csv', 'frame_stats.csv')
    with open(frame_csv, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['frame','time_s','attempted','delivered','pdr_frame','avg_e2e_ms','rolling_pdr','rolling_avg_e2e_ms','tps_cloud_frame','tps_vehicle_frame','trust_events_frame','trust_updates_frame','avg_reputation','avg_malicious_reputation','avg_honest_reputation','malicious_detected_ratio']
        w.writerow(header)
        for s in engine.frame_stats:
            w.writerow([s[h] for h in header])
    summary_csv = cfg.get('summary_csv', 'summary.csv')
    overall_attempts = engine.total_attempted_recipients
    overall_delivered = engine.total_delivered_recipients
    overall_pdr = overall_delivered / overall_attempts if overall_attempts>0 else 0.0
    avg_e2e_ms = float(np.mean(engine.e2e_latencies)*1000) if engine.e2e_latencies else 0.0
    median_e2e_ms = float(np.median(engine.e2e_latencies)*1000) if engine.e2e_latencies else 0.0
    avg_up_ms = float(np.mean(engine.uplink_latencies)*1000) if engine.uplink_latencies else 0.0
    avg_down_ms = float(np.mean(engine.downlink_latencies)*1000) if engine.downlink_latencies else 0.0
    final_tps_cloud = (engine.total_cloud_bits / (engine.total_cloud_msgs * engine.CLOUD_PROC_S)) if engine.total_cloud_msgs>0 else 0.0
    final_tps_vehicle = (engine.total_vehicle_bits / (engine.total_vehicle_msgs * engine.VEHICLE_PROC_S)) if engine.total_vehicle_msgs>0 else 0.0
    trust = engine.trust_summary()
    trust_events_csv = cfg.get('trust_events_csv', 'trust_events.csv')
    with open(trust_events_csv, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['time_s','target_id','target_malicious','truth','report','vehicle_feedback_count','uav_feedback_count','vehicle_score','uav_score','updates']
        w.writerow(header)
        for event in engine.trust_events:
            w.writerow([event.get(h, '') for h in header])
    trust_updates_csv = cfg.get('trust_updates_csv', 'trust_updates.csv')
    with open(trust_updates_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s','target_id','old_reputation','new_reputation','vehicle_score','uav_score','vehicle_count','uav_count','feedback_count','aggregation_latency_s'])
        for update in engine.trust_updates:
            w.writerow([update.time_s, update.target_id, update.old_reputation, update.new_reputation,
                        update.vehicle_score, update.uav_score, update.vehicle_count, update.uav_count,
                        update.feedback_count, update.aggregation_latency_s])
    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric','value'])
        w.writerow(['overall_attempted_recipients', overall_attempts])
        w.writerow(['overall_delivered_recipients', overall_delivered])
        w.writerow(['overall_pdr', overall_pdr])
        w.writerow(['avg_e2e_ms', avg_e2e_ms])
        w.writerow(['median_e2e_ms', median_e2e_ms])
        w.writerow(['avg_uplink_ms', avg_up_ms])
        w.writerow(['avg_downlink_ms', avg_down_ms])
        w.writerow(['uplink_losses', engine.uplink_losses])
        w.writerow(['downlink_losses', engine.downlink_losses])
        w.writerow(['backhaul_losses', engine.backhaul_losses])
        w.writerow(['final_tps_cloud_bps', final_tps_cloud])
        w.writerow(['final_tps_vehicle_bps', final_tps_vehicle])
        for k, v in trust.items():
            w.writerow([k, v])
    print("Saved frame CSV:", frame_csv)
    print("Saved summary CSV:", summary_csv)
    print("Saved trust event CSV:", trust_events_csv)
    print("Saved trust update CSV:", trust_updates_csv)
    print("SUMMARY:")
    print(f" attempted recipients: {overall_attempts}")
    print(f" delivered recipients: {overall_delivered}")
    print(f" overall PDR: {overall_pdr:.4f}")
    print(f" avg E2E latency: {avg_e2e_ms:.1f} ms  median: {median_e2e_ms:.1f} ms")
    print(f" avg uplink latency (ms): {avg_up_ms:.1f}  avg downlink latency (ms): {avg_down_ms:.1f}")
    print(f" final TPS_cloud (bps): {final_tps_cloud:.1f}  final TPS_vehicle (bps): {final_tps_vehicle:.1f}")
    print(f" trust events: {trust['trust_events_total']}  updates: {trust['trust_updates_total']}  avg reputation: {trust['avg_reputation']:.3f}")
    print(f" malicious detection ratio: {trust['malicious_detected_ratio']:.3f}")

if __name__ == "__main__":
    main()
