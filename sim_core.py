# sim_core.py -- core simulation engine (uses other modules)
import random, math
import numpy as np
from typing import List, Dict, Any, Tuple
from utils import euclid, dbm_to_mw
from vehicles import (
    EVENT_TYPES,
    TRUST_FEEDBACK_NEGATIVE,
    TRUST_FEEDBACK_NEUTRAL,
    create_vehicles,
    Vehicle,
)
from bs_placement import place_base_stations_choose
from trust_entities import CloudServiceProvider, TrustFeedback, TrustedAuthority, create_uavs

class VanetEngine:
    def __init__(self, cfg: Dict[str,Any], G, get_edge_info, shadow_field, shadow_gx, shadow_gy):
        self.cfg = cfg
        random.seed(cfg.get('seed', 42)); np.random.seed(cfg.get('seed', 42))
        # graph
        self.G = G
        self.get_edge_info = get_edge_info
        self.shadow_field = shadow_field; self.shadow_gx = shadow_gx; self.shadow_gy = shadow_gy
        # radio params
        self.C = 3e8
        self.FREQ_HZ = cfg['freq_hz']
        self.LAMBDA = self.C / self.FREQ_HZ
        self.PTX_UE_DBM_MAX = cfg['ptx_ue_dbm_max']
        self.PTX_BS_DBM = cfg['ptx_bs_dbm']
        self.BW = cfg['bw']; self.NF_DB = cfg['nf_db']
        self.NOISE_DBM = -174 + 10 * math.log10(self.BW) + self.NF_DB
        self.NOISE_MW = dbm_to_mw(self.NOISE_DBM)
        self.PL_D0 = 20 * math.log10(4 * math.pi / self.LAMBDA)
        self.PATHLOSS_EXPONENT = cfg['pathloss_exponent']
        self.SINR_THRESH_DB = cfg['sinr_thresh_db']; self.SINR_LOGISTIC_K = cfg['sinr_logistic_k']
        self.PHY_RATE_BPS = cfg['phy_rate_bps']; self.MAC_OVERHEAD_S = cfg['mac_overhead_s']
        # simulation params
        self.SIM_SECONDS = cfg['sim_seconds']; self.DT = cfg['dt']; self.STEPS = int(self.SIM_SECONDS / self.DT)
        self.N_VEHICLES = int(cfg['n_vehicles'])
        self.DEFAULT_SPEED_M_S = float(cfg['default_speed_m_s'])
        self.SPEED_MIN = float(cfg['speed_scale_min']); self.SPEED_MAX = float(cfg['speed_scale_max'])
        self.GROUP_N = int(cfg['group_n']); self.GROUP_MSG_PROB = float(cfg['group_msg_prob'])
        self.SEND_PERIOD = float(cfg['send_period']); self.SEND_JITTER = float(cfg['send_jitter'])
        self.CLOUD_UP_MEAN_S = float(cfg['cloud_up_mean_s']); self.CLOUD_UP_STD_S = float(cfg['cloud_up_std_s'])
        self.CLOUD_DOWN_MEAN_S = float(cfg['cloud_down_mean_s']); self.CLOUD_DOWN_STD_S = float(cfg['cloud_down_std_s'])
        self.CLOUD_PROC_S = float(cfg['cloud_proc_s']); self.VEHICLE_PROC_S = float(cfg['vehicle_proc_s'])
        self.BACKHAUL_LOSS_PROB = float(cfg['backhaul_loss_prob'])
        self.TARGET_RX_DBM = float(cfg['target_rx_dbm'])
        self.N_BS = int(cfg['n_bs'])
        self.MARGIN = float(cfg['margin'])
        self.MAX_SIMULTANEOUS_SENDERS = int(cfg.get('max_simultaneous_senders', 9999))
        self.VIS_BS2CLOUD_EVERY_S = float(cfg.get('visualize_bs2cloud_every_s', 0.0))
        # trust-fusion params
        self.TRUST_ENABLED = bool(cfg.get('trust_enabled', True))
        self.TRUST_EVENT_PERIOD = float(cfg.get('trust_event_period_s', 3.0))
        self.TRUST_EVENT_PROB = float(cfg.get('trust_event_prob', 1.0))
        self.VEHICLE_OBS_RANGE_M = float(cfg.get('vehicle_observation_range_m', 260.0))
        self.WEATHER_NOISE = float(cfg.get('weather_noise', 0.04))
        self.REPUTATION_BAD_THRESHOLD = float(cfg.get('reputation_bad_threshold', 60.0))
        self.NEGATIVE_FEEDBACK_SCORE = float(cfg.get('negative_feedback_score', TRUST_FEEDBACK_NEGATIVE))
        self.NEUTRAL_FEEDBACK_SCORE = float(cfg.get('neutral_feedback_score', TRUST_FEEDBACK_NEUTRAL))
        self.TRUST_UPDATE_MODE = str(cfg.get('trust_update_mode', 'event')).lower()
        self.TRUST_TARGET_POLICY = str(cfg.get('trust_target_policy', 'random')).lower()
        self.ATTACK_FOCUS_PROB = float(cfg.get('attack_focus_probability', 0.8))
        self.EVENT_SCAN_PERIOD = float(cfg.get('event_scan_period_s', max(self.DT, 1.0)))
        self.VEHICLE_FEEDBACK_ENABLED = bool(cfg.get('vehicle_feedback_enabled', True))
        self.UAV_FEEDBACK_ENABLED = bool(cfg.get('uav_feedback_enabled', True))
        self.PUF_ENABLED = bool(cfg.get('puf_enabled', True))
        self.PUF_BLOCK_PROB = float(cfg.get('puf_block_prob', 0.75))
        self.INSECURE_DUPLICATION_FACTOR = max(1, int(cfg.get('insecure_feedback_duplication_factor', 1)))
        self.next_trust_event_time = random.uniform(0.0, self.TRUST_EVENT_PERIOD) if self.TRUST_ENABLED else float('inf')
        self.next_event_scan_time = 0.0 if self.TRUST_ENABLED else float('inf')
        #encryption params
        self.DATA_TEXT = float(cfg['data_text'])
        self.RE_ENCRY_KEY_BASE = float(cfg['re_encry_key_base'])
        self.RE_ENCRY_KEY_COEF = float(cfg['re_encry_key_coef'])
        self.RE_ENCRY_TEXT_BASE = float(cfg['re_encry_text_base'])
        self.RE_ENCRY_TEXT_COEF = float(cfg['re_encry_text_coef'])

        # place BS
        xs = [self.G.nodes[n]['x'] for n in self.G.nodes]; ys = [self.G.nodes[n]['y'] for n in self.G.nodes]
        self.minx, self.maxx = min(xs), max(xs); self.miny, self.maxy = min(ys), max(ys)
        self.map_center = (sum(xs)/len(xs), sum(ys)/len(ys))
        self.cloud_pos = self.map_center
        self.BS_POSITIONS = place_base_stations_choose(self.G, self.N_BS, self.minx, self.maxx, self.miny, self.maxy, prefer_on_graph=False)

        # vehicles
        self.vehicles: List[Vehicle] = create_vehicles(self.G, self.get_edge_info, self.N_VEHICLES,
                                                      self.DEFAULT_SPEED_M_S, self.SPEED_MIN, self.SPEED_MAX,
                                                      self.SEND_PERIOD, cfg.get('seed', 42), cfg)
        self.uavs = create_uavs(self.minx, self.maxx, self.miny, self.maxy, cfg)
        self.csp = CloudServiceProvider(threshold=int(cfg.get('trust_feedback_threshold', 3)),
                                        secure_aggregation_enabled=bool(cfg.get('secure_aggregation_enabled', True)),
                                        neutral_score=self.NEUTRAL_FEEDBACK_SCORE)
        self.ta = TrustedAuthority(alpha=float(cfg.get('trust_alpha', 0.65)),
                                   beta=float(cfg.get('trust_beta', 0.20)),
                                   gamma=float(cfg.get('trust_gamma', 0.15)),
                                   weighted_fusion_enabled=bool(cfg.get('weighted_fusion_enabled', True)))
        # groups
        group_size = self.GROUP_N + 1
        self.groups = {}
        for i,v in enumerate(self.vehicles):
            g = i // group_size
            self.groups.setdefault(g, []).append(v.id)
        self.vid2group = {vid:gid for gid,m in self.groups.items() for vid in m}

        # runtime state & stats
        self.pending_to_backhaul = []  # (arrival_time_at_dest_bs, permsg, dest_bs, src_bs)
        self.bs_downlink_queues = {i:[] for i in range(self.N_BS)}
        self.bs_next_free_time = {i:0.0 for i in range(self.N_BS)}
        self.total_attempted_recipients = 0
        self.total_delivered_recipients = 0
        self.e2e_latencies = []; self.uplink_latencies = []; self.downlink_latencies = []
        self.uplink_losses = 0; self.downlink_losses = 0; self.backhaul_losses = 0
        self.uplink_sinr_db = []; self.downlink_sinr_db = []; self.downlink_concurrency_counts = []
        self.frame_stats = []
        self.total_cloud_bits = 0; self.total_cloud_msgs = 0; self.total_vehicle_bits = 0; self.total_vehicle_msgs = 0
        self.trust_events = []
        self.trust_updates = []
        self.total_trust_feedback = 0
        self.total_vehicle_feedback_msgs = 0
        self.total_uav_feedback_msgs = 0
        self.total_puf_rejections = 0
        self.aggregation_latencies = []
        self.vehicle_detection_times = {}
        self.time_sim = 0.0
        self._active_downlinks = []

    # helpers (the same logic as previous monolith)
    def pathloss_db(self, d_m: float) -> float:
        if d_m < 1.0: d_m = 1.0
        return self.PL_D0 + 10.0 * self.PATHLOSS_EXPONENT * math.log10(d_m / 1.0)

    def sinr_success_prob(self, sinr_db: float) -> float:
        val = 1.0 / (1.0 + math.exp(-self.SINR_LOGISTIC_K * (sinr_db - self.SINR_THRESH_DB)))
        return float(max(0.0, min(1.0, val)))

    def tx_time_seconds(self, bits: int) -> float:
        return float(bits) / self.PHY_RATE_BPS + self.MAC_OVERHEAD_S

    def uplink_bits_for_m(self, m:int) -> int:
        return self.DATA_TEXT + self.RE_ENCRY_KEY_BASE + self.RE_ENCRY_KEY_COEF * m

    def downlink_bits_for_m(self, m:int) -> int:
        return self.RE_ENCRY_TEXT_BASE + self.RE_ENCRY_TEXT_COEF * (m + 2)

    def _submit_trust_feedback(self, feedback: TrustFeedback):
        self.total_trust_feedback += 1
        if feedback.source_type == "vehicle":
            if feedback.source_id >= 0:
                self.total_vehicle_feedback_msgs += 1
            else:
                self.total_puf_rejections += 1
        elif feedback.source_type == "uav":
            self.total_uav_feedback_msgs += 1
        aggregate = self.csp.submit(feedback)
        if aggregate is None:
            return None
        target = self.vehicles[int(aggregate['target_id'])]
        update = self.ta.update_reputation(target, aggregate, self.time_sim)
        self.trust_updates.append(update)
        self.aggregation_latencies.append(update.aggregation_latency_s)
        return update

    def _attack_active_vehicles(self):
        return [v for v in self.vehicles if v.attack_active(self.time_sim)]

    def _choose_trust_target(self):
        attack_active = self._attack_active_vehicles()
        undetected = [v for v in attack_active if v.id not in self.vehicle_detection_times]
        if self.TRUST_TARGET_POLICY == 'attack_focus' or (self.TRUST_UPDATE_MODE == 'event' and attack_active and random.random() < self.ATTACK_FOCUS_PROB):
            pool = undetected or attack_active
            if pool:
                return random.choice(pool)
        if self.TRUST_TARGET_POLICY == 'malicious_only':
            malicious = [v for v in self.vehicles if v.is_malicious]
            if malicious:
                return random.choice(malicious)
        return random.choice(self.vehicles)

    def _refresh_detection_state(self):
        for v in self.vehicles:
            if not v.is_malicious or v.id in self.vehicle_detection_times:
                continue
            if self.time_sim < v.attack_start_s:
                continue
            if v.reputation < self.REPUTATION_BAD_THRESHOLD:
                self.vehicle_detection_times[v.id] = self.time_sim

    def _run_trust_event(self, target=None):
        if not self.TRUST_ENABLED or not self.vehicles:
            return [], []
        if random.random() > self.TRUST_EVENT_PROB:
            return [], []

        target = target or self._choose_trust_target()
        target_pos = target.pos(self.G)
        truth = random.choice(EVENT_TYPES)
        target_observation = target.observe_event(truth, 0.0, self.VEHICLE_OBS_RANGE_M, self.WEATHER_NOISE)
        report = target.report_event(target_observation, self.time_sim)
        feedbacks = []
        updates = []

        if self.PUF_ENABLED and target.attack_active(self.time_sim) and report != target_observation and random.random() < self.PUF_BLOCK_PROB:
            feedbacks.append(TrustFeedback(target.id, "vehicle", -1, self.NEGATIVE_FEEDBACK_SCORE, self.time_sim, report, target_observation))

        if self.VEHICLE_FEEDBACK_ENABLED:
            for observer in self.vehicles:
                if observer.id == target.id:
                    continue
                dist = euclid(observer.pos(self.G), target_pos)
                if dist > self.VEHICLE_OBS_RANGE_M:
                    continue
                observed = observer.observe_event(truth, dist, self.VEHICLE_OBS_RANGE_M, self.WEATHER_NOISE)
                score = observer.evaluate_report(report, observed, target=target, time_s=self.time_sim)
                feedback = TrustFeedback(target.id, "vehicle", observer.id, score, self.time_sim, report, observed)
                feedbacks.append(feedback)
                if (not self.csp.secure_aggregation_enabled and observer.attack_active(self.time_sim)
                        and observer.attack_type == "collusion" and observer.colluding and self.INSECURE_DUPLICATION_FACTOR > 1):
                    for _ in range(self.INSECURE_DUPLICATION_FACTOR - 1):
                        feedbacks.append(TrustFeedback(target.id, "vehicle", observer.id, score, self.time_sim, report, observed))

        if self.UAV_FEEDBACK_ENABLED:
            for uav in self.uavs:
                if not uav.can_observe(target_pos):
                    continue
                observed = uav.observe_event(truth, target_pos, self.WEATHER_NOISE)
                score = uav.evaluate_report(report, observed, target_malicious=target.is_malicious)
                feedbacks.append(TrustFeedback(target.id, "uav", uav.id, score, self.time_sim, report, observed))

        for feedback in feedbacks:
            update = self._submit_trust_feedback(feedback)
            if update is not None:
                updates.append(update)
        self._refresh_detection_state()

        vehicle_scores = [f.score for f in feedbacks if f.source_type == "vehicle"]
        uav_scores = [f.score for f in feedbacks if f.source_type == "uav"]
        event = {
            'time_s': self.time_sim,
            'target_id': target.id,
            'target_malicious': target.is_malicious,
            'attack_type': target.attack_type,
            'attack_active': target.attack_active(self.time_sim),
            'truth': truth,
            'report': report,
            'vehicle_feedback_count': len(vehicle_scores),
            'uav_feedback_count': len(uav_scores),
            'vehicle_score': sum(vehicle_scores) / len(vehicle_scores) if vehicle_scores else None,
            'uav_score': sum(uav_scores) / len(uav_scores) if uav_scores else None,
            'reputation_after': target.reputation,
            'updates': len(updates),
        }
        self.trust_events.append(event)
        recent_links = []
        for f in feedbacks:
            if f.source_type == "vehicle":
                recent_links.append({'type':'trust_vehicle','src_vid':f.source_id,'target_vid':f.target_id,'succ':f.score >= self.NEUTRAL_FEEDBACK_SCORE,'lat':None})
            else:
                recent_links.append({'type':'trust_uav','uav':f.source_id,'target_vid':f.target_id,'succ':f.score >= self.NEUTRAL_FEEDBACK_SCORE,'lat':None})
        return recent_links, updates

    def trust_summary(self):
        reputations = [v.reputation for v in self.vehicles]
        malicious = [v for v in self.vehicles if v.is_malicious]
        honest = [v for v in self.vehicles if not v.is_malicious]
        detected = [v for v in malicious if v.reputation < self.REPUTATION_BAD_THRESHOLD]
        false_positives = [v for v in honest if v.reputation < self.REPUTATION_BAD_THRESHOLD]
        tp = len(detected)
        fn = max(0, len(malicious) - tp)
        fp = len(false_positives)
        tn = max(0, len(honest) - fp)
        ideals = [float(v.ideal_reputation) for v in self.vehicles]
        mse = float(np.mean([(rep - ideal) ** 2 for rep, ideal in zip(reputations, ideals)])) if reputations else 0.0
        detection_delays = [self.vehicle_detection_times[v.id] - v.attack_start_s for v in malicious if v.id in self.vehicle_detection_times]
        classification_accuracy = ((tp + tn) / len(self.vehicles)) if self.vehicles else 0.0
        avg_aggregation_latency_s = float(np.mean(self.aggregation_latencies)) if self.aggregation_latencies else self.SIM_SECONDS
        return {
            'trust_events_total': len(self.trust_events),
            'trust_updates_total': len(self.trust_updates),
            'trust_feedback_total': self.total_trust_feedback,
            'vehicle_feedback_total': self.total_vehicle_feedback_msgs,
            'uav_feedback_total': self.total_uav_feedback_msgs,
            'puf_rejections_total': self.total_puf_rejections,
            'avg_reputation': float(np.mean(reputations)) if reputations else 0.0,
            'min_reputation': float(np.min(reputations)) if reputations else 0.0,
            'avg_malicious_reputation': float(np.mean([v.reputation for v in malicious])) if malicious else 0.0,
            'avg_honest_reputation': float(np.mean([v.reputation for v in honest])) if honest else 0.0,
            'avg_ideal_reputation': float(np.mean(ideals)) if ideals else 0.0,
            'avg_malicious_ideal_reputation': float(np.mean([v.ideal_reputation for v in malicious])) if malicious else 0.0,
            'avg_honest_ideal_reputation': float(np.mean([v.ideal_reputation for v in honest])) if honest else 0.0,
            'malicious_detected_ratio': (len(detected) / len(malicious)) if malicious else 0.0,
            'false_positive_rate': (fp / len(honest)) if honest else 0.0,
            'classification_accuracy': classification_accuracy,
            'system_reliability': classification_accuracy,
            'reputation_mse': mse,
            'avg_aggregation_latency_s': avg_aggregation_latency_s,
            'avg_detection_delay_s': float(np.mean(detection_delays)) if detection_delays else self.SIM_SECONDS,
            'median_detection_delay_s': float(np.median(detection_delays)) if detection_delays else self.SIM_SECONDS,
            'undetected_malicious': fn,
        }

    # shadow accessor (bilinear interpolation)
    def shadow_at_pos(self, x: float, y: float) -> float:
        # quickest bilinear sample similar to prior function
        gx, gy = self.shadow_gx, self.shadow_gy
        field = self.shadow_field
        if x < gx[0]: x = gx[0]
        if x > gx[-1]: x = gx[-1]
        if y < gy[0]: y = gy[0]
        if y > gy[-1]: y = gy[-1]
        fx = (x - gx[0]) / (gx[-1] - gx[0]) * (len(gx)-1)
        fy = (y - gy[0]) / (gy[-1] - gy[0]) * (len(gy)-1)
        ix = int(math.floor(fx)); iy = int(math.floor(fy))
        ix1 = min(ix+1, len(gx)-1); iy1 = min(iy+1, len(gy)-1)
        wx = fx - ix; wy = fy - iy
        v00 = field[iy, ix]; v10 = field[iy, ix1]; v01 = field[iy1, ix]; v11 = field[iy1, ix1]
        v = (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11
        clip_db = self.cfg.get('shadow_clip_db', 12.0)
        return max(-clip_db, min(clip_db, float(v)))

    # one step (frame)
    def step(self):
        self.time_sim += self.DT
        recent_links = []
        frame_tps_cloud = 0.0; frame_tps_vehicle = 0.0
        trust_events_frame = 0; trust_updates_frame = 0

        # advance vehicles
        for v in self.vehicles: v.advance(self.DT, self.G, self.get_edge_info)

        # event-driven trust feedback and UAV-assisted fusion
        while self.TRUST_ENABLED and self.time_sim >= self.next_trust_event_time:
            before_events = len(self.trust_events)
            target = self._choose_trust_target() if self.TRUST_UPDATE_MODE == 'event' else random.choice(self.vehicles)
            trust_links, trust_updates = self._run_trust_event(target=target)
            recent_links.extend(trust_links)
            trust_events_frame += len(self.trust_events) - before_events
            trust_updates_frame += len(trust_updates)
            self.next_trust_event_time += self.TRUST_EVENT_PERIOD
        while self.TRUST_ENABLED and self.TRUST_UPDATE_MODE == 'event' and self.time_sim >= self.next_event_scan_time and self._attack_active_vehicles():
            before_events = len(self.trust_events)
            trust_links, trust_updates = self._run_trust_event(target=self._choose_trust_target())
            recent_links.extend(trust_links)
            trust_events_frame += len(self.trust_events) - before_events
            trust_updates_frame += len(trust_updates)
            self.next_event_scan_time += self.EVENT_SCAN_PERIOD

        # 1) process pending_to_backhaul (Cloud->BS)
        arrived = [it for it in list(self.pending_to_backhaul) if it[0] <= self.time_sim]
        for arrival_at_bs, msg, dest_bs, src_bs in arrived:
            try: self.pending_to_backhaul.remove((arrival_at_bs, msg, dest_bs, src_bs))
            except ValueError: pass
            if random.random() < self.BACKHAUL_LOSS_PROB:
                self.backhaul_losses += 1
                recent_links.append({'type':'cloud2bs','src':'cloud','dst':dest_bs,'succ':False,'lat':None})
                continue
            r = msg['recipient']
            down_bits = msg.get('down_bits', self.downlink_bits_for_m(len(msg['recipients'])))
            start_time = max(arrival_at_bs, self.bs_next_free_time[dest_bs]) + random.uniform(0.0, 0.02)
            duration = self.tx_time_seconds(down_bits)
            finish_time = start_time + duration
            self.bs_downlink_queues[dest_bs].append({'start': start_time, 'finish': finish_time, 'msg': msg, 'recipient': r, 'src_bs': src_bs})
            self.bs_next_free_time[dest_bs] = finish_time + 0.001
            lat_show = msg.get('cloud_to_bs', arrival_at_bs - (self.time_sim - self.DT))
            recent_links.append({'type':'cloud2bs','src':'cloud','dst':dest_bs,'succ':True,'lat':lat_show})

        # 2) move queued downlinks into active_downlinks
        for bs_idx, q in self.bs_downlink_queues.items():
            ready = [it for it in q if it['start'] <= self.time_sim]
            for item in ready:
                q.remove(item)
                self._active_downlinks.append({'start': item['start'], 'finish': item['finish'], 'bs': bs_idx, 'recipient': item['recipient'], 'msg': item['msg']})

        # 3) process completion of active_downlinks whose finish <= time_sim
        completed = [it for it in list(self._active_downlinks) if it['finish'] <= self.time_sim]
        for comp in completed:
            try: self._active_downlinks.remove(comp)
            except ValueError: pass
            bs_tx = comp['bs']; r = comp['recipient']; msg = comp['msg']
            all_trans = [other for other in (completed + self._active_downlinks) if not (other is comp) and (other['start'] < comp['finish'] and other['finish'] > comp['start'])]
            bsp = self.BS_POSITIONS[bs_tx]; rxp = self.vehicles[r].pos(self.G)
            d = euclid(bsp, rxp); pl = self.pathloss_db(d)
            midx = 0.5*(bsp[0] + rxp[0]); midy = 0.5*(bsp[1] + rxp[1])
            sh_db = self.shadow_at_pos(midx, midy)
            pr_dbm = self.PTX_BS_DBM - pl + sh_db
            pr_mw = dbm_to_mw(pr_dbm)
            interfer_mw = 0.0
            for oth in all_trans:
                ob = oth['bs']; obp = self.BS_POSITIONS[ob]
                d_o = euclid(obp, rxp); pl_o = self.pathloss_db(d_o)
                midx_o = 0.5*(obp[0] + rxp[0]); midy_o = 0.5*(obp[1] + rxp[1])
                sh_o = self.shadow_at_pos(midx_o, midy_o)
                pr_o_dbm = self.PTX_BS_DBM - pl_o + sh_o
                interfer_mw += dbm_to_mw(pr_o_dbm)
            sinr_lin = pr_mw / (interfer_mw + self.NOISE_MW)
            sinr_db = 10.0 * math.log10(sinr_lin) if sinr_lin>0 else -200.0
            self.downlink_sinr_db.append(sinr_db)
            self.downlink_concurrency_counts.append(1 + len(all_trans))
            p_succ = self.sinr_success_prob(sinr_db)
            down_bits = self.downlink_bits_for_m(len(msg['recipients']))
            tx_t = self.tx_time_seconds(down_bits)
            net_lat = tx_t + 0.001 * random.random()
            if random.random() < p_succ:
                self.vehicles[r].recv += 1
                self.total_delivered_recipients += 1
                delivery_time = comp['finish']
                e2e = delivery_time - msg['t0']
                self.e2e_latencies.append(e2e)
                self.downlink_latencies.append(net_lat)
                recent_links.append({'type':'down','bs':bs_tx,'vid':r,'succ':True,'lat':net_lat})
            else:
                self.downlink_losses += 1
                recent_links.append({'type':'down','bs':bs_tx,'vid':r,'succ':False,'lat':None})

        # 4) uplink senders (cap concurrency)
        ready_senders = [v for v in self.vehicles if v.next_tx_time <= self.time_sim]
        senders = ready_senders[:self.MAX_SIMULTANEOUS_SENDERS]
        attempted_this_frame = 0
        if senders:
            round_senders = []
            for tx in senders:
                tx.sent += 1
                if random.random() < self.GROUP_MSG_PROB:
                    gid = self.vid2group[tx.id]; recipients = [vid for vid in self.groups[gid] if vid != tx.id]
                else:
                    recipients = [v.id for v in self.vehicles if v.id != tx.id]
                m = len(recipients)
                self.total_attempted_recipients += m
                attempted_this_frame += m
                msg = {'id': None, 'tx_vid': tx.id, 'recipients': recipients, 't0': self.time_sim, 'retries': 0}
                round_senders.append((tx, m, msg))
                tx.next_tx_time = self.time_sim + self.SEND_PERIOD + random.uniform(-self.SEND_JITTER, self.SEND_JITTER)

            sender_info = []
            for tx, m, msg in round_senders:
                pos = tx.pos(self.G)
                best_bs = None; best_pr_mean = None; best_d=None
                for bs_idx, bs_pos in enumerate(self.BS_POSITIONS):
                    d = euclid(pos, bs_pos); pl = self.pathloss_db(d)
                    pr_mean = -pl
                    if best_bs is None or pr_mean > best_pr_mean:
                        best_bs = bs_idx; best_pr_mean = pr_mean; best_d = d
                pl_ch = self.pathloss_db(best_d)
                ptx_needed_dbm = self.TARGET_RX_DBM + pl_ch
                ptx_actual_dbm = min(self.PTX_UE_DBM_MAX, ptx_needed_dbm)
                bsp = self.BS_POSITIONS[best_bs]; midx = 0.5*(pos[0] + bsp[0]); midy = 0.5*(pos[1] + bsp[1])
                sh = self.shadow_at_pos(midx, midy)
                pr_dbm = ptx_actual_dbm - pl_ch + sh
                sender_info.append({'tx':tx, 'm':m, 'msg':msg, 'bs':best_bs, 'pr_dbm':pr_dbm, 'ptx_dbm':ptx_actual_dbm})

            # enroll interference per-BS
            bs_groups = {}
            for s in sender_info: bs_groups.setdefault(s['bs'], []).append(s)

            for s in sender_info:
                bs_idx = s['bs']; pr_dbm = s['pr_dbm']; pr_mw = dbm_to_mw(pr_dbm)
                interfer_mw = 0.0
                for other in bs_groups.get(bs_idx, []):
                    if other is s: continue
                    interfer_mw += dbm_to_mw(other['pr_dbm'])
                sinr_lin = pr_mw / (interfer_mw + self.NOISE_MW)
                sinr_db = 10.0 * math.log10(sinr_lin) if sinr_lin>0 else -200.0
                self.uplink_sinr_db.append(sinr_db)
                p_succ = self.sinr_success_prob(sinr_db)
                up_bits = self.uplink_bits_for_m(s['m'])
                tx_t = self.tx_time_seconds(up_bits)
                cell_lat = tx_t + 0.001 * random.random()

                # TPS accounting vehicle
                self.total_vehicle_bits += up_bits
                self.total_vehicle_msgs += 1
                frame_tps_vehicle += (up_bits / self.VEHICLE_PROC_S)

                if random.random() < p_succ:
                    # success -> schedule backhaul for each recipient
                    down_bits_msg = self.downlink_bits_for_m(len(s['msg']['recipients']))
                    self.total_cloud_bits += down_bits_msg
                    self.total_cloud_msgs += 1
                    frame_tps_cloud += (down_bits_msg / self.CLOUD_PROC_S)

                    for r in s['msg']['recipients']:
                        # find best BS for recipient
                        rx_pos = self.vehicles[r].pos(self.G)
                        best_rb = None; best_pr = None; best_d2=None
                        for bs_j, bs_pos in enumerate(self.BS_POSITIONS):
                            d2 = euclid(bs_pos, rx_pos); pl2 = self.pathloss_db(d2)
                            pr_mean2 = self.PTX_BS_DBM - pl2
                            if best_rb is None or pr_mean2 > best_pr:
                                best_rb = bs_j; best_pr = pr_mean2; best_d2 = d2
                        bs_to_cloud = max(0.0, random.gauss(self.CLOUD_UP_MEAN_S, self.CLOUD_UP_STD_S))
                        cloud_proc = max(0.0, self.CLOUD_PROC_S)
                        cloud_to_bs = max(0.0, random.gauss(self.CLOUD_DOWN_MEAN_S, self.CLOUD_DOWN_STD_S))
                        backhaul_latency = bs_to_cloud + cloud_proc + cloud_to_bs
                        arrival_at_dest_bs = self.time_sim + cell_lat + backhaul_latency
                        permsg = {'id':None, 'tx_vid': s['tx'].id, 'recipients': s['msg']['recipients'],
                                  'recipient': r, 't0': s['msg']['t0'], 'src_bs': s['bs'],
                                  'bs_to_cloud': bs_to_cloud, 'cloud_proc': cloud_proc, 'cloud_to_bs': cloud_to_bs,
                                  'down_bits': down_bits_msg}
                        self.pending_to_backhaul.append((arrival_at_dest_bs, permsg, best_rb, s['bs']))
                        recent_links.append({'type':'bs2cloud','src': s['bs'],'dst':'cloud','succ':True,'lat': bs_to_cloud})
                    self.uplink_latencies.append(cell_lat)
                    recent_links.append({'type':'up','tx_vid': s['tx'].id, 'bs': s['bs'], 'succ':True, 'lat': cell_lat})
                else:
                    self.uplink_losses += 1
                    recent_links.append({'type':'up','tx_vid': s['tx'].id, 'bs': s['bs'], 'succ':False, 'lat': None})

        # stats and frame summary
        attempted_frame = attempted_this_frame if 'attempted_this_frame' in locals() else 0
        delivered_frame = sum(1 for rl in recent_links if rl.get('type') == 'down' and rl.get('succ') is True)
        avg_e2e_ms = float(np.mean(self.e2e_latencies)*1000) if self.e2e_latencies else 0.0
        pdr_frame = (delivered_frame / attempted_frame) if attempted_frame > 0 else 0.0
        N_roll = 8
        recent = self.frame_stats[-N_roll:]
        if recent:
            sum_attempted = sum(f.get('attempted',0) for f in recent) + attempted_frame
            sum_delivered = sum(f.get('delivered',0) for f in recent) + delivered_frame
            rolling_pdr = (sum_delivered / sum_attempted) if sum_attempted > 0 else ((self.total_delivered_recipients/self.total_attempted_recipients) if self.total_attempted_recipients>0 else 0.0)
            rolling_avg_e2e = (sum(f.get('avg_e2e_ms',0.0) for f in recent) + avg_e2e_ms) / (len(recent) + 1)
        else:
            rolling_pdr = (self.total_delivered_recipients/self.total_attempted_recipients) if self.total_attempted_recipients>0 else pdr_frame
            rolling_avg_e2e = avg_e2e_ms

        tps_cloud_overall = (self.total_cloud_bits / (self.total_cloud_msgs * self.CLOUD_PROC_S)) if self.total_cloud_msgs>0 else 0.0
        tps_vehicle_overall = (self.total_vehicle_bits / (self.total_vehicle_msgs * self.VEHICLE_PROC_S)) if self.total_vehicle_msgs>0 else 0.0

        trust_frame_summary = self.trust_summary()
        self.frame_stats.append({
            'frame': len(self.frame_stats),
            'time_s': round(self.time_sim,3),
            'attempted': attempted_frame,
            'delivered': delivered_frame,
            'pdr_frame': round(pdr_frame, 6),
            'avg_e2e_ms': avg_e2e_ms,
            'rolling_pdr': round(rolling_pdr, 6),
            'rolling_avg_e2e_ms': rolling_avg_e2e,
            'tps_cloud_frame': frame_tps_cloud,
            'tps_vehicle_frame': frame_tps_vehicle,
            'trust_events_frame': trust_events_frame,
            'trust_updates_frame': trust_updates_frame,
            'avg_reputation': trust_frame_summary['avg_reputation'],
            'avg_malicious_reputation': trust_frame_summary['avg_malicious_reputation'],
            'avg_honest_reputation': trust_frame_summary['avg_honest_reputation'],
            'malicious_detected_ratio': trust_frame_summary['malicious_detected_ratio'],
            'false_positive_rate': trust_frame_summary['false_positive_rate'],
            'classification_accuracy': trust_frame_summary['classification_accuracy'],
            'reputation_mse': trust_frame_summary['reputation_mse'],
            'avg_aggregation_latency_s': trust_frame_summary['avg_aggregation_latency_s'],
            'avg_detection_delay_s': trust_frame_summary['avg_detection_delay_s']
        })

        return recent_links, tps_cloud_overall, tps_vehicle_overall
