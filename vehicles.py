# vehicles.py -- Vehicle dataclass and factory
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import networkx as nx

EVENT_TYPES = [
    "traffic_accident",
    "traffic_condition",
    "weather_condition",
    "violation",
    "no_incident",
]

TRUST_SCORE_MIN = 0.0
TRUST_SCORE_MAX = 100.0
TRUST_FEEDBACK_NEGATIVE = 40.0
TRUST_FEEDBACK_NEUTRAL = 60.0
TRUST_FEEDBACK_POSITIVE = 80.0

@dataclass
class Vehicle:
    id:int
    node_route:List[int]
    cur_idx:int=0
    pos_frac:float=0.0
    speed_scale:float=1.0
    sent:int=0
    recv:int=0
    next_tx_time:float=0.0
    reputation:float=80.0
    ideal_reputation:float=80.0
    is_malicious:bool=False
    attack_type:str="mixed"
    attack_start_s:float=0.0
    attack_end_s:float=float("inf")
    colluding:bool=False
    observation_accuracy:float=0.82
    false_report_prob:float=0.0
    malicious_feedback_prob:float=0.0
    negative_feedback_score:float=TRUST_FEEDBACK_NEGATIVE
    positive_feedback_score:float=TRUST_FEEDBACK_POSITIVE

    def pos(self, G):
        if self.cur_idx >= len(self.node_route)-1:
            n=self.node_route[-1]; return (G.nodes[n]['x'], G.nodes[n]['y'])
        a=self.node_route[self.cur_idx]; b=self.node_route[self.cur_idx+1]
        xa,ya=G.nodes[a]['x'],G.nodes[a]['y']; xb,yb=G.nodes[b]['x'],G.nodes[b]['y']
        return (xa + (xb-xa)*self.pos_frac, ya + (yb-ya)*self.pos_frac)

    def advance(self, dt, G, get_edge_info):
        while self.cur_idx >= len(self.node_route)-1:
            start=self.node_route[-1]; nodes_list=list(G.nodes)
            target=random.choice(nodes_list)
            if target==start: continue
            try: new_route=nx.shortest_path(G, start, target, weight='travel_time')
            except: new_route=[start, target]
            if len(new_route)>=2:
                self.node_route=new_route; self.cur_idx=0; self.pos_frac=0.0; break
            else: return
        u=self.node_route[self.cur_idx]; v=self.node_route[self.cur_idx+1]
        info = get_edge_info(u,v)
        if not info:
            self.cur_idx += 1; self.pos_frac=0.0; return
        length_m, speed_m_s, tt = info
        actual_speed = speed_m_s * self.speed_scale
        dist = actual_speed * dt
        frac = dist / (length_m + 1e-9)
        self.pos_frac += frac
        while self.pos_frac >= 1.0 and self.cur_idx < len(self.node_route)-1:
            self.pos_frac -= 1.0; self.cur_idx += 1
            if self.cur_idx >= len(self.node_route)-1: break

    def observe_event(self, truth: str, distance_m: float, max_range_m: float, weather_noise: float=0.0) -> str:
        distance_penalty = max(0.0, min(0.45, distance_m / max(max_range_m, 1.0) * 0.35))
        p_correct = self.observation_accuracy - distance_penalty - weather_noise
        p_correct = max(0.05, min(0.98, p_correct))
        if random.random() < p_correct:
            return truth
        choices = [event for event in EVENT_TYPES if event != truth]
        return random.choice(choices)

    def attack_active(self, time_s: float) -> bool:
        return self.is_malicious and self.attack_start_s <= time_s <= self.attack_end_s

    def report_event(self, observed: str, time_s: float=0.0) -> str:
        false_prob = 0.0
        if self.attack_active(time_s) and self.attack_type in {"mixed", "spoofing", "collusion", "transient"}:
            false_prob = self.false_report_prob
        if random.random() >= false_prob:
            return observed
        choices = [event for event in EVENT_TYPES if event != observed]
        return random.choice(choices)

    def evaluate_report(self, reported: str, observed: str, target: Optional["Vehicle"]=None, time_s: float=0.0) -> float:
        score = self.positive_feedback_score if reported == observed else self.negative_feedback_score
        if not self.attack_active(time_s):
            return score
        if self.attack_type == "collusion" and self.colluding and target is not None and random.random() < self.malicious_feedback_prob:
            return self.positive_feedback_score if getattr(target, "is_malicious", False) else self.negative_feedback_score
        if self.attack_type in {"mixed", "false_feedback", "transient", "collusion"} and random.random() < self.malicious_feedback_prob:
            score = self.positive_feedback_score + self.negative_feedback_score - score
        return score

def sample_ideal_reputation(is_malicious: bool, cfg: Dict[str, Any], rng: random.Random) -> float:
    """Sample a fixed class-conditional ideal reputation from a truncated normal distribution."""
    threshold = float(cfg.get('reputation_bad_threshold', 60.0))
    clip_sigma = max(0.0, float(cfg.get('ideal_reputation_clip_sigma', 2.0)))
    if is_malicious:
        mean = float(cfg.get('malicious_ideal_reputation_mean', 40.0))
        std = max(0.0, float(cfg.get('malicious_ideal_reputation_std', 10.0)))
        lower = max(TRUST_SCORE_MIN, mean - clip_sigma * std)
        upper = min(TRUST_SCORE_MAX, mean + clip_sigma * std, math.nextafter(threshold, TRUST_SCORE_MIN))
    else:
        mean = float(cfg.get('honest_ideal_reputation_mean', 80.0))
        std = max(0.0, float(cfg.get('honest_ideal_reputation_std', 10.0)))
        lower = max(TRUST_SCORE_MIN, mean - clip_sigma * std, threshold)
        upper = min(TRUST_SCORE_MAX, mean + clip_sigma * std)

    if lower > upper:
        raise ValueError(f"Invalid ideal reputation bounds: [{lower}, {upper}]")
    if std == 0.0:
        return max(lower, min(upper, mean))
    for _ in range(1000):
        value = rng.gauss(mean, std)
        if lower <= value <= upper:
            return value
    return max(lower, min(upper, mean))


def create_vehicles(G, get_edge_info, n_vehicles:int, default_speed_m_s:float, speed_min:float, speed_max:float, send_period:float, seed: int=42, cfg: Dict[str, Any]=None):
    import random, numpy as np
    cfg = cfg or {}
    random.seed(seed)
    node_list = list(G.nodes)
    vehicles = []
    malicious_ratio = float(cfg.get('malicious_vehicle_ratio', 0.2))
    malicious_count = int(round(n_vehicles * malicious_ratio))
    malicious_ids = set(random.sample(range(n_vehicles), malicious_count)) if malicious_count > 0 else set()
    attack_type = str(cfg.get('attack_type', 'mixed')).lower()
    attack_start_s = float(cfg.get('attack_start_s', 0.0))
    attack_end_s = float(cfg.get('attack_end_s', float('inf')))
    collusion_ratio = float(cfg.get('collusion_ratio', malicious_ratio))
    colluding_count = min(malicious_count, int(round(n_vehicles * collusion_ratio)))
    colluding_ids = set(random.sample(list(malicious_ids), colluding_count)) if colluding_count > 0 else set()
    initial_reputation = float(cfg.get('initial_reputation', 80.0))
    ideal_rng = random.Random(int(seed) + 7919)
    honest_obs_acc = float(cfg.get('vehicle_observation_accuracy', 0.82))
    malicious_obs_acc = float(cfg.get('malicious_vehicle_observation_accuracy', honest_obs_acc))
    false_report_prob = float(cfg.get('malicious_false_report_prob', 0.75))
    malicious_feedback_prob = float(cfg.get('malicious_feedback_flip_prob', 0.65))
    negative_feedback_score = float(cfg.get('negative_feedback_score', TRUST_FEEDBACK_NEGATIVE))
    positive_feedback_score = float(cfg.get('positive_feedback_score', TRUST_FEEDBACK_POSITIVE))
    for vid in range(n_vehicles):
        s = random.choice(node_list); t = random.choice(node_list)
        if s == t: t = random.choice(node_list)
        try:
            r = nx.shortest_path(G, s, t, weight='travel_time')
        except:
            r = [s, t]
        is_malicious = vid in malicious_ids
        ideal_reputation = sample_ideal_reputation(is_malicious, cfg, ideal_rng)
        v = Vehicle(id=vid, node_route=r, speed_scale=random.uniform(speed_min, speed_max),
                    reputation=initial_reputation, ideal_reputation=ideal_reputation, is_malicious=is_malicious,
                    attack_type=attack_type if is_malicious else "none",
                    attack_start_s=attack_start_s if is_malicious else float("inf"),
                    attack_end_s=attack_end_s if is_malicious else float("-inf"),
                    colluding=vid in colluding_ids,
                    observation_accuracy=malicious_obs_acc if is_malicious else honest_obs_acc,
                    false_report_prob=false_report_prob if is_malicious else 0.0,
                    malicious_feedback_prob=malicious_feedback_prob if is_malicious else 0.0,
                    negative_feedback_score=negative_feedback_score,
                    positive_feedback_score=positive_feedback_score)
        v.cur_idx = 0; v.pos_frac = random.random() * 0.8
        v.next_tx_time = random.uniform(0.0, send_period)
        vehicles.append(v)
    return vehicles
