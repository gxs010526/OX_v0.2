# trust_entities.py -- UAV-assisted trust fusion entities
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils import euclid
from vehicles import (
    EVENT_TYPES,
    TRUST_FEEDBACK_NEGATIVE,
    TRUST_FEEDBACK_NEUTRAL,
    TRUST_FEEDBACK_POSITIVE,
    TRUST_SCORE_MAX,
    TRUST_SCORE_MIN,
)


@dataclass
class TrustFeedback:
    target_id: int
    source_type: str
    source_id: int
    score: float
    time_s: float
    report: str
    observation: str


@dataclass
class ReputationUpdate:
    target_id: int
    old_reputation: float
    new_reputation: float
    vehicle_score: float
    uav_score: float
    vehicle_count: int
    uav_count: int
    feedback_count: int
    aggregation_latency_s: float
    time_s: float


@dataclass
class UAV:
    id: int
    x: float
    y: float
    altitude_m: float = 120.0
    coverage_radius_m: float = 550.0
    observation_accuracy: float = 0.93
    communication_delay_s: float = 0.003
    malicious: bool = False
    malicious_feedback_prob: float = 0.0
    negative_feedback_score: float = TRUST_FEEDBACK_NEGATIVE
    positive_feedback_score: float = TRUST_FEEDBACK_POSITIVE

    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def can_observe(self, point: Tuple[float, float]) -> bool:
        return euclid(self.pos(), point) <= self.coverage_radius_m

    def observe_event(self, truth: str, point: Tuple[float, float], weather_noise: float=0.0) -> str:
        dist = euclid(self.pos(), point)
        distance_penalty = max(0.0, min(0.25, dist / max(self.coverage_radius_m, 1.0) * 0.18))
        p_correct = self.observation_accuracy - distance_penalty - 0.5 * weather_noise
        p_correct = max(0.05, min(0.99, p_correct))
        if random.random() < p_correct:
            return truth
        choices = [event for event in EVENT_TYPES if event != truth]
        return random.choice(choices)

    def evaluate_report(self, reported: str, observed: str, target_malicious: Optional[bool]=None) -> float:
        score = self.positive_feedback_score if reported == observed else self.negative_feedback_score
        if self.malicious and random.random() < self.malicious_feedback_prob:
            if target_malicious is None:
                score = self.positive_feedback_score + self.negative_feedback_score - score
            else:
                score = self.positive_feedback_score if target_malicious else self.negative_feedback_score
        return score


@dataclass
class CloudServiceProvider:
    threshold: int
    secure_aggregation_enabled: bool = True
    neutral_score: float = TRUST_FEEDBACK_NEUTRAL
    pending: Dict[int, List[TrustFeedback]] = field(default_factory=dict)
    aggregate_count: int = 0

    def _aggregate_scores(self, scores: List[float]) -> float:
        if not scores:
            return self.neutral_score
        return sum(scores) / len(scores)

    def submit(self, feedback: TrustFeedback) -> Optional[Dict[str, object]]:
        bucket = self.pending.setdefault(feedback.target_id, [])
        bucket.append(feedback)
        vehicle_feedbacks = [f for f in bucket if f.source_type == "vehicle" and f.source_id >= 0]
        uav_feedbacks = [f for f in bucket if f.source_type == "uav"]
        eligible_feedback_count = len(vehicle_feedbacks) + len(uav_feedbacks)
        if eligible_feedback_count < self.threshold:
            return None
        items = self.pending.pop(feedback.target_id)
        self.aggregate_count += 1
        puf_scores = [f.score for f in items if f.source_type == "vehicle" and f.source_id < 0]
        vehicle_scores = [f.score for f in vehicle_feedbacks] + puf_scores
        uav_scores = [f.score for f in uav_feedbacks]
        return {
            "target_id": feedback.target_id,
            "vehicle_score": self._aggregate_scores(vehicle_scores),
            "uav_score": self._aggregate_scores(uav_scores),
            "vehicle_count": len(vehicle_feedbacks),
            "uav_count": len(uav_feedbacks),
            "puf_count": len(puf_scores),
            "feedback_count": eligible_feedback_count,
            "secure_aggregation_enabled": self.secure_aggregation_enabled,
            "first_feedback_time_s": min(f.time_s for f in items),
            "last_feedback_time_s": max(f.time_s for f in items),
        }


@dataclass
class TrustedAuthority:
    alpha: float = 0.65
    beta: float = 0.20
    gamma: float = 0.15
    weighted_fusion_enabled: bool = True
    min_reputation: float = TRUST_SCORE_MIN
    max_reputation: float = TRUST_SCORE_MAX
    updates: List[ReputationUpdate] = field(default_factory=list)

    def update_reputation(self, vehicle, aggregate: Dict[str, object], time_s: float) -> ReputationUpdate:
        old = float(vehicle.reputation)
        if self.weighted_fusion_enabled:
            weighted_values = [(self.alpha, old)]
            if int(aggregate["vehicle_count"]) > 0:
                weighted_values.append((self.beta, float(aggregate["vehicle_score"])))
            if int(aggregate["uav_count"]) > 0:
                weighted_values.append((self.gamma, float(aggregate["uav_score"])))
            total = sum(weight for weight, _ in weighted_values)
            new_value = sum(weight * value for weight, value in weighted_values) / total
        else:
            scores = []
            if int(aggregate["vehicle_count"]) > 0:
                scores.append(float(aggregate["vehicle_score"]))
            if int(aggregate["uav_count"]) > 0:
                scores.append(float(aggregate["uav_score"]))
            optimistic_score = max(scores) if scores else old
            new_value = 0.5 * old + 0.5 * optimistic_score
        new_value = max(self.min_reputation, min(self.max_reputation, new_value))
        vehicle.reputation = new_value
        update = ReputationUpdate(
            target_id=vehicle.id,
            old_reputation=old,
            new_reputation=new_value,
            vehicle_score=float(aggregate["vehicle_score"]),
            uav_score=float(aggregate["uav_score"]),
            vehicle_count=int(aggregate["vehicle_count"]),
            uav_count=int(aggregate["uav_count"]),
            feedback_count=int(aggregate.get("feedback_count", 0)),
            aggregation_latency_s=float(aggregate.get("last_feedback_time_s", time_s) - aggregate.get("first_feedback_time_s", time_s)),
            time_s=time_s,
        )
        self.updates.append(update)
        return update


def create_uavs(minx: float, maxx: float, miny: float, maxy: float, cfg) -> List[UAV]:
    n_uavs = int(cfg.get("n_uavs", 4))
    if n_uavs <= 0:
        return []
    rows = max(1, int(n_uavs ** 0.5))
    cols = max(1, int((n_uavs + rows - 1) // rows))
    xs = [minx + (maxx - minx) * (i + 1) / (cols + 1) for i in range(cols)]
    ys = [miny + (maxy - miny) * (j + 1) / (rows + 1) for j in range(rows)]
    positions = []
    for y in ys:
        for x in xs:
            positions.append((x, y))
    malicious_ratio = float(cfg.get("malicious_uav_ratio", 0.0))
    malicious_count = int(round(n_uavs * malicious_ratio))
    malicious_ids = set(random.sample(range(n_uavs), malicious_count)) if malicious_count > 0 else set()
    return [
        UAV(
            id=i,
            x=float(positions[i][0]),
            y=float(positions[i][1]),
            altitude_m=float(cfg.get("uav_altitude_m", 120.0)),
            coverage_radius_m=float(cfg.get("uav_coverage_radius_m", 550.0)),
            observation_accuracy=float(cfg.get("uav_observation_accuracy", 0.93)),
            communication_delay_s=float(cfg.get("uav_comm_delay_s", 0.003)),
            malicious=i in malicious_ids,
            malicious_feedback_prob=float(cfg.get("malicious_uav_feedback_flip_prob", 0.4)),
            negative_feedback_score=float(cfg.get("negative_feedback_score", TRUST_FEEDBACK_NEGATIVE)),
            positive_feedback_score=float(cfg.get("positive_feedback_score", TRUST_FEEDBACK_POSITIVE)),
        )
        for i in range(n_uavs)
    ]
