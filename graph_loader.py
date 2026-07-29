# graph_loader.py -- load OSM graph and provide edge info helper
import osmnx as ox
import math
from typing import Tuple, Any

def load_osm_graph(place: str, use_bbox: bool, bbox, network_type: str):
    if use_bbox:
        north, south, east, west = bbox
        top, bottom = max(north, south), min(north, south)
        right, left = max(east, west), min(east, west)
        G = ox.graph_from_bbox((left, bottom, right, top), network_type=network_type)
    else:
        G = ox.graph_from_place(place, network_type=network_type)
    G = ox.project_graph(G)
    return G

def add_speeds_and_travel_time(G):
    try:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        for u,v,k,data in G.edges(keys=True, data=True):
            sp = data.get('speed_kph', None)
            if sp is not None:
                data['speed_m_s'] = float(sp) * 1000.0 / 3600.0
            else:
                data['speed_m_s'] = data.get('speed_m_s', None)
    except Exception:
        # fallback: do nothing (caller should handle missing values)
        pass
    return G

def get_edge_info_factory(G, default_speed_m_s=13.9):
    # returns a function get_edge_info(u,v) -> (length_m, speed_m_s, travel_time_s)
    def get_edge_info(u, v):
        if G.is_multigraph():
            best = None
            for k, data in G[u][v].items():
                tt = data.get('travel_time', data.get('length', 0.0) / (data.get('speed_m_s', default_speed_m_s) + 1e-9))
                if best is None or tt < best[2]:
                    best = (data.get('length', math.hypot(G.nodes[u]['x']-G.nodes[v]['x'], G.nodes[u]['y']-G.nodes[v]['y'])),
                            data.get('speed_m_s', default_speed_m_s), tt)
            return best
        else:
            data = G[u][v]
            return (data.get('length', math.hypot(G.nodes[u]['x']-G.nodes[v]['x'], G.nodes[u]['y']-G.nodes[v]['y'])),
                    data.get('speed_m_s', default_speed_m_s), data.get('travel_time', (data.get('length', 0.0) / (data.get('speed_m_s', default_speed_m_s) + 1e-9))))
    return get_edge_info
