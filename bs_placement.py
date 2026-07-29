# bs_placement.py -- place base stations on grid or snapped to graph nodes
import random
import numpy as np
from typing import List, Tuple

def place_base_stations_grid(minx, maxx, miny, maxy, n_bs:int) -> List[Tuple[float,float]]:
    rows = int((n_bs**0.5)//1)
    if rows == 0: rows = 1
    cols = int((n_bs + rows - 1)//rows)
    xs_grid = np.linspace(minx, maxx, cols+1)[1:-1] if cols>1 else [(minx+maxx)/2.0]
    ys_grid = np.linspace(miny, maxy, rows+1)[1:-1] if rows>1 else [(miny+maxy)/2.0]
    bss=[]
    for yi in ys_grid:
        for xi in xs_grid:
            if len(bss) < n_bs: bss.append((float(xi), float(yi)))
    return bss[:n_bs]

def place_base_stations_on_graph_nodes(G, n_bs:int) -> List[Tuple[float,float]]:
    node_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes]
    random.shuffle(node_coords)
    return node_coords[:n_bs]

def place_base_stations_choose(G, n_bs:int, minx, maxx, miny, maxy, prefer_on_graph=False):
    bss = []
    if prefer_on_graph:
        bss = place_base_stations_on_graph_nodes(G, n_bs)
    else:
        bss = place_base_stations_grid(minx, maxx, miny, maxy, n_bs)
        if len(bss) < n_bs:
            node_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes]; random.shuffle(node_coords)
            for coord in node_coords:
                if len(bss) >= n_bs: break
                bss.append(coord)
    return bss[:n_bs]
