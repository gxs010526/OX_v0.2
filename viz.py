# viz.py -- visualization and runner that ties engine to matplotlib animation

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

# colors for up/down
COLOR_UP = (0.9, 0.6, 0.0)    # orange
COLOR_DOWN = (0.2, 0.6, 0.2)  # green
COLOR_UAV = (0.25, 0.45, 0.95)
COLOR_TRUST = (0.55, 0.10, 0.75)

def setup_plot(engine):
    """
    Prepare figure, axes and initial scatter/text objects.
    Returns: fig, ax, veh_sc, veh_texts, veh_colors, cmap, LAT_MAP_MAX
    """
    margin = engine.MARGIN
    xmin_vis, xmax_vis = engine.minx - margin, engine.maxx + margin
    ymin_vis, ymax_vis = engine.miny - margin, engine.maxy + margin

    # create square figure but leave top margin for title (use subplots_adjust)
    fig, ax = plt.subplots(figsize=(12,12), dpi=120)
    fig.subplots_adjust(top=0.92)   # leave space for suptitle
    # place axes slightly inset so suptitle isn't overlapped
    ax.set_position([0.05, 0.03, 0.90, 0.88])
    ax.set_xlim(xmin_vis, xmax_vis); ax.set_ylim(ymin_vis, ymax_vis)
    ax.set_aspect('equal', 'box')
    ax.axis('off')

    # draw base road network faintly
    for u,v,k,data in engine.G.edges(keys=True, data=True):
        try:
            ax.plot([engine.G.nodes[u]['x'], engine.G.nodes[v]['x']],
                    [engine.G.nodes[u]['y'], engine.G.nodes[v]['y']],
                    color='lightgray', linewidth=0.6, zorder=0)
        except Exception:
            pass

    # draw cloud at center (keep visible)
    ax.scatter([engine.cloud_pos[0]], [engine.cloud_pos[1]], s=300, marker='o', color='orange', alpha=0.6, zorder=5)
    ax.text(engine.cloud_pos[0], engine.cloud_pos[1]+18, "Cloud", fontsize=9, ha='center', zorder=7)

    # draw UAVs as aerial auxiliary observers
    if getattr(engine, "uavs", None):
        xs_u = [u.x for u in engine.uavs]
        ys_u = [u.y for u in engine.uavs]
        ax.scatter(xs_u, ys_u, s=120, marker='^', color=COLOR_UAV, edgecolors='k', linewidths=0.4, zorder=6)
        for u in engine.uavs:
            ax.text(u.x, u.y+14, f"UAV{u.id}", fontsize=7, ha='center', color=COLOR_UAV, zorder=7)

    # --- compute colors per group for vehicle markers ---
    groups = engine.groups
    num_groups = max(1, len(groups))
    if num_groups <= 20:
        cmap_groups = plt.get_cmap('tab20')
        colors_groups = [cmap_groups(i % 20) for i in range(num_groups)]
    else:
        cmap_groups = plt.get_cmap('hsv')
        colors_groups = [cmap_groups(i / max(1, num_groups)) for i in range(num_groups)]

    # map each vehicle id to its group's color (vid2group should exist)
    veh_colors = []
    for v in engine.vehicles:
        gid = engine.vid2group.get(v.id, 0)
        color = colors_groups[gid % num_groups]
        veh_colors.append(color)

    # initial positions for scatter must match number of colors
    xs_v = []; ys_v = []
    for v in engine.vehicles:
        x,y = v.pos(engine.G)
        xs_v.append(x); ys_v.append(y)

    # vehicle scatter: provide initial offsets and set facecolors to veh_colors
    veh_sc = ax.scatter(xs_v, ys_v, s=48, zorder=8, facecolors=veh_colors, edgecolors='k', linewidths=0.2)
    veh_texts = [ax.text(0,0,"", fontsize=7, ha='center', va='center') for _ in engine.vehicles]

    cmap = plt.get_cmap('plasma'); LAT_MAP_MAX = 0.6

    # create an initial figure title using suptitle (will be updated each frame)
    fig.suptitle("", fontsize=12)

    ax._vanet_lines = []
    return fig, ax, veh_sc, veh_texts, veh_colors, cmap, LAT_MAP_MAX

def latency_color(cmap, LAT_MAP_MAX, lat_s):
    if lat_s is None: return (0.7,0.7,0.7)
    v = min(lat_s / LAT_MAP_MAX, 1.0)
    return cmap(v)

def run_animation(engine, output_gif, fps=10):
    """
    Run animation using engine.step().
    Visual: vehicles colored by group; no BS markers; up lines orange; down lines green.
    """
    fig, ax, veh_sc, veh_texts, veh_colors, cmap, LAT_MAP_MAX = setup_plot(engine)

    def _update(i):
        recent_links, tps_c, tps_v = engine.step()

        # update vehicle locations and labels
        xs_v=[]; ys_v=[]
        for ii,v in enumerate(engine.vehicles):
            x,y = v.pos(engine.G)
            xs_v.append(x); ys_v.append(y)
            veh_texts[ii].set_position((x, y-6.0)); veh_texts[ii].set_text(f"{v.id}:{v.reputation:.2f}")
        if len(xs_v)>0:
            veh_sc.set_offsets(np.column_stack((xs_v, ys_v)))
            # keep colors persistent
            veh_sc.set_facecolors(veh_colors)

        # remove old lines from previous frame
        if hasattr(ax, "_vanet_lines"):
            for la in list(ax._vanet_lines):
                try: la.remove()
                except: pass
        ax._vanet_lines = []

        # draw only vehicle <-> cloud connections
        for rl in recent_links:
            # Up: vehicle -> cloud (uniform orange)
            if rl['type'] == 'up':
                txp = engine.vehicles[rl['tx_vid']].pos(engine.G)
                cp = engine.cloud_pos
                color = COLOR_UP
                lw = 1.6 if rl.get('succ') else 0.8
                alpha = 0.95 if rl.get('succ') else 0.25
                l = ax.plot([txp[0], cp[0]], [txp[1], cp[1]], color=color, linewidth=lw, alpha=alpha, zorder=9)[0]
                ax._vanet_lines.append(l)

            # Down_start / Down: cloud -> vehicle (uniform green)
            elif rl['type'] in ('down_start', 'down'):
                vid = rl.get('vid')
                if vid is None:
                    continue
                rxp = engine.vehicles[vid].pos(engine.G)
                cp = engine.cloud_pos
                color = COLOR_DOWN
                lw = 1.6 if (rl.get('succ') or rl['type']=='down_start') else 0.8
                alpha = 0.9 if rl.get('succ') else 0.25
                l = ax.plot([cp[0], rxp[0]], [cp[1], rxp[1]], color=color, linewidth=lw, alpha=alpha, zorder=9)[0]
                ax._vanet_lines.append(l)

            elif rl['type'] == 'trust_vehicle':
                srcp = engine.vehicles[rl['src_vid']].pos(engine.G)
                dstp = engine.vehicles[rl['target_vid']].pos(engine.G)
                alpha = 0.55 if rl.get('succ') else 0.22
                l = ax.plot([srcp[0], dstp[0]], [srcp[1], dstp[1]], color=COLOR_TRUST, linewidth=0.9, alpha=alpha, zorder=8)[0]
                ax._vanet_lines.append(l)

            elif rl['type'] == 'trust_uav':
                uav = engine.uavs[rl['uav']]
                dstp = engine.vehicles[rl['target_vid']].pos(engine.G)
                alpha = 0.75 if rl.get('succ') else 0.28
                l = ax.plot([uav.x, dstp[0]], [uav.y, dstp[1]], color=COLOR_UAV, linewidth=1.2, alpha=alpha, zorder=8)[0]
                ax._vanet_lines.append(l)

            # ignore bs2cloud / cloud2bs visualization to avoid showing BS
            else:
                pass

        # update figure-level title (suptitle)
        tps_c_kbps = tps_c / 1000.0
        tps_v_kbps = tps_v / 1000.0
        pdr = (engine.total_delivered_recipients/engine.total_attempted_recipients) if engine.total_attempted_recipients>0 else 1.0
        rolling_avg_e2e = engine.frame_stats[-1]['rolling_avg_e2e_ms'] if engine.frame_stats else 0.0
        trust = engine.trust_summary() if hasattr(engine, 'trust_summary') else {'avg_reputation': 0.0, 'malicious_detected_ratio': 0.0}
        fig.suptitle(f"t={engine.time_sim:.1f}s | PDR={pdr:.3f} | avgE2E={rolling_avg_e2e:.1f} ms | TPS_cloud={tps_c_kbps:.1f} kbps TPS_vehicle={tps_v_kbps:.1f} kbps | Rep={trust['avg_reputation']:.2f} Det={trust['malicious_detected_ratio']:.2f}",
                     fontsize=12)

        artists = [veh_sc] + veh_texts + list(ax._vanet_lines)
        return artists

    anim = animation.FuncAnimation(fig, _update, frames=engine.STEPS, interval=1000*engine.DT, blit=False)
    print("Rendering GIF (may take a while)...")
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_gif, writer=writer)
        print("Saved GIF:", output_gif)
    except Exception as e:
        print("Animation save failed:", e)
