import numpy as np, math, pandas as pd, matplotlib.pyplot as plt, networkx as nx
from pathlib import Path


R = 100.0
T = 100.0
step = 0.2
N = int(round(1.0 / step))
names = ["Avi", "Benni", "Gabbi"]
rooms = ["Livingroom", "Bedroom", "Basement"]  # index 0,1,2 respectively

values = {
    "Avi": np.array([70.0, 20.0, 10.0]),
    "Benni": np.array([35.0, 45.0, 20.0]),
    "Gabbi": np.array([45.0, 45.0, 10.0])
}

combined_label_rule = "majority"
tenant_for_label = "Avi"


outdir = Path(".venv/data/simplex_rental_harmony")
outdir.mkdir(parents=True, exist_ok=True)


points = []
index_lookup = {}
for i in range(N + 1):
    for j in range(N + 1 - i):
        k = N - i - j
        idx = len(points)
        points.append((i, j, k))
        index_lookup[(i, j)] = idx
points = np.array(points, dtype=int)
num_points = len(points)


points_frac = points / N
p1 = points_frac[:, 0]
p2 = points_frac[:, 1]
p3 = points_frac[:, 2]
sqrt3 = math.sqrt(3.0)
xy = np.zeros((num_points, 2))
xy[:, 0] = p2 + 0.5 * p3
xy[:, 1] = (sqrt3 / 2.0) * p3

triangles = []
for i in range(N):
    for j in range(N - i):
        a = index_lookup[(i, j)]
        b = index_lookup.get((i + 1, j), None)
        c = index_lookup.get((i, j + 1), None)
        d = index_lookup.get((i + 1, j + 1), None)
        if b is not None and c is not None:
            triangles.append((a, b, c))
        if b is not None and c is not None and d is not None:
            triangles.append((b, d, c))
triangles = np.array(triangles, dtype=int)

prices = points_frac * R



def label_for_tenant_at_price(tenant_values, price_vec):
    """Return the index (0/1/2) of the room that maximizes utilities = value - price.
       Tie breaks by smallest index."""
    utilities = tenant_values - price_vec
    maxval = utilities.max()
    candidates = np.where(np.isclose(utilities, maxval))[0]
    return int(candidates.min())


labels_per_tenant = {
    name: np.array([label_for_tenant_at_price(values[name], prices[i]) for i in range(num_points)], dtype=int) for name
    in names}



def combined_label_at_vertex(vertex_idx):
    per = [labels_per_tenant[name][vertex_idx] for name in names]
    if combined_label_rule == "majority":
        counts = np.bincount(per, minlength=3)
        max_count = counts.max()
        candidates = np.where(counts == max_count)[0]
        return int(candidates.min())  # tie-break by smallest room index among tied modes
    elif combined_label_rule == "lexicographic":
        # lexicographic over tenants order: pick smallest room index appearing in the tenant list order
        return int(min(per))
    elif combined_label_rule == "by_tenant":
        return int(labels_per_tenant[tenant_for_label][vertex_idx])
    else:
        raise ValueError("Unknown combined_label_rule")


combined_labels = np.array([combined_label_at_vertex(i) for i in range(num_points)], dtype=int)

# Save vertex table
df_vertices = pd.DataFrame({
    "vertex_idx": np.arange(num_points),
    "i": points[:, 0], "j": points[:, 1], "k": points[:, 2],
    "p1_frac": points_frac[:, 0], "p2_frac": points_frac[:, 1], "p3_frac": points_frac[:, 2],
    "price_p1": prices[:, 0], "price_p2": prices[:, 1], "price_p3": prices[:, 2],
    "label_Avi": labels_per_tenant["Avi"],
    "label_Benni": labels_per_tenant["Benni"],
    "label_Gabbi": labels_per_tenant["Gabbi"],
    "combined_label": combined_labels
})
df_vertices.to_csv(outdir / "vertices_labels.csv", index=False)


fully_labeled = []
for t_idx, tri in enumerate(triangles):
    labs = set(combined_labels[list(tri)])
    if labs == {0, 1, 2}:
        fully_labeled.append((t_idx, tuple(tri)))

# Save triangles summary
tri_rows = []
for t_idx, tri in fully_labeled:
    tri_rows.append({"triangle_index": int(t_idx), "v0": int(tri[0]), "v1": int(tri[1]), "v2": int(tri[2])})
df_fully = pd.DataFrame(tri_rows)
df_fully.to_csv(outdir / "fully_labeled_triangles_summary.csv", index=False)


def utilities_at_price(price_vec):
    return {name: values[name] - price_vec for name in names}


def top_choices_at_price(price_vec):
    utils = utilities_at_price(price_vec)
    return {name: set(np.where(np.isclose(utils[name], utils[name].max()))[0].tolist()) for name in names}


def find_perfect_matching_for_topchoices(top_choices):
    """Return a tenant->room assignment if a perfect matching exists where each tenant
       gets one of their top choices; else return None.
    """
    B = nx.Graph()
    tenant_nodes = [f"t_{t}" for t in names]
    room_nodes = [f"r_{r}" for r in range(3)]
    B.add_nodes_from(tenant_nodes, bipartite=0)
    B.add_nodes_from(room_nodes, bipartite=1)
    for t in names:
        for r in top_choices[t]:
            B.add_edge(f"t_{t}", f"r_{r}")
    try:
        matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(B, set(tenant_nodes))
    except Exception:
        mm = nx.max_weight_matching(B, maxcardinality=True)
        matching = {}
        for a, b in mm:
            matching[a] = b;
            matching[b] = a
    if not all(f"t_{t}" in matching for t in names):
        return None
    assignment = {}
    for t in names:
        room_node = matching.get(f"t_{t}")
        if room_node is None:
            return None
        room_idx = int(room_node.split("_")[1])
        assignment[t] = room_idx
    return assignment


def verify_envyfree(price_vec, assignment):
    utils = utilities_at_price(price_vec)
    envyfree = True
    details = {}
    for t in names:
        assigned_room = assignment[t]
        u_assigned = float(utils[t][assigned_room])
        prefers_other = []
        for r in range(3):
            u_r = float(utils[t][r])
            if u_r > u_assigned + 1e-9:
                prefers_other.append((r, u_r))
        details[t] = {"assigned": int(assigned_room), "u_assigned": u_assigned, "prefers_other": prefers_other}
        if prefers_other:
            envyfree = False
    return envyfree, details


results = []

for t_idx, tri in fully_labeled:
    verts = list(tri)
    v_prices = [prices[v] for v in verts]
    v_bary = [points_frac[v] for v in verts]
    centroid = np.mean(np.array(v_prices), axis=0)
    centroid = centroid * (R / centroid.sum())  # normalize to sum exactly R in case of tiny drift
    centroid_top = top_choices_at_price(centroid)
    centroid_assignment = find_perfect_matching_for_topchoices(centroid_top)
    centroid_ef = None
    centroid_details = None
    if centroid_assignment is not None:
        centroid_ef, centroid_details = verify_envyfree(centroid, centroid_assignment)
    found = None
    if centroid_assignment is None:
        samp_step = 0.02
        alphas = np.arange(0, 1 + samp_step, samp_step)
        done = False
        for a in alphas:
            if done: break
            for b in alphas:
                c = 1.0 - a - b
                if c < -1e-9:
                    continue
                if c < 0: c = 0.0
                p = a * np.array(v_prices[0]) + b * np.array(v_prices[1]) + c * np.array(v_prices[2])
                # normalize to sum R
                p = p * (R / p.sum())
                top = top_choices_at_price(p)
                assignment = find_perfect_matching_for_topchoices(top)
                if assignment is not None:
                    ef, details = verify_envyfree(p, assignment)
                    found = {"price": tuple(np.round(p, 6)), "sum": float(np.round(p.sum(), 6)), "top_choices": top,
                             "assignment": assignment, "envyfree": ef, "envy_details": details,
                             "bary_weights": (a, b, c)}
                    done = True
                    break
    results.append({
        "triangle_index": int(t_idx),
        "vertices": verts,
        "vertex_prices": [tuple(np.round(vp, 6)) for vp in v_prices],
        "centroid_price": tuple(np.round(centroid, 6)),
        "centroid_top_choices": centroid_top,
        "centroid_assignment": centroid_assignment,
        "centroid_envyfree": centroid_ef,
        "found_interior": found
    })

rows = []
for res in results:
    for v_idx, vp in zip(res["vertices"], res["vertex_prices"]):
        rows.append({
            "triangle_index": res["triangle_index"],
            "vertex_index": int(v_idx),
            "price_p1": vp[0], "price_p2": vp[1], "price_p3": vp[2],
            "centroid_price_p1": res["centroid_price"][0],
            "centroid_price_p2": res["centroid_price"][1],
            "centroid_price_p3": res["centroid_price"][2]
        })
df_results = pd.DataFrame(rows)
df_results.to_csv(outdir / "fully_labeled_triangles_detailed.csv", index=False)

assign_rows = []
for res in results:
    found = res["found_interior"]
    if found is not None:
        tri_idx = res["triangle_index"]
        for t in names:
            assign_rows.append({
                "triangle_index": tri_idx,
                "tenant": t,
                "assigned_room_idx": found["assignment"][t],
                "assigned_room": rooms[found["assignment"][t]],
                "price_p1": found["price"][0],
                "price_p2": found["price"][1],
                "price_p3": found["price"][2],
                "envyfree": found["envyfree"]
            })
if assign_rows:
    pd.DataFrame(assign_rows).to_csv(outdir / "found_interior_assignments.csv", index=False)

plt.figure(figsize=(6, 6))
plt.triplot(xy[:, 0], xy[:, 1], triangles, linewidth=0.8)
plt.scatter(xy[:, 0], xy[:, 1], s=30)
for idx in range(num_points):
    lab = combined_labels[idx]
    short = ["L", "B", "S"][lab]
    plt.text(xy[idx, 0], xy[idx, 1], f"{short}{lab}", fontsize=8, ha="center", va="center")
plt.title("Triangulation with combined labels at vertices")
plt.axis('equal');
plt.axis('off');
plt.tight_layout()
plt.savefig(outdir / "triangulation_combined_labels.png")
plt.show()

plt.figure(figsize=(6, 6))
plt.triplot(xy[:, 0], xy[:, 1], triangles, linewidth=0.8)
plt.scatter(xy[:, 0], xy[:, 1], s=20)
for res in results:
    tri = triangles[res["triangle_index"]]
    coords = xy[list(tri)]
    if res["found_interior"] is not None:
        plt.fill(coords[:, 0], coords[:, 1], alpha=0.25)
        p = np.array(res["found_interior"]["price"])
        frac = p / R
        px = frac[1] + 0.5 * frac[2]
        py = (sqrt3 / 2.0) * frac[2]
        plt.scatter([px], [py], s=50)
plt.title("Fully-labeled triangles (shaded) and found interior envy-free points (dots)")
plt.axis('equal');
plt.axis('off');
plt.tight_layout()
plt.savefig(outdir / "fully_labeled_triangles_highlighted_with_points.png")
plt.show()

print(f"Grid step: {step} (N={N}), total vertices: {num_points}, small triangles: {len(triangles)}")

print(f"Found {len(fully_labeled)} fully-labeled small triangle(s). Results saved to: {outdir}")


