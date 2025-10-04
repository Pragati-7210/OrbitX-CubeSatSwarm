# cubeSat_triangulation_kepler_inclined.py
# Keplerian inclined asteroid + moving CubeSat swarm + 3D animated triangulation
# Earth now moves along its orbit!

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Utility / triangulation maths
# -----------------------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def multiline_best_point(points, dirs):
    S = np.zeros((3, 3))
    b = np.zeros(3)
    for p, u in zip(points, dirs):
        u = u / np.linalg.norm(u)
        A = np.eye(3) - np.outer(u, u)
        S += A
        b += A @ p
    try:
        x = np.linalg.solve(S, b)
    except np.linalg.LinAlgError:
        x = np.mean(points, axis=0)
    return x

# -----------------------------
# Kepler orbit helper functions
# -----------------------------
def solve_kepler(M, e, tol=1e-8, max_iter=100):
    if e < 0.8:
        E = M
    else:
        E = math.pi
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            return E
    return E

def kepler_to_cartesian(a, e, i_deg, raan_deg, argp_deg, M_rad):
    E = solve_kepler(M_rad, e)
    cos_v = (np.cos(E) - e) / (1 - e * np.cos(E))
    sin_v = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e * np.cos(E))
    v = math.atan2(sin_v, cos_v)
    r = a * (1 - e * np.cos(E))
    x_orb = r * np.cos(v)
    y_orb = r * np.sin(v)
    z_orb = 0.0
    i = math.radians(i_deg)
    raan = math.radians(raan_deg)
    argp = math.radians(argp_deg)
    R_z_raan = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan),  np.cos(raan), 0],
        [0, 0, 1]
    ])
    R_x_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i),  np.cos(i)]
    ])
    R_z_argp = np.array([
        [np.cos(argp), -np.sin(argp), 0],
        [np.sin(argp),  np.cos(argp), 0],
        [0, 0, 1]
    ])
    R = R_z_raan @ R_x_i @ R_z_argp
    pos_orb = np.array([x_orb, y_orb, z_orb])
    pos = R @ pos_orb
    return pos

# -----------------------------
# Parameters
# -----------------------------
NUM_CUBESATS = 6
NUM_FRAMES = 360

# Orbital elements
earth_elements = dict(a=1.0, e=0.0167, i_deg=0.0, raan_deg=0.0, argp_deg=102.9)
venus_elements = dict(a=0.723, e=0.0067, i_deg=3.39, raan_deg=76.7, argp_deg=54.9)

# Asteroid (example values)
a = 1.4
e = 0.25
inclination = 20.0
raan = 30.0
arg_peri = 45.0

M0 = 0.0
AU_TO_KM = 149597870

# Precompute asteroid positions
Ms = M0 + np.linspace(0, 2*np.pi, NUM_FRAMES, endpoint=False)
asteroid_true = np.array([kepler_to_cartesian(a, e, inclination, raan, arg_peri, M) for M in Ms])

venus_orbit = np.array([kepler_to_cartesian(**venus_elements, M_rad=M) for M in Ms])
earth_orbit = np.array([kepler_to_cartesian(**earth_elements, M_rad=M) for M in Ms])

# CubeSats placed along Venus’s orbit
sat_base_angles = np.linspace(0, 2*np.pi, NUM_CUBESATS, endpoint=False)

# -----------------------------
# Matplotlib setup
# -----------------------------
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
plt.tight_layout()
ax.set_facecolor("k")
fig.patch.set_facecolor("k")

lim = 2.0
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim/2, lim/2)

ax.set_xlabel("X (AU)", color="white")
ax.set_ylabel("Y (AU)", color="white")
ax.set_zlabel("Z (AU)", color="white")
ax.set_title("CubeSat Swarm Triangulation of Inclined Asteroid", color="white")

# Sun
ax.scatter(0, 0, 0, color="yellow", s=200, label="Sun")

# Orbits
ax.plot(earth_orbit[:,0], earth_orbit[:,1], earth_orbit[:,2],
        color="royalblue", linestyle="--", linewidth=1, label="Earth orbit")
ax.plot(venus_orbit[:,0], venus_orbit[:,1], venus_orbit[:,2],
        color="orange", linestyle=":", linewidth=1, label="Venus orbit (CubeSats)")
ax.plot(asteroid_true[:,0], asteroid_true[:,1], asteroid_true[:,2],
        color="gray", linestyle="dashdot", linewidth=1, alpha=0.6, label="Asteroid orbit")

# Artists
ast_marker, = ax.plot([], [], [], marker='D', color='white', markersize=6, label="Asteroid (true)")
sat_markers, = ax.plot([], [], [], marker='o', linestyle='', color='red', markersize=6, label="CubeSats")
beam_lines = [ax.plot([], [], [], color='cyan', linewidth=1, alpha=0.7)[0] for _ in range(NUM_CUBESATS)]
est_marker, = ax.plot([], [], [], marker='o', color='lime', markersize=7, label="Estimated (triangulated)")
est_trace_x, est_trace_y, est_trace_z = [], [], []
(est_trace_line,) = ax.plot([], [], [], color='lime', linewidth=1.5, linestyle='--', alpha=0.9, label="Estimated orbit trace")

# NEW: Earth moving marker
earth_marker, = ax.plot([], [], [], marker='o', color='blue', markersize=8, label="Earth")

err_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color="white")

leg = ax.legend(loc='upper right')
for text in leg.get_texts():
    text.set_color("white")

# -----------------------------
# Animation update
# -----------------------------
def update(frame_idx):
    global est_trace_x, est_trace_y, est_trace_z

    ast_pos = asteroid_true[frame_idx % len(asteroid_true)]
    venus_pos = venus_orbit[frame_idx % len(venus_orbit)]
    earth_pos = earth_orbit[frame_idx % len(earth_orbit)]

    sats = []
    for base_ang in sat_base_angles:
        ang = base_ang
        pos = kepler_to_cartesian(venus_elements["a"], venus_elements["e"], venus_elements["i_deg"],
                                  venus_elements["raan_deg"], venus_elements["argp_deg"], ang)
        sats.append(pos)
    sats = np.array(sats)

    dirs = [unit(ast_pos - p) for p in sats]
    est = multiline_best_point(sats, dirs)
    err = np.linalg.norm(est - ast_pos) * AU_TO_KM

    # Update markers
    ast_marker.set_data([ast_pos[0]], [ast_pos[1]])
    ast_marker.set_3d_properties([ast_pos[2]])

    earth_marker.set_data([earth_pos[0]], [earth_pos[1]])
    earth_marker.set_3d_properties([earth_pos[2]])

    sat_markers.set_data(sats[:,0], sats[:,1])
    sat_markers.set_3d_properties(sats[:,2])

    for i, line in enumerate(beam_lines):
        sx, sy, sz = sats[i]
        line.set_data([sx, ast_pos[0]], [sy, ast_pos[1]])
        line.set_3d_properties([sz, ast_pos[2]])

    est_marker.set_data([est[0]], [est[1]])
    est_marker.set_3d_properties([est[2]])

    est_trace_x.append(est[0]); est_trace_y.append(est[1]); est_trace_z.append(est[2])
    est_trace_line.set_data(est_trace_x, est_trace_y)
    est_trace_line.set_3d_properties(est_trace_z)

    err_text.set_text(f"Frame {frame_idx+1}/{NUM_FRAMES}   Triangulation error = {err:,.0f} km")

    ax.view_init(elev=25, azim=(frame_idx * 0.5) % 360)

    return [ast_marker, sat_markers, est_marker, est_trace_line, earth_marker, err_text] + beam_lines

# -----------------------------
# Run animation
# -----------------------------
anim = FuncAnimation(fig, update, frames=NUM_FRAMES, interval=70, blit=False)
plt.show()
