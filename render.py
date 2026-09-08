#!/usr/bin/env python3
"""Read VMC binary output and render with Rerun."""

import argparse
import os
import struct
import sys

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# -- Binary format ----------------------------------------------------------
# Header (24 bytes):
#   uint64  num_particles
#   float64 box_length
#   uint64  measure_steps
#
# Per frame (32 + num_particles*24 bytes):
#   float64 local_energy
#   float64 mean_energy
#   float64 standard_error
#   float64 acceptance_rate
#   float64 positions[num_particles * 3]   (x0,y0,z0,x1,y1,z1,...)

HEADER_SIZE = 24  # bytes
SCALARS_PER_FRAME = 4  # local_energy, mean_energy, se, acceptance_rate

# Series colours, also reused for the 3D overlays so the whole scene reads as one palette.
COLOR_LOCAL = [0xF2, 0x8C, 0x3C]
COLOR_MEAN = [0x3F, 0xE0, 0xC8]
COLOR_ERROR = [0xC9, 0x9A, 0x3A]
COLOR_ACCEPT = [0x8A, 0x9B, 0xF0]
COLOR_BOX = [0x4A, 0x6B, 0x8A]


def read_header(path: str):
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        sys.exit(f"File too small for header: {path}")
    num_particles, box_length, measure_steps = struct.unpack("<QdQ", raw)
    return num_particles, box_length, measure_steps


def frame_bytes(num_particles: int) -> int:
    return (SCALARS_PER_FRAME + num_particles * 3) * 8


def read_all(path: str, num_particles: int, stride: int):
    """Read the whole recording into memory.

    Returns (scalars, positions) where scalars is (F, 4) holding
    [local_energy, mean_energy, standard_error, acceptance_rate] and positions
    is (F, num_particles, 3). Recordings are a few hundred KB to a few hundred
    MB, and every downstream step (colour ranges, speeds, g(r)) needs more than
    one pass, so streaming it repeatedly costs more than just holding it.
    """
    per_frame = SCALARS_PER_FRAME + num_particles * 3
    raw = np.fromfile(path, dtype=np.float64, offset=HEADER_SIZE)
    total = raw.size // per_frame
    raw = raw[: total * per_frame].reshape(total, per_frame)

    raw = raw[::stride]
    scalars = raw[:, :SCALARS_PER_FRAME].copy()
    positions = raw[:, SCALARS_PER_FRAME:].reshape(len(raw), num_particles, 3).copy()
    return scalars, positions


# -- Colour -----------------------------------------------------------------
# Multi-stop ramp: deep indigo -> blue -> cyan -> green -> amber -> white-hot.
# Perceptually increasing in lightness, so "fast" reads as "bright" even before
# you work out which end of the scale is which.
_STOPS = np.array(
    [
        [0.00, 0.05, 0.03, 0.20],
        [0.15, 0.09, 0.16, 0.55],
        [0.32, 0.04, 0.48, 0.80],
        [0.50, 0.05, 0.80, 0.74],
        [0.66, 0.38, 0.93, 0.38],
        [0.80, 0.93, 0.86, 0.20],
        [0.92, 1.00, 0.56, 0.18],
        [1.00, 1.00, 0.93, 0.86],
    ]
)


def colormap(t: np.ndarray) -> np.ndarray:
    """Map t in [0, 1] to uint8 RGB."""
    t = np.clip(t, 0.0, 1.0)
    channels = [np.interp(t, _STOPS[:, 0], _STOPS[:, c]) for c in (1, 2, 3)]
    return (np.stack(channels, axis=-1) * 255.0).astype(np.uint8)


def minimum_image(delta: np.ndarray, box_length: float) -> np.ndarray:
    """Wrap displacements into [-L/2, L/2] so periodic jumps aren't counted."""
    return delta - box_length * np.round(delta / box_length)


def step_speeds(positions: np.ndarray, box_length: float) -> np.ndarray:
    """Per-particle distance moved since the previous frame, shape (F, N)."""
    deltas = minimum_image(np.diff(positions, axis=0), box_length)
    speeds = np.linalg.norm(deltas, axis=2)
    # First frame has no predecessor; reuse the second frame's values so the
    # colouring doesn't flash on frame 0.
    return np.concatenate([speeds[:1], speeds], axis=0) if len(speeds) else np.zeros(positions.shape[:2])


def motion_heat(speeds: np.ndarray, decay: float = 0.72) -> np.ndarray:
    """Speed with an exponential tail, so a particle that just moved glows and
    then fades over the next few frames.

    A single Metropolis sweep only accepts a move for a fraction of the
    particles, so raw per-frame speed is zero for most of them and the cloud
    renders as mostly-black dots. Carrying a decaying memory of the last move
    keeps every recently-active particle visible and makes the sampling itself
    legible as motion.
    """
    heat = np.empty_like(speeds)
    carried = speeds[0]
    heat[0] = carried
    for index in range(1, len(speeds)):
        carried = np.maximum(speeds[index], carried * decay)
        heat[index] = carried
    return heat


def local_density(positions: np.ndarray, box_length: float, cutoff: float) -> np.ndarray:
    """Smooth local density around each particle, shape (F, N).

    A hard neighbour count quantises into a handful of integers and renders as
    colour banding, so neighbours are weighted by a Gaussian kernel instead,
    which varies continuously. Unlike speed this is non-zero for every
    particle, and it exposes the structure the Jastrow factor exists to create:
    the correlation hole shows up as a deficit of close neighbours.
    """
    density = np.empty(positions.shape[:2])
    for index, frame in enumerate(positions):
        delta = minimum_image(frame[:, None, :] - frame[None, :, :], box_length)
        distances = np.linalg.norm(delta, axis=-1)
        weights = np.exp(-((distances / cutoff) ** 2))
        np.fill_diagonal(weights, 0.0)  # drop self
        density[index] = weights.sum(axis=1)
    return density


def grid_density(points: np.ndarray, box_length: float, cells: int = 26) -> np.ndarray:
    """Density at each point, from a coarse occupancy grid.

    Pairwise distances over a cloud this size would be a ~200M-element matrix,
    so the cloud is binned once and every point reads back its own cell count.
    """
    edges = np.linspace(0.0, box_length, cells + 1)
    wrapped = np.mod(points, box_length)
    counts, _ = np.histogramdd(wrapped, bins=(edges, edges, edges))
    index = np.clip((wrapped / box_length * cells).astype(int), 0, cells - 1)
    return counts[index[:, 0], index[:, 1], index[:, 2]]


def sampling_cloud(positions: np.ndarray, box_length: float, max_points: int, rng):
    """Every position the walkers visited, as one point cloud.

    A single frame is 123 dots; the *cloud* is what the Monte Carlo is actually
    sampling, so accumulating every configuration is both the honest picture of
    |psi|^2 and the one that reads as a gas rather than confetti.
    """
    cloud = positions.reshape(-1, 3)
    if len(cloud) > max_points:
        cloud = cloud[rng.choice(len(cloud), max_points, replace=False)]
    return cloud, grid_density(cloud, box_length)


def correlation_cloud(
    positions: np.ndarray, box_length: float, max_points: int, rng
):
    """Pair displacements r_j - r_i, stacked at the origin.

    This is g(r) as a picture: the exchange-correlation hole becomes a literal
    void at the centre of the cloud, because electrons never sit on top of one
    another. Symmetrised (+d and -d) since every pair contributes both.
    """
    upper = np.triu_indices(positions.shape[1], k=1)
    r_max = box_length / 2.0

    chunks = []
    for frame in positions:
        delta = minimum_image(frame[:, None, :] - frame[None, :, :], box_length)
        delta = delta[upper]
        delta = delta[np.linalg.norm(delta, axis=1) < r_max]
        chunks.append(delta)

    cloud = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    cloud = np.concatenate([cloud, -cloud])
    if len(cloud) > max_points:
        cloud = cloud[rng.choice(len(cloud), max_points, replace=False)]

    radii = np.linalg.norm(cloud, axis=1)
    return cloud, radii


def occupied_orbitals(num_orbitals: int, box_length: float):
    """Rebuild the plane-wave basis the Slater determinant actually uses.

    Mirrors SlaterPlaneWave::initialize(): canonical half-space n-vectors
    (first non-zero component positive, see utilities/matrix.hpp is_canonical),
    sorted by |n|^2 then lexicographically, each contributing cos(k.r) and --
    for n != 0 -- sin(k.r), until num_orbitals are filled.

    Returns (n_vectors, k_vectors, is_sine) for the occupied orbitals.
    """
    n_max = int(np.ceil(np.cbrt(num_orbitals))) + 2
    span = np.arange(-n_max, n_max + 1)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)

    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]
    canonical = (x > 0) | ((x == 0) & (y > 0)) | ((x == 0) & (y == 0) & (z >= 0))
    grid = grid[canonical]

    magnitude = (grid**2).sum(axis=1)
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0], magnitude))
    grid = grid[order]

    n_vectors, is_sine = [], []
    for candidate in grid:
        if len(n_vectors) >= num_orbitals:
            break
        n_vectors.append(candidate)
        is_sine.append(False)
        if candidate.any() and len(n_vectors) < num_orbitals:
            n_vectors.append(candidate)
            is_sine.append(True)

    n_vectors = np.array(n_vectors[:num_orbitals])
    return n_vectors, n_vectors * (2.0 * np.pi / box_length), np.array(is_sine[:num_orbitals])


def orbital_field(k_vector, sine: bool, box_length: float, cells: int, keep: float):
    """Sample one plane-wave orbital on a grid, keeping only the strong lobes.

    Returns (points, amplitude). Dropping everything near the nodal surfaces is
    what makes the wavefronts legible -- the full grid is a solid block.
    """
    axis = (np.arange(cells) + 0.5) * (box_length / cells)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    phase = points @ k_vector
    amplitude = np.sin(phase) if sine else np.cos(phase)

    mask = np.abs(amplitude) > keep
    return points[mask], amplitude[mask]


def diverging_colors(amplitude: np.ndarray) -> np.ndarray:
    """Blue for negative lobes, orange for positive, dark through the node."""
    magnitude = np.clip(np.abs(amplitude), 0.0, 1.0)
    positive = amplitude >= 0.0
    colors = np.empty((len(amplitude), 4), dtype=np.uint8)
    colors[positive, 0] = (90 + 165 * magnitude[positive]).astype(np.uint8)
    colors[positive, 1] = (40 + 130 * magnitude[positive]).astype(np.uint8)
    colors[positive, 2] = 40
    colors[~positive, 0] = 40
    colors[~positive, 1] = (90 + 110 * magnitude[~positive]).astype(np.uint8)
    colors[~positive, 2] = (110 + 145 * magnitude[~positive]).astype(np.uint8)
    colors[:, 3] = (40 + 150 * magnitude).astype(np.uint8)
    return colors


def trail_strips(history: np.ndarray, box_length: float):
    """Build one polyline per particle from recent positions.

    `history` is (K, N, 3), oldest first. A particle that wraps across a face
    would otherwise draw a line straight through the box, so each trail is cut
    at its most recent wrap and only the contiguous tail is kept.
    """
    if len(history) < 2:
        return []

    steps = np.linalg.norm(np.diff(history, axis=0), axis=2)  # (K-1, N)
    wrapped = steps > box_length / 2.0

    strips = []
    for particle in range(history.shape[1]):
        breaks = np.nonzero(wrapped[:, particle])[0]
        start = int(breaks[-1]) + 1 if len(breaks) else 0
        if history.shape[0] - start >= 2:
            strips.append(history[start:, particle, :])
    return strips


def pair_correlation(positions: np.ndarray, box_length: float, bins: int = 40):
    """Radial distribution g(r), averaged over the supplied frames.

    Normalised against an ideal gas of the same density, so g -> 1 at large r
    and the dip at small r is the exchange-correlation hole.
    """
    frames, num_particles, _ = positions.shape
    r_max = box_length / 2.0
    edges = np.linspace(0.0, r_max, bins + 1)
    upper = np.triu_indices(num_particles, k=1)

    counts = np.zeros(bins)
    for frame in positions:
        delta = minimum_image(frame[:, None, :] - frame[None, :, :], box_length)
        distances = np.linalg.norm(delta, axis=-1)[upper]
        counts += np.histogram(distances, bins=edges)[0]

    shell_volume = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = num_particles / box_length**3
    ideal = 0.5 * num_particles * density * shell_volume * frames

    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(ideal > 0, counts / ideal, 0.0)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, g


def box_edges(box_length: float):
    """The 12 edges of the simulation cell as separate line strips."""
    corners = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    )
    pairs = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    return [corners[[a, b]] * box_length for a, b in pairs]


DARK = [0x07, 0x09, 0x10]


def build_blueprint(energy_range, accept_range, error_range):
    """Explicit layout, so the viewer opens composed instead of auto-arranged."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Tabs(
                rrb.Spatial3DView(
                    origin="/world",
                    name="Electron gas",
                    # The default gradient washes out the cool end of the ramp;
                    # a near-black cell makes the colours carry.
                    background=rrb.Background(color=DARK),
                ),
                rrb.Spatial3DView(
                    origin="/orbital",
                    name="Orbitals",
                    background=rrb.Background(color=DARK),
                ),
                rrb.Spatial3DView(
                    origin="/correlation",
                    name="Correlation hole",
                    background=rrb.Background(color=DARK),
                ),
                rrb.Spatial3DView(
                    origin="/kspace",
                    name="Fermi sea",
                    background=rrb.Background(color=DARK),
                ),
            ),
            rrb.Vertical(
                rrb.Tabs(
                    rrb.TimeSeriesView(
                        origin="/scalars",
                        name="Energy",
                        contents=["+ /scalars/local_energy", "+ /scalars/mean_energy"],
                        axis_y=rrb.ScalarAxis(range=energy_range, zoom_lock=True),
                        plot_legend=rrb.PlotLegend(visible=True),
                    ),
                    rrb.TimeSeriesView(
                        origin="/scalars",
                        name="Acceptance",
                        contents=["+ /scalars/acceptance_rate"],
                        axis_y=rrb.ScalarAxis(range=accept_range, zoom_lock=True),
                    ),
                    rrb.TimeSeriesView(
                        origin="/scalars",
                        name="Std. error",
                        contents=["+ /scalars/standard_error"],
                        axis_y=rrb.ScalarAxis(range=error_range, zoom_lock=True),
                    ),
                ),
                rrb.BarChartView(origin="/structure/g_r", name="Pair correlation g(r)"),
                rrb.TextDocumentView(origin="/run", name="Run"),
                row_shares=[3, 3, 2],
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def padded_range(values, pad_fraction=0.08, low_quantile=0.0, high_quantile=1.0):
    low = float(np.quantile(values, low_quantile))
    high = float(np.quantile(values, high_quantile))
    if high <= low:
        high = low + 1.0
    pad = (high - low) * pad_fraction
    return (low - pad, high + pad)


def main():
    parser = argparse.ArgumentParser(description="Render VMC binary output with Rerun")
    parser.add_argument("--input", default="output/vmc.bin", help="Path to binary file")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame")
    parser.add_argument(
        "--trail",
        type=int,
        default=12,
        help="Length of the motion trail behind each particle (0 disables)",
    )
    parser.add_argument(
        "--color-by",
        choices=("motion", "density", "energy", "none"),
        default="motion",
        help="Per-particle colouring (default: recent motion, with a decaying tail)",
    )
    parser.add_argument(
        "--no-cloud",
        dest="cloud",
        action="store_false",
        help="Skip the accumulated sampling and pair-correlation clouds",
    )
    parser.add_argument(
        "--cloud-points", type=int, default=60000, help="Cap on points per cloud"
    )
    parser.add_argument(
        "--cloud-alpha", type=int, default=120, help="Cloud point opacity, 0-255"
    )
    parser.add_argument(
        "--orbitals", type=int, default=16, help="Plane-wave orbitals to sample (0 disables)"
    )
    parser.add_argument(
        "--orbital-cells", type=int, default=34, help="Grid resolution per axis for orbitals"
    )
    args = parser.parse_args()

    path = args.input
    stride = max(1, args.stride)

    num_particles, box_length, measure_steps = read_header(path)
    total_frames = (os.path.getsize(path) - HEADER_SIZE) // frame_bytes(num_particles)
    print(f"Particles: {num_particles}  Box: {box_length}  Steps: {measure_steps}")

    scalars, positions = read_all(path, num_particles, stride)
    frames = len(scalars)
    print(f"Frames: {total_frames}  Stride: {stride}  Rendering: {frames}")
    if frames == 0:
        sys.exit("No frames in recording")

    local_e, mean_e, error, accept = (scalars[:, i] for i in range(SCALARS_PER_FRAME))

    energy_lo, energy_hi = float(local_e.min()), float(local_e.max())
    energy_span = (energy_hi - energy_lo) or 1.0

    # Precompute whichever per-particle field drives the colour, then normalise
    # it once against a robust range so the ramp is stable across the playback
    # instead of rescaling every frame.
    if args.color_by == "motion":
        field = motion_heat(step_speeds(positions, box_length))
    elif args.color_by == "density":
        r_s = (3.0 * box_length**3 / (4.0 * np.pi * num_particles)) ** (1.0 / 3.0)
        field = local_density(positions, box_length, cutoff=1.6 * r_s)
    else:
        field = None

    if field is not None:
        # Clip at the 2nd/98th percentile so one outlier doesn't flatten the rest.
        low = float(np.quantile(field, 0.02))
        high = float(np.quantile(field, 0.98))
        field = (field - low) / ((high - low) or 1.0)

    print("Computing pair correlation...")
    gr_centers, gr_values = pair_correlation(positions[-min(frames, 60) :], box_length)

    rr.init("VMC_Simulation", spawn=True)

    nonzero_error = error[error != 0.0]
    rr.send_blueprint(
        build_blueprint(
            padded_range(np.concatenate([local_e, mean_e])),
            padded_range(accept, pad_fraction=0.15),
            padded_range(nonzero_error) if len(nonzero_error) else (0.0, 1.0),
        ),
        make_active=True,
        make_default=True,
    )

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/box",
        rr.LineStrips3D(
            box_edges(box_length),
            colors=[COLOR_BOX] * 12,
            radii=[box_length * 0.0012] * 12,
        ),
        static=True,
    )

    for entity, color, width, name in (
        ("scalars/local_energy", COLOR_LOCAL, 1.0, "local"),
        ("scalars/mean_energy", COLOR_MEAN, 2.5, "running mean"),
        ("scalars/standard_error", COLOR_ERROR, 2.0, "std. error"),
        ("scalars/acceptance_rate", COLOR_ACCEPT, 2.0, "acceptance"),
    ):
        rr.log(entity, rr.SeriesLines(colors=[color], widths=[width], names=[name]), static=True)

    r_s = (3.0 / (4.0 * np.pi * num_particles / box_length**3)) ** (1.0 / 3.0)
    rr.log(
        "run",
        rr.TextDocument(
            f"## Variational Monte Carlo\n\n"
            f"| | |\n|---|---|\n"
            f"| Particles | {num_particles} |\n"
            f"| Box length | {box_length:.3f} a0 |\n"
            f"| Wigner-Seitz r_s | {r_s:.2f} a0 |\n"
            f"| Frames | {frames} (stride {stride}) |\n"
            f"| Mean energy | {mean_e[-1]:.6f} Ha |\n"
            f"| Std. error | {error[-1]:.6f} |\n"
            f"| Acceptance | {accept[-1] * 100.0:.2f}% |\n",
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    rr.log(
        "structure/g_r",
        rr.BarChart(gr_values, color=COLOR_MEAN, abscissa=gr_centers),
        static=True,
    )

    rng = np.random.default_rng(0)

    # -- The sampling cloud -------------------------------------------------
    # Every configuration the walkers visited, overlaid. This is the picture of
    # what the Monte Carlo is actually doing; a single frame is just dots.
    if args.cloud:
        print("Building sampling cloud...")
        cloud, cloud_density = sampling_cloud(positions, box_length, args.cloud_points, rng)
        shade = cloud_density / (np.quantile(cloud_density, 0.97) or 1.0)
        cloud_colors = np.concatenate(
            [
                colormap(0.10 + 0.75 * np.clip(shade, 0.0, 1.0)),
                np.full((len(cloud), 1), args.cloud_alpha, dtype=np.uint8),
            ],
            axis=1,
        )
        rr.log(
            "world/cloud",
            rr.Points3D(cloud, colors=cloud_colors, radii=box_length * 0.0030),
            static=True,
        )
        print(f"  {len(cloud)} points")

        # -- The correlation hole, in 3D ------------------------------------
        # Pair separations stacked at the origin: the void at the centre is the
        # same physics as the g(r) dip, but you can see straight into it.
        hole_frames = positions[-min(frames, 40) :]
        hole, hole_radii = correlation_cloud(hole_frames, box_length, args.cloud_points, rng)
        if len(hole):
            hole_shade = hole_radii / (hole_radii.max() or 1.0)
            hole_colors = np.concatenate(
                [
                    colormap(0.15 + 0.80 * hole_shade),
                    np.full((len(hole), 1), args.cloud_alpha, dtype=np.uint8),
                ],
                axis=1,
            )
            rr.log(
                "correlation/pairs",
                rr.Points3D(hole, colors=hole_colors, radii=box_length * 0.0022),
                static=True,
            )
            print(f"  correlation cloud: {len(hole)} pair separations")

    # -- The plane-wave basis ----------------------------------------------
    n_vectors, k_vectors, is_sine = occupied_orbitals(num_particles, box_length)
    shells = (n_vectors**2).sum(axis=1)

    # Occupied k-points: the discrete shells that make N = 7, 19, 27, 33, 57...
    # the "closed shell" particle counts the config file lists.
    rr.log(
        "kspace/occupied",
        rr.Points3D(
            k_vectors,
            colors=colormap(shells / (shells.max() or 1)),
            radii=float(np.max(np.abs(k_vectors)) or 1.0) * 0.035,
        ),
        static=True,
    )
    rr.log(
        "kspace/axes",
        rr.LineStrips3D(
            [np.array([[-1.0, 0, 0], [1.0, 0, 0]]) * float(np.max(np.abs(k_vectors))) * 1.2,
             np.array([[0, -1.0, 0], [0, 1.0, 0]]) * float(np.max(np.abs(k_vectors))) * 1.2,
             np.array([[0, 0, -1.0], [0, 0, 1.0]]) * float(np.max(np.abs(k_vectors))) * 1.2],
            colors=[[60, 80, 110]] * 3,
            radii=float(np.max(np.abs(k_vectors))) * 0.004,
        ),
        static=True,
    )

    # Orbital wavefronts, on their own timeline so scrubbing through the basis
    # is separate from scrubbing through simulation time.
    shown = min(args.orbitals, len(k_vectors))
    if shown:
        print(f"Sampling {shown} orbital fields...")
        for index in range(shown):
            rr.set_time("orbital", sequence=index)
            points, amplitude = orbital_field(
                k_vectors[index], bool(is_sine[index]), box_length, args.orbital_cells, 0.45
            )
            rr.log(
                "orbital/psi",
                rr.Points3D(
                    points,
                    colors=diverging_colors(amplitude),
                    radii=box_length / args.orbital_cells * 0.30,
                ),
            )
            rr.log(
                "orbital/label",
                rr.TextDocument(
                    f"**Orbital {index}** of {num_particles} — "
                    f"{'sin' if is_sine[index] else 'cos'}(k·r), "
                    f"n = ({n_vectors[index][0]}, {n_vectors[index][1]}, {n_vectors[index][2]}), "
                    f"shell |n|² = {shells[index]}",
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )
        rr.set_time("orbital", sequence=0)

    print("Rendering frames...")
    base_radius = box_length * 0.0075
    trail = max(0, args.trail)

    for index in range(frames):
        rr.set_time("frame", sequence=index * stride)

        if field is not None:
            intensity = field[index]
        elif args.color_by == "energy":
            intensity = np.full(num_particles, (local_e[index] - energy_lo) / energy_span)
        else:
            intensity = np.full(num_particles, 0.6)

        # Floor the ramp: a particle at the bottom of the scale should still be
        # a visible cool blue, not black-on-black.
        intensity = 0.18 + 0.82 * np.clip(intensity, 0.0, 1.0)
        rr.log(
            "world/electrons",
            rr.Points3D(
                positions=positions[index],
                colors=colormap(intensity),
                # Active particles read slightly larger, selling the motion at a glance.
                radii=base_radius * (0.65 + 0.9 * intensity),
            ),
        )

        if trail:
            history = positions[max(0, index - trail) : index + 1]
            strips = trail_strips(history, box_length)
            if strips:
                rr.log(
                    "world/trails",
                    rr.LineStrips3D(
                        strips,
                        colors=[[*COLOR_MEAN, 130]] * len(strips),
                        radii=[base_radius * 0.13] * len(strips),
                    ),
                )

        rr.log("scalars/local_energy", rr.Scalars(local_e[index]))
        rr.log("scalars/mean_energy", rr.Scalars(mean_e[index]))
        if error[index] != 0.0:
            rr.log("scalars/standard_error", rr.Scalars(error[index]))
        rr.log("scalars/acceptance_rate", rr.Scalars(accept[index]))

    print(f"Done. {frames} frames logged to Rerun.")


if __name__ == "__main__":
    main()
