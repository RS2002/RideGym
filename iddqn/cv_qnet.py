"""CV-Net: coordinate-value Q-network with multi-scale grid position embeddings.

A classic ride-sharing design choice: instead of feeding raw continuous
coordinates through a linear layer, every location is DISCRETISED onto grids of
several resolutions and encoded by a learnable embedding table (one embedding
per grid cell) -- exactly like token embeddings in NLP. The service area is
partitioned by ``n`` grids of increasing granularity (small / medium / high);
a coordinate lands in one cell per grid, yielding ``n`` cell embeddings that are
averaged into a single position embedding.

Why multi-scale: the coarse grid groups nearby locations so their embeddings are
shared (spatial smoothing / generalisation, robust to demand sparsity), while
the fine grid preserves resolution so distinct locations stay distinguishable.
Averaging fuses generalisation and precision.

Interface parity with :class:`iddqn.qnet.PairQNet`
--------------------------------------------------
CVNet is a DROP-IN replacement for ``PairQNet``: it consumes the SAME
pre-assembled ``[driver_feat ++ order_feat]`` pair vector every caller already
builds (driver first, order second) and returns a scalar Q. It therefore reuses
the existing :class:`iddqn.agent.IDDQNAgent`, :class:`iddqn.inference.IDDQNActor`
and :func:`iddqn.inference.q_matrix_for_state` pipeline unchanged -- only the
POSITION-ENCODING is different, so a CVNet-vs-PairQNet comparison isolates that
single factor.

Coordinate layout (fixed by :class:`iddqn.features.FeatureEncoder`)
------------------------------------------------------------------
The encoder places NORMALISED coordinates (in ``[0, 1]`` after dividing by the
service-area extent) at the front of each side's feature vector:

* driver features : ``[x, y, <status one-hot + capacity/onboard/committed/time>]``
  -> the driver location is columns ``[0, 1]``.
* order  features : ``[ox, oy, dx, dy, <party, wait, dummy_flag>]``
  -> the order origin is columns ``[0, 1]`` and destination ``[2, 3]``.

CVNet slices those normalised coordinates back out (no upstream change needed),
maps each ``(x, y)`` to its cell in every grid, and looks up / averages the
embeddings. The remaining (non-coordinate) features are concatenated with the
position embeddings and passed through the two-tower head, mirroring PairQNet.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Hexagonal-grid geometry constants.
#
# The paper (and Uber's H3) partition space with a HIERARCHY of hexagons where
# each finer level has 1/7 the AREA of its parent (so ~7x as many cells). Since
# hexagon area scales with the square of the edge length, the edge-length ratio
# between adjacent levels is 1/sqrt(7); we halve/scale the hex "size" (the
# centre-to-vertex circumradius) by this factor at every level. Hexagons are the
# natural spatial-embedding cell because a pointy-top hexagon is EQUIDISTANT to
# all six of its neighbours, unlike a square whose diagonal neighbours are
# sqrt(2) farther -- so neighbourhood generalisation is isotropic.
# --------------------------------------------------------------------------- #
HEX_AREA_RATIO = 1.0 / 7.0                 # child area / parent area (H3-style)
HEX_EDGE_RATIO = math.sqrt(HEX_AREA_RATIO)  # child edge / parent edge = 1/sqrt(7)
_SQRT3 = math.sqrt(3.0)
_SQRT7 = math.sqrt(7.0)

# Default per-level hex "resolutions" (cells-per-axis) encoding the H3-style
# hierarchy: each finer level packs sqrt(7) x as many hexes per axis, so its
# hexes have 1/7 the AREA (1/sqrt(7) the edge length) of the parent's. Starting
# from a coarse 4-across grid this gives (4, 4*sqrt(7), 4*7) = (4, ~10.58, 28),
# the hexagonal analogue of the square default (4, 16, 64). Override with any
# resolutions you like -- this default merely bakes in the 1/7-area relation.
HEX_DEFAULT_RESOLUTIONS = (4.0, 4.0 * _SQRT7, 28.0)


def _axial_round(qf: torch.Tensor, rf: torch.Tensor):
    """Round fractional axial hex coords ``(qf, rf)`` to the nearest hex centre.

    Uses cube-coordinate rounding (the standard robust hex-rounding): convert
    axial -> cube ``(x=q, z=r, y=-x-z)``, round each, then repair the component
    with the largest rounding residual so the cube constraint ``x+y+z == 0``
    holds exactly. Returns integer axial ``(q, r)`` tensors (same shape as input).
    This is what makes hex cells equidistant-to-neighbours: a point is assigned
    to the geometrically nearest hexagon centre, not a rectangular bucket.
    """
    yf = -qf - rf
    xr = torch.round(qf)
    yr = torch.round(yf)
    zr = torch.round(rf)
    dx = torch.abs(xr - qf)
    dy = torch.abs(yr - yf)
    dz = torch.abs(zr - rf)
    # Repair the coordinate with the largest rounding residual so x+y+z == 0.
    # Only x (=q) and z (=r) are returned, so we never need to repair y: if y
    # has the largest residual it is fixed implicitly (x and z are already the
    # correct rounded values); otherwise repair whichever of x / z is larger.
    fix_x = (dx > dy) & (dx > dz)
    fix_z = (~fix_x) & (dz > dy)
    # torch.where on the two returned coords only (2 ops instead of the old 4).
    q = torch.where(fix_x, -yr - zr, xr)
    r = torch.where(fix_z, -q - yr, zr)
    return q.long(), r.long()


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    """Build an MLP ``in_dim -> hidden... -> out_dim`` with ReLU between layers."""
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(nn.ReLU())
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class MultiScaleGridEmbedding(nn.Module):
    """Encode a normalised ``(x, y)`` coordinate via multi-scale grid embeddings.

    The area ``[0, 1]^2`` is tiled by ``len(resolutions)`` square grids. Grid
    ``k`` has ``resolutions[k]`` cells per axis (so ``r_k^2`` cells total) and
    owns a learnable embedding table of shape ``[r_k^2, embed_dim]``. A point is
    mapped to its (row, col) cell in each grid, the corresponding embeddings are
    gathered, and the ``n`` embeddings are aggregated (mean by default) into one
    position embedding.

    Parameters
    ----------
    resolutions:
        Cells-per-axis for each grid, e.g. ``(4, 16, 64)`` for a small / medium
        / high-granularity trio. All entries must be >= 1.
    embed_dim:
        Width of each per-grid embedding (and, under ``aggregate="mean"``, of
        the final position embedding).
    aggregate:
        ``"mean"`` (default) averages the ``n`` grid embeddings -> output width
        ``embed_dim``. ``"concat"`` concatenates them -> output width
        ``n * embed_dim`` (see :attr:`out_dim`).
    """

    def __init__(
        self,
        resolutions: Sequence[int] = (4, 16, 64),
        embed_dim: int = 16,
        aggregate: str = "mean",
    ):
        super().__init__()
        res = [int(r) for r in resolutions]
        if not res or any(r < 1 for r in res):
            raise ValueError(
                f"resolutions must be a non-empty sequence of positive ints, "
                f"got {resolutions!r}."
            )
        if aggregate not in ("mean", "concat"):
            raise ValueError("aggregate must be 'mean' or 'concat'.")
        self.resolutions = res
        self.embed_dim = int(embed_dim)
        self.aggregate = aggregate
        # One embedding table per grid: r_k^2 cells -> embed_dim.
        self.tables = nn.ModuleList(
            [nn.Embedding(r * r, self.embed_dim) for r in res]
        )
        # Small init so early Q-values start near zero (stable bootstrap).
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)

    @property
    def out_dim(self) -> int:
        """Width of the position embedding this module produces."""
        if self.aggregate == "mean":
            return self.embed_dim
        return self.embed_dim * len(self.resolutions)

    def _cell_index(self, xy: torch.Tensor, res: int) -> torch.Tensor:
        """Row-major cell index of each ``(x, y)`` in a ``res x res`` grid.

        ``xy`` is ``[*, 2]`` normalised to ``[0, 1]``. Coordinates are clamped
        into ``[0, 1]`` (a driver/order may sit marginally outside the modelled
        box) and the top edge maps to the last cell (index ``res - 1``) rather
        than overflowing to ``res``.
        """
        xy = xy.clamp(0.0, 1.0)
        # floor(coord * res), capped at res-1 so coord == 1.0 stays in-range.
        col = torch.clamp((xy[..., 0] * res).long(), max=res - 1)
        row = torch.clamp((xy[..., 1] * res).long(), max=res - 1)
        return row * res + col  # row-major flatten

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Encode ``xy`` ``[*, 2]`` (normalised) into a position embedding.

        Returns ``[*, out_dim]``: the per-grid embeddings averaged (mean) or
        concatenated (concat).
        """
        embs = []
        for table, res in zip(self.tables, self.resolutions):
            idx = self._cell_index(xy, res)  # [*]
            embs.append(table(idx))          # [*, embed_dim]
        if self.aggregate == "mean":
            # Stack over the grid axis and average -> [*, embed_dim].
            return torch.stack(embs, dim=0).mean(dim=0)
        # concat over the last dim -> [*, n * embed_dim].
        return torch.cat(embs, dim=-1)


class MultiScaleHexEmbedding(nn.Module):
    """Encode a normalised ``(x, y)`` via multi-scale HEXAGONAL grid embeddings.

    The hexagonal counterpart of :class:`MultiScaleGridEmbedding`, mirroring the
    paper's H3-style hierarchy. The area ``[0, 1]^2`` is tiled by
    ``num_levels`` pointy-top hexagonal grids; level 0 is the coarsest and each
    finer level shrinks the hexagon so its AREA is ``1/7`` of its parent's
    (edge-length ratio ``1/sqrt(7)`` -- see :data:`HEX_EDGE_RATIO`). A point is
    mapped to the NEAREST hexagon centre at each level (via cube rounding, so
    the assignment is truly nearest-centre and hence equidistant-to-neighbours),
    and the per-level embeddings are aggregated (mean by default).

    Why hexagons: a pointy-top hexagon is equidistant to all six neighbours,
    whereas a square's diagonal neighbours are ``sqrt(2)`` farther. Equidistant
    neighbours give isotropic spatial smoothing, which is why H3 / the original
    CV-Net paper prefer hexagonal cells.

    Cell hashing
    ------------
    A hexagonal tiling of a rectangle does not yield a clean ``r x r`` cell
    enumeration, so each level owns an embedding table of a fixed capacity
    (``table_size``) and integer axial coords ``(q, r)`` are hashed into it
    (a spatial-hash embedding, like feature hashing). ``table_size`` is chosen
    per level to comfortably exceed the number of hexes actually covering the
    unit square at that resolution, so collisions are rare.

    Parameters
    ----------








    resolutions:
        Approximate number of hexagons spanning one axis at each level (the hex
        analogue of the square grid's cells-per-axis), coarse -> fine. Defaults
        to :data:`HEX_DEFAULT_RESOLUTIONS` ``(4, 4*sqrt(7), 28)``, which bakes in
        the H3-style 1/7-area hierarchy (each level has sqrt(7)x the per-axis
        resolution -> 1/7 the hex area). You may pass ANY resolutions (e.g.
        ``(4, 8, 16)``); values may be non-integer since they set a continuous
        hex size. All entries must be > 0.
    embed_dim:
        Width of each per-level embedding (and, under ``aggregate="mean"``, of
        the final position embedding).
    aggregate:
        ``"mean"`` (default) averages the levels -> width ``embed_dim``;
        ``"concat"`` concatenates them -> width ``num_levels * embed_dim``.
    table_capacity_factor:
        Multiplier on the estimated hex count per level when sizing that level's
        hash table (larger -> fewer hash collisions, more parameters).
    """

    def __init__(
        self,


        resolutions: Sequence[float] = HEX_DEFAULT_RESOLUTIONS,
        embed_dim: int = 16,
        aggregate: str = "mean",
        table_capacity_factor: float = 4.0,
    ):
        super().__init__()




        res = [float(r) for r in resolutions]
        if not res or any(r <= 0 for r in res):
            raise ValueError(
                f"resolutions must be a non-empty sequence of positive numbers, "
                f"got {resolutions!r}."
            )
        if aggregate not in ("mean", "concat"):
            raise ValueError("aggregate must be 'mean' or 'concat'.")


        self.resolutions = res
        self.num_levels = len(res)
        self.embed_dim = int(embed_dim)
        self.aggregate = aggregate










        # Per-level hex circumradius (size) from its resolution. For pointy-top
        # hexes the horizontal spacing between adjacent hex centres is
        # sqrt(3) * size, so to fit ``res`` hexes across the unit width:
        #   size = 1 / (res * sqrt(3)).
        # With the default resolutions this reproduces the 1/sqrt(7) size ratio
        # (1/7 area) between adjacent levels automatically.
        self._sizes = [1.0 / (r * _SQRT3) for r in res]
        # Per-level hash-table capacity: estimate the number of hexes covering
        # the unit square (area 1 / hex_area) and inflate by the factor. Hex
        # area = (3*sqrt(3)/2) * size^2.
        self._table_sizes = []
        for size in self._sizes:
            hex_area = (3.0 * _SQRT3 / 2.0) * (size * size)
            n_hex = max(1.0, 1.0 / hex_area)
            cap = int(math.ceil(n_hex * table_capacity_factor)) + 1
            self._table_sizes.append(cap)

        self.tables = nn.ModuleList(
            [nn.Embedding(cap, self.embed_dim) for cap in self._table_sizes]
        )
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)

    @property
    def out_dim(self) -> int:
        """Width of the position embedding this module produces."""
        if self.aggregate == "mean":
            return self.embed_dim

        return self.embed_dim * len(self.resolutions)

    def _hex_hash(self, xy: torch.Tensor, size: float, table_size: int) -> torch.Tensor:
        """Map ``(x, y)`` to a hashed hex-cell index for a given hex ``size``.

        Pixel -> fractional axial coords (pointy-top layout), cube-round to the
        nearest hexagon centre, then hash the integer axial ``(q, r)`` into
        ``[0, table_size)``. Coordinates are clamped to ``[0, 1]`` first.
        """
        xy = xy.clamp(0.0, 1.0)
        x = xy[..., 0]
        y = xy[..., 1]
        # Pointy-top pixel -> axial (inverse of the standard hex layout matrix).
        qf = (_SQRT3 / 3.0 * x - 1.0 / 3.0 * y) / size
        rf = (2.0 / 3.0 * y) / size
        q, r = _axial_round(qf, rf)
        # Spatial hash of the integer axial pair into the table. Two large odd
        # primes decorrelate the axes; modulo folds into the table range. Add an
        # offset so negative coords hash into range as well.
        h = (q * 92837111) ^ (r * 689287499)
        return (h % table_size + table_size) % table_size

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Encode ``xy`` ``[*, 2]`` (normalised) into a position embedding.

        Returns ``[*, out_dim]``: the per-level hex embeddings averaged (mean)
        or concatenated (concat).
        """
        embs = []
        for table, size, tsize in zip(self.tables, self._sizes, self._table_sizes):
            idx = self._hex_hash(xy, size, tsize)   # [*]
            embs.append(table(idx))                 # [*, embed_dim]
        if self.aggregate == "mean":
            return torch.stack(embs, dim=0).mean(dim=0)
        return torch.cat(embs, dim=-1)


class CVNet(nn.Module):
    """Two-tower Q-net with multi-scale grid position embeddings (CV-Net).

    Drop-in for :class:`iddqn.qnet.PairQNet`: same ``forward(pairs) -> Q``
    contract over the pre-concatenated ``[driver_feat ++ order_feat]`` vector.
    The ONLY difference is that positions are encoded by grid embeddings rather
    than passed as raw continuous values, so results are directly comparable.

    Feature split (see module docstring): the first ``driver_dim`` columns are
    the driver features (location at cols 0-1); the remaining columns are the
    order features (origin at cols 0-1, destination at cols 2-3, relative to the
    order block). Non-coordinate features on each side are kept and concatenated
    with the position embeddings.

    Parameters
    ----------
    pair_dim:
        Width of the concatenated ``[driver_feat, order_feat]`` input.
    driver_dim:
        Split point: first ``driver_dim`` entries are driver features.
    grid_type:
        Spatial-cell shape for the position embedding:

        * ``"square"`` (default) -- axis-aligned square grids at each resolution
          in ``resolutions`` (see :class:`MultiScaleGridEmbedding`).
        * ``"hex"`` -- H3-style pointy-top HEXAGONAL grids (see
          :class:`MultiScaleHexEmbedding`), each level 1/7 the area of its
          parent. Hexagons are equidistant to all six neighbours, giving
          isotropic spatial generalisation as in the original CV-Net paper.
    resolutions:
        (``grid_type="square"`` only) cells-per-axis of the grids
        (small / medium / high), default ``(4, 16, 64)``.
    hex_resolutions:
        (``grid_type="hex"`` only) per-level hexagons-per-axis, coarse -> fine.
        Defaults to :data:`HEX_DEFAULT_RESOLUTIONS` ``(4, 4*sqrt(7), 28)`` which
        encodes the H3-style 1/7-area hierarchy; override with any resolutions.
    pos_embed_dim:
        Per-grid / per-level position embedding width.
    hidden:
        Hidden widths for the two per-side towers and the fusion head.
    embed_dim:
        Per-tower output embedding width (fusion head sees ``2 * embed_dim``).
    aggregate:
        Multi-scale aggregation, ``"mean"`` (default) or ``"concat"``.
    """

    def __init__(
        self,
        pair_dim: int,
        driver_dim: int,
        grid_type: str = "square",
        resolutions: Sequence[int] = (4, 16, 64),
        hex_resolutions: Sequence[float] = HEX_DEFAULT_RESOLUTIONS,
        pos_embed_dim: int = 16,
        hidden: Sequence[int] = (128, 128),
        embed_dim: int = 64,
        aggregate: str = "mean",
    ):
        super().__init__()
        self.pair_dim = int(pair_dim)
        self.driver_dim = int(driver_dim)
        if not (0 < self.driver_dim < self.pair_dim):
            raise ValueError(
                f"driver_dim ({driver_dim}) must be in (0, pair_dim) "
                f"= (0, {self.pair_dim})."
            )
        self.order_dim = self.pair_dim - self.driver_dim
        if self.driver_dim < 2:
            raise ValueError(
                "driver features must contain at least the 2 location columns."
            )
        if self.order_dim < 4:
            raise ValueError(
                "order features must contain at least the 4 origin/dest columns."
            )
        self.embed_dim = int(embed_dim)

        if grid_type not in ("square", "hex"):
            raise ValueError(
                f"grid_type must be 'square' or 'hex', got {grid_type!r}."
            )
        self.grid_type = grid_type

        # Shared multi-scale position embedding: ONE table set reused for the
        # driver location, the order origin, and the order destination, so all
        # positions live in the same learned spatial space (a location means the
        # same thing whether it is a vehicle or an endpoint). Square or hex cells
        # per ``grid_type``.
        if grid_type == "hex":
            self.pos_embed = MultiScaleHexEmbedding(
                resolutions=hex_resolutions,
                embed_dim=pos_embed_dim,
                aggregate=aggregate,
            )
        else:
            self.pos_embed = MultiScaleGridEmbedding(
                resolutions=resolutions,
                embed_dim=pos_embed_dim,
                aggregate=aggregate,
            )
        pos_dim = self.pos_embed.out_dim

        # Non-coordinate feature widths on each side.
        self.driver_rest_dim = self.driver_dim - 2      # drop (x, y)
        self.order_rest_dim = self.order_dim - 4        # drop (ox, oy, dx, dy)
        self.driver_encoder = nn.Linear(self.driver_rest_dim, self.embed_dim)
        self.order_encoder = nn.Linear(self.order_rest_dim, self.embed_dim)

        self.head = _mlp(2 * self.embed_dim + 3 * pos_dim, hidden, 1)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        """Score a batch of pair vectors ``[*, pair_dim]`` -> Q ``[*]``.

        ``pairs`` layout: driver features first (location at cols 0-1), order
        features second (origin at cols 0-1, destination at 2-3 of the order
        block). The dummy (no-order) order carries zero coordinates plus its
        ``dummy_flag`` in the non-coordinate tail, so it maps to the (0, 0) cell
        but is still distinguishable via that flag.
        """
        drv = pairs[..., : self.driver_dim]
        ordr = pairs[..., self.driver_dim :]

        drv_xy = drv[..., 0:2]
        drv_rest = drv[..., 2:]
        ord_origin = ordr[..., 0:2]
        ord_dest = ordr[..., 2:4]
        ord_rest = ordr[..., 4:]

        # --- Positions: encode the driver location, order origin and order
        # destination in a SINGLE pos_embed call. Stacking the three coordinate
        # sets along a new leading axis and passing them together lets the
        # embedding's per-level table lookups (and, for hex, the axial rounding)
        # run once over 3x the rows instead of three separate calls -- far
        # fewer Python-level ops / kernel launches for the same result.
        xy3 = torch.stack([drv_xy, ord_origin, ord_dest], dim=0)  # [3, *, 2]
        pos3 = self.pos_embed(xy3)                                # [3, *, pos_dim]
        drv_pos, ori_pos, dst_pos = pos3[0], pos3[1], pos3[2]

        # --- Non-coordinate features on each side. ---
        drv_emb = self.driver_encoder(drv_rest)                 # [*, embed_dim]
        ord_emb = self.order_encoder(ord_rest)                    # [*, embed_dim]

        # --- Fuse -> scalar Q. ---
        fused = torch.cat([drv_emb, drv_pos, ori_pos, dst_pos, ord_emb], dim=-1)
        return self.head(fused).squeeze(-1)
