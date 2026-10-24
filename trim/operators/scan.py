"""Spatial scan routes and policy metadata for state-space adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

__all__ = [
    "ScanRoute",
    "ScanRouteBank",
    "build_feature_scan_route_bank",
    "build_scan_route_bank",
]

_STATE_SPACE_SCAN_POLICIES = (
    "corner_raster1",
    "corner_raster4",
    "vmamba_cross4",
    "mair_nss4",
    "hilbert_global1",
    "oss6",
    "oss8",
    "objectness_block_snake1",
    "dasm_rowcol_shuffle4",
)
_STATE_SPACE_SCAN_POLICY_CODES = {
    name: index for index, name in enumerate(_STATE_SPACE_SCAN_POLICIES)
}
_STATE_SPACE_CODE_TO_SCAN_POLICY = {
    code: name for name, code in _STATE_SPACE_SCAN_POLICY_CODES.items()
}
_STATE_SPACE_DEFAULT_NSS_STRIPE_WIDTH = 8
_STATE_SPACE_DYNAMIC_SCAN_POLICIES = (
    "objectness_block_snake1",
    "dasm_rowcol_shuffle4",
)
_STATE_SPACE_SCAN_BRANCH_NAMES = {
    "corner_raster1": ("tl_br",),
    "corner_raster4": ("tl_br", "br_tl", "tr_bl", "bl_tr"),
    "vmamba_cross4": ("row_fwd", "row_rev", "col_fwd", "col_rev"),
    "mair_nss4": ("nss_tl_br", "nss_tr_bl", "nss_bl_tr", "nss_br_tl"),
    "hilbert_global1": ("hilbert",),
    "oss6": ("row_fwd", "row_rev", "col_fwd", "col_rev", "sumdiag_fwd", "sumdiag_rev"),
    "oss8": (
        "row_fwd",
        "row_rev",
        "col_fwd",
        "col_rev",
        "sumdiag_fwd",
        "sumdiag_rev",
        "diffdiag_fwd",
        "diffdiag_rev",
    ),
    "objectness_block_snake1": ("obj_block_snake",),
    "dasm_rowcol_shuffle4": (
        "row_shuffle_fwd",
        "row_shuffle_rev",
        "col_shuffle_fwd",
        "col_shuffle_rev",
    ),
}


@dataclass(frozen=True)
class _ScanRouteSpec:
    name: str
    forward_index: tuple[int, ...]
    inverse_index: tuple[int, ...]


@dataclass(frozen=True)
class _ScanRouteBankSpec:
    policy: str
    scan_height: int
    scan_width: int
    crop_height: int
    crop_width: int
    routes: tuple[_ScanRouteSpec, ...]


@dataclass(frozen=True)
class ScanRoute:
    name: str
    forward_index: torch.Tensor
    inverse_index: torch.Tensor


@dataclass(frozen=True)
class ScanRouteBank:
    policy: str
    scan_height: int
    scan_width: int
    crop_height: int
    crop_width: int
    routes: tuple[ScanRoute, ...]

    @property
    def branch_count(self) -> int:
        return len(self.routes)

    @property
    def token_inflation_ratio(self) -> float:
        crop_tokens = self.crop_height * self.crop_width
        if crop_tokens <= 0:
            return 1.0
        return (self.scan_height * self.scan_width) / crop_tokens


def _scan_policy_is_dynamic(policy: str) -> bool:
    return policy in _STATE_SPACE_DYNAMIC_SCAN_POLICIES


def _scan_policy_branch_names(policy: str) -> tuple[str, ...]:
    if policy not in _STATE_SPACE_SCAN_BRANCH_NAMES:
        raise ValueError(f"Unsupported Mamba scan policy: {policy}")
    return _STATE_SPACE_SCAN_BRANCH_NAMES[policy]


def _scan_route_from_coords(
    name: str, coords: list[tuple[int, int]], width: int
) -> _ScanRouteSpec:
    forward_index = tuple(row * width + col for row, col in coords)
    if len(set(forward_index)) != len(forward_index):
        raise ValueError(f"Scan route `{name}` contains duplicate spatial positions.")
    inverse_index = [0] * len(forward_index)
    for sequence_index, flat_index in enumerate(forward_index):
        inverse_index[flat_index] = sequence_index
    return _ScanRouteSpec(
        name=name,
        forward_index=forward_index,
        inverse_index=tuple(inverse_index),
    )


def _reverse_route(route: _ScanRouteSpec, name: str) -> _ScanRouteSpec:
    reversed_forward = tuple(reversed(route.forward_index))
    inverse_index = [0] * len(reversed_forward)
    for sequence_index, flat_index in enumerate(reversed_forward):
        inverse_index[flat_index] = sequence_index
    return _ScanRouteSpec(
        name=name, forward_index=reversed_forward, inverse_index=tuple(inverse_index)
    )


def _flip_coords(
    coords: list[tuple[int, int]],
    *,
    height: int,
    width: int,
    flip_h: bool = False,
    flip_w: bool = False,
) -> list[tuple[int, int]]:
    flipped = coords
    if flip_h:
        flipped = [(height - 1 - row, col) for row, col in flipped]
    if flip_w:
        flipped = [(row, width - 1 - col) for row, col in flipped]
    return flipped


def _row_major_coords(height: int, width: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(height) for col in range(width)]


def _column_major_coords(height: int, width: int) -> list[tuple[int, int]]:
    return [(row, col) for col in range(width) for row in range(height)]


def _serpentine_rows(
    rows: list[int],
    cols: list[int],
    *,
    start_left: bool,
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    forward_cols = list(cols)
    reverse_cols = list(reversed(cols))
    for row_offset, row in enumerate(rows):
        use_forward = (row_offset % 2 == 0) == start_left
        row_cols = forward_cols if use_forward else reverse_cols
        coords.extend((row, col) for col in row_cols)
    return coords


def _sum_diagonal_zigzag_coords(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for diagonal in range(height + width - 1):
        diagonal_coords: list[tuple[int, int]] = []
        row_start = max(0, diagonal - (width - 1))
        row_end = min(height - 1, diagonal)
        for row in range(row_start, row_end + 1):
            diagonal_coords.append((row, diagonal - row))
        if diagonal % 2 == 1:
            diagonal_coords.reverse()
        coords.extend(diagonal_coords)
    return coords


def _diff_diagonal_zigzag_coords(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for diagonal_index, diagonal in enumerate(range(-(width - 1), height)):
        diagonal_coords: list[tuple[int, int]] = []
        row_start = max(0, diagonal)
        row_end = min(height - 1, width - 1 + diagonal)
        for row in range(row_start, row_end + 1):
            diagonal_coords.append((row, row - diagonal))
        if diagonal_index % 2 == 1:
            diagonal_coords.reverse()
        coords.extend(diagonal_coords)
    return coords


def _mair_nested_s_coords(
    height: int,
    width: int,
    *,
    block_index: int,
    stripe_width: int,
) -> list[tuple[int, int]]:
    stripe_width = max(1, min(stripe_width, width))
    shift = stripe_width // 2 if block_index % 2 == 1 else 0
    stripe_starts = list(range(-shift, width, stripe_width))

    coords: list[tuple[int, int]] = []
    for stripe_index, stripe_start in enumerate(stripe_starts):
        col_start = max(0, stripe_start)
        col_end = min(width, stripe_start + stripe_width)
        if col_start >= col_end:
            continue
        cols = list(range(col_start, col_end))
        if stripe_index % 2 == 0:
            rows = list(range(height))
            start_left = True
        else:
            rows = list(range(height - 1, -1, -1))
            start_left = False
        coords.extend(_serpentine_rows(rows, cols, start_left=start_left))
    return coords


def _hilbert_rotate(scale: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = scale - 1 - x
            y = scale - 1 - y
        x, y = y, x
    return x, y


def _hilbert_d2xy(size: int, distance: int) -> tuple[int, int]:
    x = 0
    y = 0
    scale = 1
    t = distance
    while scale < size:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _hilbert_rotate(scale, x, y, rx, ry)
        x += scale * rx
        y += scale * ry
        t //= 4
        scale *= 2
    return y, x


def _hilbert_coords(size: int) -> list[tuple[int, int]]:
    return [_hilbert_d2xy(size, distance) for distance in range(size * size)]


def _default_scan_policy_for_branch_count(branch_count: int) -> str:
    if branch_count <= 1:
        return "corner_raster1"
    return "corner_raster4"


def _build_inverse_index(forward_index: torch.Tensor) -> torch.Tensor:
    if forward_index.dim() == 1:
        inverse_index = torch.empty_like(forward_index)
        inverse_index[forward_index] = torch.arange(
            forward_index.numel(),
            device=forward_index.device,
            dtype=forward_index.dtype,
        )
        return inverse_index
    if forward_index.dim() != 2:
        raise ValueError(
            f"Expected 1D or 2D forward indices, got shape={tuple(forward_index.shape)}"
        )
    inverse_index = torch.empty_like(forward_index)
    sequence_index = (
        torch.arange(
            forward_index.shape[1],
            device=forward_index.device,
            dtype=forward_index.dtype,
        )
        .unsqueeze(0)
        .expand_as(forward_index)
    )
    inverse_index.scatter_(1, forward_index, sequence_index)
    return inverse_index


def _route_from_forward_index(name: str, forward_index: torch.Tensor) -> ScanRoute:
    return ScanRoute(
        name=name,
        forward_index=forward_index,
        inverse_index=_build_inverse_index(forward_index),
    )


def _build_objectness_score_map(x: torch.Tensor) -> torch.Tensor:
    activation_energy = x.abs().mean(dim=1, keepdim=True)
    local_context_3 = F.avg_pool2d(
        activation_energy, kernel_size=3, stride=1, padding=1
    )
    local_context_5 = F.avg_pool2d(
        activation_energy, kernel_size=5, stride=1, padding=2
    )
    return (activation_energy + 0.5 * local_context_3 + 0.25 * local_context_5).squeeze(
        1
    )


def _window_snake_coords(
    height: int, width: int, *, window_size: int
) -> list[list[tuple[int, int]]]:
    coords_by_window: list[list[tuple[int, int]]] = []
    for row_start in range(0, height, window_size):
        for col_start in range(0, width, window_size):
            rows = list(range(row_start, min(height, row_start + window_size)))
            cols = list(range(col_start, min(width, col_start + window_size)))
            coords_by_window.append(_serpentine_rows(rows, cols, start_left=True))
    return coords_by_window


@lru_cache(maxsize=256)
def _build_window_snake_index_table(
    height: int,
    width: int,
    window_size: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[bool, ...], ...]]:
    max_tokens = window_size * window_size
    index_rows: list[tuple[int, ...]] = []
    mask_rows: list[tuple[bool, ...]] = []
    for coords in _window_snake_coords(height, width, window_size=window_size):
        flat_indices = [row * width + col for row, col in coords]
        padding = max_tokens - len(flat_indices)
        index_rows.append(tuple(flat_indices + [0] * padding))
        mask_rows.append(tuple([True] * len(flat_indices) + [False] * padding))
    return tuple(index_rows), tuple(mask_rows)


def _build_objectness_block_snake_route(
    score_map: torch.Tensor,
    *,
    device: torch.device,
) -> ScanRoute:
    batch_size, height, width = score_map.shape
    window_size = max(1, min(4, height, width))
    window_scores = []
    for row_start in range(0, height, window_size):
        for col_start in range(0, width, window_size):
            window_scores.append(
                score_map[
                    :,
                    row_start : min(height, row_start + window_size),
                    col_start : min(width, col_start + window_size),
                ].mean(dim=(1, 2))
            )
    stacked_scores = torch.stack(window_scores, dim=1)
    sorted_windows = torch.argsort(stacked_scores, dim=1, descending=True, stable=True)
    index_rows, mask_rows = _build_window_snake_index_table(height, width, window_size)
    window_indices = torch.tensor(index_rows, dtype=torch.long, device=device)
    valid_mask = torch.tensor(mask_rows, dtype=torch.bool, device=device)
    ordered_indices = window_indices[sorted_windows]
    ordered_mask = valid_mask[sorted_windows]
    forward_index = ordered_indices.masked_select(ordered_mask).reshape(
        batch_size, height * width
    )
    return _route_from_forward_index("obj_block_snake", forward_index)


def _build_dasm_rowcol_shuffle_routes(
    score_map: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[ScanRoute, ...]:
    batch_size, height, width = score_map.shape
    row_order = torch.argsort(
        score_map.mean(dim=2), dim=1, descending=True, stable=True
    )
    col_order = torch.argsort(
        score_map.mean(dim=1), dim=1, descending=True, stable=True
    )
    forward_cols = torch.arange(width, dtype=torch.long, device=device).view(
        1, 1, width
    )
    reverse_cols = torch.arange(
        width - 1, -1, -1, dtype=torch.long, device=device
    ).view(1, 1, width)
    forward_rows = torch.arange(height, dtype=torch.long, device=device).view(
        1, 1, height
    )
    reverse_rows = torch.arange(
        height - 1, -1, -1, dtype=torch.long, device=device
    ).view(1, 1, height)

    row_shuffle_fwd = (row_order.unsqueeze(-1) * width + forward_cols).reshape(
        batch_size, height * width
    )
    row_shuffle_rev = (row_order.unsqueeze(-1) * width + reverse_cols).reshape(
        batch_size, height * width
    )
    col_shuffle_fwd = (forward_rows * width + col_order.unsqueeze(-1)).reshape(
        batch_size, height * width
    )
    col_shuffle_rev = (reverse_rows * width + col_order.unsqueeze(-1)).reshape(
        batch_size, height * width
    )

    return (
        _route_from_forward_index("row_shuffle_fwd", row_shuffle_fwd),
        _route_from_forward_index("row_shuffle_rev", row_shuffle_rev),
        _route_from_forward_index("col_shuffle_fwd", col_shuffle_fwd),
        _route_from_forward_index("col_shuffle_rev", col_shuffle_rev),
    )


@lru_cache(maxsize=256)
def _build_scan_route_bank_spec(
    height: int,
    width: int,
    policy: str,
    block_index: int,
    stripe_width: int,
) -> _ScanRouteBankSpec:
    if policy not in _STATE_SPACE_SCAN_POLICIES:
        raise ValueError(f"Unsupported Mamba scan policy: {policy}")
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Scan bank expects positive spatial shape, got {(height, width)}"
        )

    scan_height = height
    scan_width = width

    if policy in {"corner_raster1", "corner_raster4"}:
        base = _row_major_coords(height, width)
        route_specs = [
            _scan_route_from_coords("tl_br", base, width),
        ]
        if policy == "corner_raster4":
            route_specs.extend(
                [
                    _scan_route_from_coords(
                        "br_tl",
                        _flip_coords(
                            base, height=height, width=width, flip_h=True, flip_w=True
                        ),
                        width,
                    ),
                    _scan_route_from_coords(
                        "tr_bl",
                        _flip_coords(base, height=height, width=width, flip_w=True),
                        width,
                    ),
                    _scan_route_from_coords(
                        "bl_tr",
                        _flip_coords(base, height=height, width=width, flip_h=True),
                        width,
                    ),
                ]
            )
    elif policy == "vmamba_cross4":
        row_forward = _scan_route_from_coords(
            "row_fwd", _row_major_coords(height, width), width
        )
        column_forward = _scan_route_from_coords(
            "col_fwd", _column_major_coords(height, width), width
        )
        route_specs = [
            row_forward,
            _reverse_route(row_forward, "row_rev"),
            column_forward,
            _reverse_route(column_forward, "col_rev"),
        ]
    elif policy == "mair_nss4":
        base = _mair_nested_s_coords(
            height,
            width,
            block_index=block_index,
            stripe_width=stripe_width,
        )
        route_specs = [
            _scan_route_from_coords("nss_tl_br", base, width),
            _scan_route_from_coords(
                "nss_tr_bl",
                _flip_coords(base, height=height, width=width, flip_w=True),
                width,
            ),
            _scan_route_from_coords(
                "nss_bl_tr",
                _flip_coords(base, height=height, width=width, flip_h=True),
                width,
            ),
            _scan_route_from_coords(
                "nss_br_tl",
                _flip_coords(
                    base, height=height, width=width, flip_h=True, flip_w=True
                ),
                width,
            ),
        ]
    elif policy == "hilbert_global1":
        scan_size = 1 << int(math.ceil(math.log2(max(height, width))))
        hilbert_coords = tuple(
            (row, col)
            for row, col in _hilbert_coords(scan_size)
            if row < height and col < width
        )
        route_specs = [
            _scan_route_from_coords("hilbert", hilbert_coords, width),
        ]
    elif policy == "oss6":
        row_forward = _scan_route_from_coords(
            "row_fwd", _row_major_coords(height, width), width
        )
        column_forward = _scan_route_from_coords(
            "col_fwd", _column_major_coords(height, width), width
        )
        diagonal_forward = _scan_route_from_coords(
            "sumdiag_fwd", _sum_diagonal_zigzag_coords(height, width), width
        )
        route_specs = [
            row_forward,
            _reverse_route(row_forward, "row_rev"),
            column_forward,
            _reverse_route(column_forward, "col_rev"),
            diagonal_forward,
            _reverse_route(diagonal_forward, "sumdiag_rev"),
        ]
    elif policy == "oss8":
        row_forward = _scan_route_from_coords(
            "row_fwd", _row_major_coords(height, width), width
        )
        column_forward = _scan_route_from_coords(
            "col_fwd", _column_major_coords(height, width), width
        )
        diagonal_forward = _scan_route_from_coords(
            "sumdiag_fwd", _sum_diagonal_zigzag_coords(height, width), width
        )
        diff_forward = _scan_route_from_coords(
            "diffdiag_fwd", _diff_diagonal_zigzag_coords(height, width), width
        )
        route_specs = [
            row_forward,
            _reverse_route(row_forward, "row_rev"),
            column_forward,
            _reverse_route(column_forward, "col_rev"),
            diagonal_forward,
            _reverse_route(diagonal_forward, "sumdiag_rev"),
            diff_forward,
            _reverse_route(diff_forward, "diffdiag_rev"),
        ]
    else:  # pragma: no cover - guarded by the supported policy check above
        raise ValueError(f"Unsupported Mamba scan policy: {policy}")

    expected_positions = scan_height * scan_width
    for route in route_specs:
        if len(route.forward_index) != expected_positions:
            raise ValueError(
                f"Scan route `{route.name}` in policy `{policy}` only covers "
                f"{len(route.forward_index)} / {expected_positions} positions."
            )

    return _ScanRouteBankSpec(
        policy=policy,
        scan_height=scan_height,
        scan_width=scan_width,
        crop_height=height,
        crop_width=width,
        routes=tuple(route_specs),
    )


def build_scan_route_bank(
    height: int,
    width: int,
    *,
    policy: str = "corner_raster4",
    block_index: int = 0,
    stripe_width: int = _STATE_SPACE_DEFAULT_NSS_STRIPE_WIDTH,
    device: str | torch.device | None = None,
) -> ScanRouteBank:
    if _scan_policy_is_dynamic(policy):
        raise ValueError(
            f"Scan policy `{policy}` is feature-driven. "
            "Use build_feature_scan_route_bank(feature_map, ...) instead of build_scan_route_bank(height, width, ...)."
        )
    spec = _build_scan_route_bank_spec(
        height,
        width,
        policy,
        int(block_index),
        int(stripe_width),
    )
    route_device = torch.device(device) if device is not None else torch.device("cpu")
    routes = tuple(
        ScanRoute(
            name=route.name,
            forward_index=torch.tensor(
                route.forward_index, dtype=torch.long, device=route_device
            ),
            inverse_index=torch.tensor(
                route.inverse_index, dtype=torch.long, device=route_device
            ),
        )
        for route in spec.routes
    )
    return ScanRouteBank(
        policy=spec.policy,
        scan_height=spec.scan_height,
        scan_width=spec.scan_width,
        crop_height=spec.crop_height,
        crop_width=spec.crop_width,
        routes=routes,
    )


def build_feature_scan_route_bank(
    feature_map: torch.Tensor,
    *,
    policy: str,
    block_index: int = 0,
) -> ScanRouteBank:
    if feature_map.dim() != 4:
        raise ValueError(
            f"feature_map must be [B, C, H, W], got shape={tuple(feature_map.shape)}"
        )
    _, _, height, width = feature_map.shape
    device = feature_map.device

    if not _scan_policy_is_dynamic(policy):
        return build_scan_route_bank(
            height,
            width,
            policy=policy,
            block_index=block_index,
            device=device,
        )

    score_map = _build_objectness_score_map(feature_map)
    if policy == "objectness_block_snake1":
        routes = (_build_objectness_block_snake_route(score_map, device=device),)
    elif policy == "dasm_rowcol_shuffle4":
        routes = _build_dasm_rowcol_shuffle_routes(score_map, device=device)
    else:  # pragma: no cover - guarded by _scan_policy_is_dynamic
        raise ValueError(f"Unsupported feature-driven Mamba scan policy: {policy}")

    return ScanRouteBank(
        policy=policy,
        scan_height=height,
        scan_width=width,
        crop_height=height,
        crop_width=width,
        routes=routes,
    )
