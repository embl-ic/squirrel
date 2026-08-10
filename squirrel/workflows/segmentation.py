
import numpy as np


def positions_to_forward_neighbors(positions):
    """
    Convert grid positions to a forward-neighbor list.

    Parameters
    ----------
    positions : list[list[int]]
        List of N-dimensional integer coordinates.

    Returns
    -------
    list[list[Optional[int]]]
        neighbors[i][d] is the index of the tile one step forward
        along dimension d, or None if no such neighbor exists.
    """
    if not positions:
        return []

    ndim = len(positions[0])

    # Map position -> index
    pos_to_idx = {tuple(pos): i for i, pos in enumerate(positions)}

    neighbors = []

    for pos in positions:
        row = []
        for d in range(ndim):
            neighbor = list(pos)
            neighbor[d] += 1
            row.append(pos_to_idx.get(tuple(neighbor)))
        neighbors.append(row)

    return neighbors


def _make_empty_grid(positions, fill=None):
    """
    Create the smallest N-dimensional grid that can contain all positions.

    Returns
    -------
    grid : nested list
        N-dimensional nested list filled with `fill`.
    origin : tuple
        Minimum coordinate in each dimension. If positions do not start at
        zero, subtract this from coordinates to index into the grid.
    """
    if not positions:
        return fill.copy(), ()

    ndim = len(positions[0])

    mins = [min(p[d] for p in positions) for d in range(ndim)]
    maxs = [max(p[d] for p in positions) for d in range(ndim)]
    shape = [maxs[d] - mins[d] + 1 for d in range(ndim)]

    def build(shape):
        if len(shape) == 1:
            return [fill.copy() for _ in range(shape[0])]
        return [build(shape[1:]) for _ in range(shape[0])]

    return build(shape), tuple(mins)


def _concatenate_grid(grid, axis=0):
    if isinstance(grid, np.ndarray):
        return grid

    return np.concatenate(
        [_concatenate_grid(sub, axis + 1) for sub in grid],
        axis=axis
    )


def stitch_segmentation_workflow(
        tile_paths,
        output_path,
        tile_keys=None,
        make_ids_in_tiles_unique=False,
        tile_positions=None,
        verbose=False
):
    """
    Workflow for stitching segmentation results.

    tile_positions: list of tuples
        [(z, y, x), ...] positions of the tiles in the global coordinate system. 
    """
    
    from squirrel.library.stitching import build_local_graph, solve_global_multicut, relabel_segmentation
    from squirrel.library.io import load_data_handle, write_stack

    if len(tile_paths) != len(tile_positions):
        raise ValueError("Number of tile paths must match number of tile positions.")

    # Load datasets
    data = []
    for idx, (tile_path, tile_key) in enumerate(zip(tile_paths, tile_keys)):
        print(f"Loading tile {idx} from {tile_path}...")
        h, _ = load_data_handle(tile_path, tile_key)
        data.append(h[:].astype("uint64"))
        if make_ids_in_tiles_unique:
            data[-1] += (idx + 1) * 2 ** 16 * idx  # Offset IDs to make them unique

    # Compute the label mapping 
    edges = []
    disaffinities = []
    forward_neighbors = positions_to_forward_neighbors(tile_positions)
    for i, (h, neighbors) in enumerate(zip(data, forward_neighbors)):
        if verbose:
            print(f"Building local graph for tile {i}...")
        this_edges, this_disaffinities = build_local_graph(
            seg=h,
            overlap=(1, 1, 1),
            right=data[neighbors[2]] if neighbors[2] is not None else None,
            bottom=data[neighbors[1]] if neighbors[1] is not None else None,
            behind=data[neighbors[0]] if neighbors[0] is not None else None,
            background=0,
            default_disaffinity=0.9,
        )
        edges.append(this_edges)
        disaffinities.append(this_disaffinities)

    label_mapping = solve_global_multicut(edges, disaffinities, beta=0.5)

    # Relabel the segmentations and save to disk
    stitched_seg = _make_empty_grid(tile_positions, fill=np.zeros(data[0].shape, dtype=data[0].dtype))[0]
    for idx, seg in enumerate(data): 
        if verbose:
            print(f"Relabelling tile {idx}...")
            # print(np.unique(seg))
        stitched_seg[
            tile_positions[idx][0]][
            tile_positions[idx][1]][
            tile_positions[idx][2]][
            :seg.shape[0], :seg.shape[1], :seg.shape[2]
        ] = relabel_segmentation(seg, label_mapping, background=0)
    stitched_seg = _concatenate_grid(stitched_seg)

    write_stack(output_path, stitched_seg, key='s0')


if __name__ == "__main__":

    # Example usage
    tile_paths = [
        "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000001_labels.ome.zarr",
        "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000002_labels.ome.zarr",
        "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000003_labels.ome.zarr",
        "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000004_labels.ome.zarr",
        "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000005_labels.ome.zarr",
        "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000006_labels.ome.zarr",
        "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000007_labels.ome.zarr",
        "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000008_labels.ome.zarr",
        # "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000010_labels.ome.zarr",
        # "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000019_labels.ome.zarr",
        # "/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000040_labels.ome.zarr",
    ]
    output_path = "/media/julian/Data/projects/hennies/multicut-stitching-devel/stitched.ome.zarr"
    tile_keys = ["scale0/nuclei_labels"] * len(tile_paths) 
    tile_positions = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
        # (0, 2, 1),
        # (1, 0, 2),
        # (3, 1, 3)
    ] 

    stitch_segmentation_workflow(
        tile_paths,
        output_path,
        tile_keys=tile_keys,
        make_ids_in_tiles_unique=True,
        tile_positions=tile_positions,
        verbose=True
    )
