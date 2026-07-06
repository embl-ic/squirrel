
import numpy as np


def _halo_faces(direction, ndim, overlap):
    """
    Return matching halo slices for one neighbouring tile.

    Parameters
    ----------
    direction : {"right", "bottom", "behind"}
    ndim : int
    overlap : tuple[int]

    Returns
    -------
    this_face, neighbour_face : tuple[slice]
    """

    axis = {
        "behind": 0,
        "bottom": 1,
        "right": 2,
    }[direction]

    this_face = []
    neighbour_face = []

    for ax in range(ndim):

        o = overlap[ax]

        if ax == axis:
            # positive halo of this tile
            this_face.append(slice(-o, None))

            # negative halo of neighbour
            neighbour_face.append(slice(0, o))

        else:
            this_face.append(slice(None))
            neighbour_face.append(slice(None))

    return tuple(this_face), tuple(neighbour_face)


def _add_boundary_edges(
    edge_dict,
    seg,
    neighbour,
    direction,
    overlap,
    background,
):
    """
    Compute overlap-derived disaffinities for one neighbouring tile.
    """

    import bioimage_cpp as bic
    import numpy as np

    this_face, neighbour_face = _halo_faces(
        direction,
        seg.ndim,
        overlap,
    )

    table = bic.utils.segmentation_overlap(
        np.ascontiguousarray(seg[this_face], dtype="uint64"),
        np.ascontiguousarray(neighbour[neighbour_face], dtype="uint64"),
    ).overlap_table()

    if table.shape[0] == 0:
        return

    # Compute normalized overlaps exactly as in bioimage-py

    label_a = table["label_a"]
    label_b = table["label_b"]
    counts = table["count"].astype(np.float64)

    # Total overlap size of each label_a
    unique_a, inverse = np.unique(label_a, return_inverse=True)

    size_a = np.zeros(len(unique_a), dtype=np.float64)
    np.add.at(size_a, inverse, counts)

    overlap_fraction = counts / size_a[inverse]

    # Convert to disaffinities

    for la, lb, frac in zip(label_a, label_b, overlap_fraction):

        if la == background or lb == background:
            continue

        edge = (min(int(la), int(lb)), max(int(la), int(lb)))

        # If multiple overlap entries exist for the same edge,
        # keep the strongest overlap (smallest disaffinity),
        # matching the original implementation.
        disaffinity = 1.0 - frac

        if edge in edge_dict:
            edge_dict[edge] = min(edge_dict[edge], disaffinity)
        else:
            edge_dict[edge] = disaffinity


def build_local_graph(
    seg,
    overlap,
    right=None,
    bottom=None,
    behind=None,
    background=0,
    default_disaffinity=0.9,
    n_workers=1
):
    """
    Build the graph contribution of one tile.

    Parameters
    ----------
    seg : ndarray
        Tile segmentation including overlap.

    overlap : tuple[int]
        Overlap width in pixels (z, y, x).

    right, bottom, behind : ndarray | None
        Forward neighbouring tiles.

    Returns
    -------
    edges : (N,2) ndarray

    disaffinities : (N,) ndarray
    """

    import bioimage_cpp as bic

    # Graph edges inside one tile get the default disaffinity

    rag = bic.graph.region_adjacency_graph(seg, number_of_threads=n_workers)

    edge_dict = {}

    for u, v in rag.uv_ids():

        if u == background or v == background:
            continue
        edge_dict[(min(u, v), max(u, v))] = default_disaffinity

    # Boundary edges get disaffinites based on overlap with objects in the neighbouring tile

    for direction, neighbour in (("right", right), ("bottom", bottom), ("behind", behind)):

        if neighbour is None:
            continue

        _add_boundary_edges(
            edge_dict=edge_dict,
            seg=seg,
            neighbour=neighbour,
            direction=direction,
            overlap=overlap,
            background=background,
        )

    edges = np.asarray(
        list(edge_dict.keys()),
        dtype=np.uint64,
    )

    disaffinities = np.asarray(
        list(edge_dict.values()),
        dtype=np.float32,
    )

    return edges, disaffinities


def _compute_edge_costs(disaffinities, beta=0.5, eps=1e-6):
    """
    Convert disaffinities in [0,1] into multicut edge costs.

    Parameters
    ----------
    disaffinities : ndarray
    beta : float

    Returns
    -------
    costs : ndarray
    """

    p = np.clip(disaffinities, eps, 1.0 - eps)

    costs = np.log((1.0 - p) / p)
    costs += np.log((1.0 - beta) / beta)

    return costs.astype(np.float32)


def solve_global_multicut(
    edge_list,
    disaffinity_list,
    beta=0.5,
):
    """
    Solve the global multicut problem.

    Parameters
    ----------
    edge_list : list[np.ndarray]
        List of edge arrays returned by ``build_local_graph``.
        Each array has shape (N, 2).

    disaffinity_list : list[np.ndarray]
        Corresponding disaffinity arrays.

    beta : float
        Prior parameter for the multicut cost transform.

    Returns
    -------
    node_labels : np.ndarray
        Multicut component label for each graph node.

    graph_labels : np.ndarray
        Mapping from graph node index back to the original segmentation label.
    """

    import nifty.graph as ng
    import nifty.graph.opt.multicut as nmc

    # Merge all local graphs
    edges = np.concatenate(edge_list, axis=0)
    disaffinities = np.concatenate(disaffinity_list)

    # Remove duplicate edges.
    # Keep the strongest attraction (= smallest disaffinity).
    edge_dict = {}
    for edge, dis in zip(edges, disaffinities):

        edge = (min(int(edge[0]), int(edge[1])),
                max(int(edge[0]), int(edge[1])))

        if edge in edge_dict:
            edge_dict[edge] = min(edge_dict[edge], dis)
        else:
            edge_dict[edge] = dis

    # Convert back to arrays
    edges = np.asarray(list(edge_dict.keys()), dtype=np.uint64)
    disaffinities = np.asarray(list(edge_dict.values()), dtype=np.float32)

    # Relabel graph nodes to a dense index set
    graph_labels = np.unique(edges.ravel())
    dense_edges = np.searchsorted(graph_labels, edges)

    # Build graph
    graph = ng.UndirectedGraph(len(graph_labels))
    graph.insertEdges(dense_edges)

    # Compute multicut costs
    costs = _compute_edge_costs(
        disaffinities,
        beta=beta,
    )

    # Solve
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.kernighanLinFactory().create(objective)
    node_labels = solver.optimize()

    # Compress component ids to consecutive labels
    unique_components = np.unique(node_labels)
    component_to_label = {
        component: i + 1
        for i, component in enumerate(unique_components)
    }

    # Final mapping:
    # segmentation label -> stitched label
    label_mapping = {
        int(label): component_to_label[component]
        for label, component in zip(graph_labels, node_labels)
    }

    return label_mapping


def relabel_segmentation(
    seg,
    label_mapping,
    background=0,
):
    """
    Apply the multicut label mapping to a segmentation.

    Parameters
    ----------
    seg : np.ndarray

    label_mapping : dict[int, int]
        Output of solve_global_multicut().

    background : int

    Returns
    -------
    relabeled : np.ndarray
    """

    labels, inverse = np.unique(seg, return_inverse=True)

    new_labels = labels.copy()

    for i, label in enumerate(labels):

        if label == background:
            continue

        new_labels[i] = label_mapping.get(int(label), int(label))

    return new_labels[inverse].reshape(seg.shape)


if __name__ == '__main__':

    import zarr

    with zarr.open("/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000005_labels.ome.zarr/", mode="r") as f:
        tile1 = f['scale0/nuclei_labels'][:].astype('uint32')
    with zarr.open("/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000006_labels.ome.zarr/", mode="r") as f:
        tile2 = f['scale0/nuclei_labels'][:].astype('uint32') + 2 ** 16

    # One job per tile to build the local graph and compute disaffinities

    edges1, disaffinities1 = build_local_graph(
        seg=tile1,
        overlap=(1, 1, 1),
        right=tile2,
        bottom=None,
        behind=None,
        background=0,
        default_disaffinity=0.9,
    )

    edges2, disaffinities2 = build_local_graph(
        seg=tile2,
        overlap=(1, 1, 1),
        right=None,
        bottom=None,
        behind=None,
        background=0,
        default_disaffinity=0.9,
    )

    # One global multicut job for all tiles

    label_mapping = solve_global_multicut(
        edge_list=[edges1, edges2],
        disaffinity_list=[disaffinities1, disaffinities2],
        beta=0.5
    )

    # One job per tile to relabel and write the stitched segmentation

    tile1_stitched = relabel_segmentation(
        seg=tile1,
        label_mapping=label_mapping,
        background=0,
    )

    tile2_stitched = relabel_segmentation(
        seg=tile2,
        label_mapping=label_mapping,
        background=0,
    )


    def _write_stitched_tile(input_store, output_store, segmentation):
        import shutil
        shutil.copytree(input_store, output_store, dirs_exist_ok=True)

        with zarr.open(output_store, mode="r+") as f:
            f["scale0/nuclei_labels"][:] = segmentation.astype(
                f["scale0/nuclei_labels"].dtype
            )

    _write_stitched_tile(
        input_store="/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000005_labels.ome.zarr/",
        output_store="/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000005_labels_stitched.ome.zarr/",
        segmentation=tile1_stitched,
    )
    _write_stitched_tile(
        input_store="/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000006_labels.ome.zarr/",
        output_store="/media/julian/Data/projects/hennies/multicut-stitching-devel/cellpose/platy__tile_000006_labels_stitched.ome.zarr/",
        segmentation=tile2_stitched,
    )
    
