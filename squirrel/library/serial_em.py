from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from copy import deepcopy
import re
import numpy as np


# ============================================================================
# Standalone utility functions
# ============================================================================

def get_unique_key(
    base_key: str,
    items: Mapping[str, Any],
) -> str:
    """
    Return a key that does not yet exist in items.

    The base key is returned unchanged when available. Otherwise,
    integer suffixes are appended until a unique key is found.
    """
    if base_key not in items:
        return base_key
    
    idx = 1
    while f"{base_key}-{idx}" in items:
        idx += 1

    return f"{base_key}-{idx}"


def navigator_file_to_dict(filepath: str | Path) -> dict[str, Any]:
    """Parse a SerialEM Navigator file into a dictionary."""
    data = {}
    items = {}
    current_item = None

    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            match = re.match(r"(\w+)\s*=\s*(.+)", line)

            if match:
                key, value = match.groups()

                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

                if current_item is None:
                    data[key] = value
                else:
                    items[current_item][key] = value

                continue

            match = re.match(r"\[Item\s*=\s*(.*?)\]", line)

            if match:
                base_item = match.group(1)
                current_item = get_unique_key(base_item, items)
                items[current_item] = {}

    data["items"] = items

    return data


def extend_navigator_dict(filepath: str | Path, nav_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Extend a parsed Navigator dictionary with another Navigator file."""

    def get_base_key(key: str) -> str:
        return re.sub(r"^(.*)-(0|[1-9]\d*)$", r"\1", key)

    def get_items_with_base_key(items: Mapping[str, Any], base_key: str):
        for key, item in items.items():
            if get_base_key(key) == base_key:
                yield key, item

    out_dict = deepcopy(nav_dict)
    extend_dict = navigator_file_to_dict(filepath)

    if extend_dict["AdocVersion"] != nav_dict["AdocVersion"]:
        raise ValueError(f'AdocVersions do not match: {nav_dict["AdocVersion"]} != {extend_dict["AdocVersion"]}')

    all_base_keys = {get_base_key(key) for key in nav_dict["items"]}

    for key, nav_item in extend_dict["items"].items():

        clean_key = get_base_key(key)

        if clean_key not in all_base_keys:
            out_dict["items"][clean_key] = nav_item
            all_base_keys.add(clean_key)
            continue

        existing_items = list(get_items_with_base_key(out_dict["items"], clean_key))

        if "MapID" in nav_item:
            matching_item = next((item for _, item in existing_items if item.get("MapID") == nav_item["MapID"]), None)
        else:
            matching_item = next((item for _, item in existing_items if item == nav_item), None)

        if matching_item is None:
            unique_key = get_unique_key(clean_key, out_dict["items"])
            out_dict["items"][unique_key] = nav_item
            continue

        if "MapFile" in nav_item:
            if matching_item.get("MapFile") != nav_item["MapFile"]:
                raise ValueError(f'Items with MapID {nav_item.get("MapID")} have different MapFile values.')

    return out_dict


def get_value_from_item(item: Mapping[str, Any], key: str) -> float | None:
    """Return a SerialEM item value as a float."""
    if key not in item:
        return None
    return float(item[key])


def get_value_list_from_item(item: Mapping[str, Any], key: str) -> list[float] | None:
    """Return a whitespace-separated SerialEM item value as floats."""
    if key not in item:
        return None
    return [float(value) for value in str(item[key]).split()]


def match_regex(value: Any, regex: str) -> str | None:
    """
    Match a value against a regular expression.

    If the expression contains capture groups, their contents are
    concatenated. Otherwise, the complete match is returned.
    """
    match = re.search(regex, str(value))

    if match is None:
        return None

    return "".join(match.groups()) or match.group(0)


def get_filepath_from_nav_item(nav_filepath: str | Path, map_item: Mapping[str, Any], item_id: str) -> Path:
    """
    Resolve a filepath stored in a SerialEM Navigator item.

    SerialEM may store paths using Windows-style separators. Only the
    filename is used, and the file is assumed to be located relative to
    the Navigator file.
    """
    nav_filepath = Path(nav_filepath)
    item_filepath = Path(str(map_item[item_id]).replace("\\", "/"))
    return nav_filepath.parent / item_filepath.name


def get_map_filepath_from_nav_item(nav_filepath: str | Path, map_item: Mapping[str, Any]) -> Path:
    """Resolve the MapFile filepath from a SerialEM Navigator item."""
    return get_filepath_from_nav_item(nav_filepath, map_item, "MapFile")


def get_map_items_by_glob(nav_dict: Mapping[str, Any], nav_filepath: str | Path, glob: str = "*.mrc") -> dict[str, dict]:
    """
    Return Navigator items whose resolved MapFile matches a glob pattern.
    """
    out_items = {}

    for key, item in nav_dict["items"].items():

        if "MapFile" not in item:
            continue

        map_filepath = get_map_filepath_from_nav_item(nav_filepath, item)

        if map_filepath.match(glob):
            out_items[key] = item

    return out_items


def get_resolution_from_mrc_header(filepath: str | Path, unit: str = 'micrometer') -> np.ndarray:
    """
    Read XYZ voxel spacing from an MRC header.

    Returns
    -------
    np.ndarray
        Resolution in [x, y, z] order.
    """
    import mrcfile

    unit_factors = {
        'angstrom': 1.0,
        'nanometer': 1e-1,
        'micrometer': 1e-4,
    }

    if unit not in unit_factors:
        raise ValueError(f'Unsupported unit: {unit}. Supported units are {list(unit_factors)}')

    with mrcfile.open(filepath, permissive=True, header_only=True) as mrc:

        resolution = np.array([
            mrc.voxel_size.x,
            mrc.voxel_size.y,
            mrc.voxel_size.z,
        ], dtype=float)

    return resolution * unit_factors[unit]


def get_map_scale_xy(map_item: Mapping[str, Any]) -> list[float] | None:
    """Return the XY components of the SerialEM StageXYZ entry."""
    stage_xyz = get_value_list_from_item(map_item, "StageXYZ")

    if stage_xyz is None:
        return None

    return stage_xyz[:2]


def get_map_scale_matrix_from_item(item: Mapping[str, Any]) -> np.ndarray | None:
    """Return SerialEM MapScaleMat as a 2x2 matrix."""
    values = get_value_list_from_item(item, "MapScaleMat")

    if values is None:
        return None

    if len(values) != 4:
        raise ValueError(f"MapScaleMat must contain exactly four values. Found {len(values)}: {values}")

    return np.asarray(values, dtype=float).reshape(2, 2)


def get_map_shape_from_serialem_item(map_item: Mapping[str, Any], binning: int) -> np.ndarray:
    """
    Compute the map XY shape from SerialEM metadata.

    Returns
    -------
    np.ndarray
        Shape in [x, y] / [width, height] order.
    """
    map_binning = get_value_from_item(map_item, "MapBinning")
    mont_binning = get_value_from_item(map_item, "MontBinning")
    map_width_height = get_value_list_from_item(map_item, "MapWidthHeight")
    bin_factor = (map_binning / (mont_binning * binning))

    return (np.asarray(map_width_height) * bin_factor).astype(int)


def get_mrc_shape(filepath: str | Path) -> tuple[int, int, int]:
    """Read the array shape from an MRC file."""
    import mrcfile
    with mrcfile.open(filepath, permissive=True, header_only=True) as mrc:
        shape = (int(mrc.header.nz), int(mrc.header.ny), int(mrc.header.nx))
    return shape


def get_contrast_limits_from_map(filepath: str | Path) -> tuple[float, float]:
    """Compute display contrast limits for an MRC map."""
    import mrcfile
    from squirrel.library.data import get_contrast_limits

    with mrcfile.open(filepath, permissive=True) as mrc:
        image = mrc.data
        contrast_limits = get_contrast_limits(image)

    return tuple(float(value) for value in contrast_limits)


# ============================================================================
# Core data objects
# ============================================================================

@dataclass
class NavigatorMap:
    """
    One logical map known to the Navigator.

    Spatial conventions
    -------------------
    shape:
        Array axis order, i.e. (z, y, x).

    resolution:
        Spatial coordinate order, i.e. [x, y, z].
    """

    id: str
    map_type: str
    serialem_item: dict[str, Any]

    source_filepath: Path
    filepath: Path

    binning: int = 1

    resolution: np.ndarray | None = None
    shape: tuple[int, ...] | None = None
    section_id: int | None = None
    contrast_limits: tuple[float, float] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_filepath = Path(self.source_filepath)
        self.filepath = Path(self.filepath)

    @property
    def note(self) -> str | None:
        return self.get_serialem_value("Note", default=None)

    @property
    def stage_xy(self) -> np.ndarray | None:
        stage_xy = get_map_scale_xy(self.serialem_item)

        if stage_xy is None:
            return None

        return np.asarray(stage_xy, dtype=float)
    
    @property
    def map_scale_matrix(self) -> np.ndarray | None:
        return get_map_scale_matrix_from_item(self.serialem_item)
    
    def has_serialem_key(self, key: str) -> bool:
        return key in self.serialem_item

    def get_serialem_value(self, key: str, default: Any = None) -> Any:
        return self.serialem_item.get(key, default)

    def load_resolution(self, force: bool = False) -> np.ndarray:
        if self.resolution is None or force:
            self.resolution = get_resolution_from_mrc_header(self.filepath, unit="micrometer")
        return self.resolution

    def load_shape(self, force: bool = False) -> tuple[int, ...]:
        if self.shape is None or force:
            self.shape = get_mrc_shape(self.filepath)
        return self.shape

    def load_contrast_limits(self, force: bool = False) -> tuple[float, float]:
        if self.contrast_limits is None or force:
            self.contrast_limits = get_contrast_limits_from_map(self.filepath)
        return self.contrast_limits

    def refresh_metadata(self, include_contrast_limits: bool = False) -> None:
        self.load_resolution(force=True)
        self.load_shape(force=True)

        if include_contrast_limits:
            self.load_contrast_limits(force=True)


@dataclass
class MatchRule:
    """
    Description of how a map is matched to another map.

    A rule can either match directly on one SerialEM item value, or
    perform a second lookup through nav_dict.

    Example
    -------
    Direct match:
        MatchRule(
            item_key="MapFile",
            regex=r"L(\d{2})",
        )

    Two-step match:
        MatchRule(
            item_key="SomeReference",
            regex=r"(\d+)",
            secondary_item_key="MapFile",
            secondary_regex=r"L(\d{2})",
        )
    """

    item_key: str
    regex: str

    secondary_item_key: str | None = None
    secondary_regex: str | None = None

    @classmethod
    def from_config(cls, config: Sequence[str] | "MatchRule") -> "MatchRule":
        if isinstance(config, cls):
            return config
        if len(config) not in (2, 4):
            raise ValueError("MatchRule config must contain either 2 or 4 entries.")
        return cls(*config)

    def match(self, item: Mapping[str, Any], nav_dict: Mapping[str, Any] | None = None) -> str | None:

        if self.item_key not in item:
            return None

        match_id = match_regex(item[self.item_key], self.regex)
        if match_id is None:
            return None

        has_secondary_rule = (self.secondary_item_key is not None or self.secondary_regex is not None)

        if not has_secondary_rule:
            return match_id

        if (self.secondary_item_key is None or self.secondary_regex is None):
            raise ValueError("Both secondary_item_key and secondary_regex must be provided for a secondary match.")
        if nav_dict is None:
            raise ValueError("nav_dict is required for a secondary match.")

        try:
            referenced_item = nav_dict["items"][match_id]
        except KeyError:
            return None

        if self.secondary_item_key not in referenced_item:
            return None

        return match_regex(referenced_item[self.secondary_item_key], self.secondary_regex)
    

# ============================================================================
# Map collection
# ============================================================================

class MapCollection:
    """Container and query interface for all NavigatorMap instances."""

    def __init__(self):
        self._maps: dict[str, dict[str, NavigatorMap]] = {}

    def add(self, map_: NavigatorMap) -> None:
        self._maps.setdefault(map_.map_type, {})

        if map_.id in self._maps[map_.map_type]:
            raise ValueError(f'Map already exists: map_type="{map_.map_type}", id="{map_.id}"')

        self._maps[map_.map_type][map_.id] = map_

    def add_many(self, maps: Iterable[NavigatorMap]) -> None:
        for map_ in maps:
            self.add(map_)

    def remove(
        self,
        map_type: str,
        map_id: str,
    ) -> NavigatorMap:
        
        if map_type not in self._maps:
            raise KeyError(f'Unknown map type: "{map_type}"')
        if map_id not in self._maps[map_type]:
            raise KeyError(f'Unknown map: map_type="{map_type}", id="{map_id}"')

        map_ = self._maps[map_type].pop(map_id)

        if not self._maps[map_type]:
            del self._maps[map_type]

        return map_

    def get(self, map_type: str, map_id: str) -> NavigatorMap:
        if map_type not in self._maps:
            raise KeyError(f'Unknown map type: "{map_type}"')
        if map_id not in self._maps[map_type]:
            raise KeyError(f'Unknown map: map_type="{map_type}", id="{map_id}"')

        return self._maps[map_type][map_id]

    def get_optional(self, map_type: str, map_id: str) -> NavigatorMap | None:
        return self._maps.get(map_type, {}).get(map_id)

    def by_type(self, map_type: str) -> dict[str, NavigatorMap]:
        return dict(self._maps.get(map_type, {}))

    def ids(self, map_type: str) -> list[str]:
        return list(self._maps.get(map_type, {}).keys())

    def map_types(self) -> list[str]:
        return list(self._maps.keys())

    def all(self) -> Iterator[NavigatorMap]:
        for maps_of_type in self._maps.values():
            yield from maps_of_type.values()

    def find(self, map_type: str, serialem_item: str, regex: str, target_value: str) -> NavigatorMap | None:
        for map_ in self._maps.get(map_type, {}).values():

            if not map_.has_serialem_key(serialem_item):
                continue

            value = map_.get_serialem_value(serialem_item)
            candidate = match_regex(value, regex)

            if candidate == target_value:
                return map_

        return None

    def find_by_filepath(self, filepath: str | Path) -> NavigatorMap | None:
        filepath = Path(filepath)

        for map_ in self.all():
            if map_.filepath == filepath:
                return map_

        return None

    def __contains__(self, key: tuple[str, str]) -> bool:
        map_type, map_id = key
        return (map_type in self._maps and map_id in self._maps[map_type])

    def __iter__(self) -> Iterator[NavigatorMap]:
        yield from self.all()

    def __len__(self) -> int:
        return sum(len(maps_of_type) for maps_of_type in self._maps.values())
    

# ============================================================================
# Hierarchy / relationships
# ============================================================================

class MapHierarchy:
    """
    Stores parent-child relationships between maps.

    Relationships are stored in both directions:

        child -> parent
        parent -> ordered children
    """

    def __init__(
        self,
        map_type_order: Sequence[str] | None = None,
    ):
        self._map_type_order = list(map_type_order or [])
        self._nodes: dict[tuple[str, str], None] = {}
        self._parent: dict[tuple[str, str], tuple[str, str]] = {}
        self._children: dict[tuple[str, str], list[tuple[str, str]]] = {}

    def add_node(self, map_type: str, map_id: str) -> None:
        if map_type not in self._map_type_order:
            raise ValueError(f'Unknown map type: "{map_type}"')
        self._nodes.setdefault((map_type, map_id), None)

    @property
    def map_type_order(self) -> list[str]:
        return self._map_type_order.copy()

    def add_map_type(self, map_type: str) -> None:
        if map_type in self._map_type_order:
            raise ValueError(f'Map type already exists: "{map_type}"')
        self._map_type_order.append(map_type)

    def add_relation(self, parent_type: str, parent_id: str, child_type: str, child_id: str) -> None:

        if parent_type not in self._map_type_order:
            raise ValueError(f'Unknown parent map type: "{parent_type}"')
        if child_type not in self._map_type_order:
            raise ValueError(f'Unknown child map type: "{child_type}"')

        expected_child_type = self.child_type(parent_type)
        if expected_child_type != child_type:
            raise ValueError(
                f'Invalid relationship: "{parent_type}" -> "{child_type}". Expected child type: "{expected_child_type}"'
            )

        parent = (parent_type, parent_id)
        child = (child_type, child_id)

        # Register both maps as hierarchy nodes.
        self._nodes.setdefault(parent, None)
        self._nodes.setdefault(child, None)

        if child in self._parent:
            existing_parent = self._parent[child]
            if existing_parent == parent:
                return
            raise ValueError(f'Child already has a parent: {child} -> {existing_parent}')

        self._parent[child] = parent
        self._children.setdefault(parent, []).append(child)

    def remove_relation(self, child_type: str, child_id: str) -> None:

        child = (child_type, child_id)
        if child not in self._parent:
            raise KeyError(f'No parent relation found for: {child}')

        parent = self._parent.pop(child)
        children = self._children[parent]
        children.remove(child)

        if not children:
            del self._children[parent]

    def parent_type(self, map_type: str) -> str | None:
        if map_type not in self._map_type_order:
            raise ValueError(f'Unknown map type: "{map_type}"')

        idx = self._map_type_order.index(map_type)
        if idx == 0:
            return None

        return self._map_type_order[idx - 1]

    def child_type(self, map_type: str) -> str | None:
        if map_type not in self._map_type_order:
            raise ValueError(f'Unknown map type: "{map_type}"')

        idx = self._map_type_order.index(map_type)
        if idx == len(self._map_type_order) - 1:
            return None

        return self._map_type_order[idx + 1]

    def parent_id(self, map_type: str, map_id: str) -> str | None:
        parent = self._parent.get((map_type, map_id))

        if parent is None:
            return None
        return parent[1]

    def children_ids(self, map_type: str, map_id: str) -> list[str]:
        children = self._children.get((map_type, map_id), [])
        return [child_id for _, child_id in children]

    def siblings_ids(self, map_type: str, map_id: str) -> list[str]:

        child = (map_type, map_id)
        parent = self._parent.get(child)

        if parent is None:
            return []

        return [sibling_id for sibling_type, sibling_id in self._children[parent] if sibling_type == map_type and sibling_id != map_id]

    def ancestors(self, map_type: str, map_id: str) -> list[tuple[str, str]]:
        """
        Return all ancestors of a map.

        The immediate parent is returned first, followed by progressively
        higher ancestors.
        """
        ancestors = []
        current = (map_type, map_id)

        while current in self._parent:
            current = self._parent[current]
            ancestors.append(current)

        return ancestors

    def descendants(self, map_type: str, map_id: str) -> list[tuple[str, str]]:
        """
        Return all descendants of a map in depth-first order.
        """
        descendants = []

        def recurse(parent: tuple[str, str]) -> None:
            for child in self._children.get(parent, []):
                descendants.append(child)
                recurse(child)

        recurse((map_type, map_id))
        return descendants

    def roots(self) -> list[tuple[str, str]]:
        if not self._map_type_order:
            return []
        root_type = self._map_type_order[0]
        return [node for node in self._nodes if node[0] == root_type and node not in self._parent]

    def iter_depth_first(self, root_type: str | None = None, root_id: str | None = None) -> Iterator[tuple[str, str, list[int]]]:
        """
        Iterate maps in deterministic depth-first pre-order.
        """

        if (root_type is None) != (root_id is None):
            raise ValueError("root_type and root_id must either both be supplied or both be None.")

        def recurse(map_type: str, map_id: str, idx_path: list[int]) -> Iterator[tuple[str, str, list[int]]]:

            yield (map_type, map_id, idx_path)

            children = self._children.get((map_type, map_id), [])
            for idx, (child_type, child_id) in enumerate(children):
                yield from recurse(child_type, child_id, idx_path + [idx])

        if root_type is not None:

            root = (root_type, root_id)
            if root not in self._nodes:
                raise KeyError(f"Unknown hierarchy node: {root}")

            yield from recurse(root_type, root_id, [0])
            return

        for root_idx, (root_map_type, root_map_id) in enumerate(
            self.roots()
        ):
            yield from recurse(root_map_type, root_map_id, [root_idx])

    def validate(self) -> None:
        """
        Validate the internal consistency of the hierarchy.

        Raises
        ------
        ValueError
            If an invalid or inconsistent relationship is found.
        """
        if len(self._map_type_order) != len(set(self._map_type_order)):
            raise ValueError("map_type_order contains duplicate map types.")

        for child, parent in self._parent.items():

            child_type, child_id = child
            parent_type, parent_id = parent

            if child_type not in self._map_type_order:
                raise ValueError(f'Unknown child map type: "{child_type}"')
            if parent_type not in self._map_type_order:
                raise ValueError(f'Unknown parent map type: "{parent_type}"')

            expected_parent_type = self.parent_type(child_type)
            if parent_type != expected_parent_type:
                raise ValueError(f'Invalid relationship: "{parent_type}" -> "{child_type}". Expected parent type: "{expected_parent_type}"')

            children = self._children.get(parent, [])
            if child not in children:
                raise ValueError(f'Parent relation exists for {child}, but {child} is missing from the children of {parent}.')

        seen_children = set()

        for parent, children in self._children.items():

            parent_type, parent_id = parent
            if parent_type not in self._map_type_order:
                raise ValueError(f'Unknown parent map type: "{parent_type}"')

            expected_child_type = self.child_type(parent_type)

            for child in children:

                child_type, child_id = child
                if child in seen_children:
                    raise ValueError(f'Child occurs more than once in hierarchy: {child}')

                seen_children.add(child)

                if child_type != expected_child_type:
                    raise ValueError(f'Invalid relationship: "{parent_type}" -> "{child_type}". Expected child type: "{expected_child_type}"')

                if self._parent.get(child) != parent:
                    raise ValueError(f'Child relation exists for {parent} -> {child}, but the parent lookup is inconsistent.')

        for map_type, map_id in self._nodes:
            if map_type not in self._map_type_order:
                raise ValueError(f'Unknown node map type: "{map_type}"')


# ============================================================================
# Workflow / acquisition profile
# ============================================================================

class NavigatorProfile:
    MAP_TYPES: Sequence[str] = ()
    SEARCH_STRINGS: Mapping[str, str | None] = {}
    MAP_BINNINGS: Mapping[str, int] = {}
    MATCH_RULES: Mapping[str, Mapping[str, MatchRule]] = {}

    def __init__(
        self,
        search_strings: Mapping[str, str | None] | None = None,
        map_binnings: Mapping[str, int] | None = None,
        map_types: Sequence[str] | None = None,
        match_rules: Mapping[str, Mapping[str, Sequence[str] | MatchRule]] | None = None,
        stitched_dirpath: str | Path | None = None,
    ):
        self._map_types = list(self.MAP_TYPES if map_types is None else map_types)
        self._validate_map_types()

        if search_strings is None:
            self._search_strings = {map_type: self.SEARCH_STRINGS[map_type] for map_type in self._map_types}
        else:
            self._search_strings = dict(search_strings)

        if map_binnings is None:
            self._map_binnings = {map_type: self.MAP_BINNINGS[map_type] for map_type in self._map_types}
        else:
            self._map_binnings = dict(map_binnings)

        raw_match_rules = (self.MATCH_RULES if match_rules is None else match_rules)
        self._match_rules = {
            child_type: {map_type: MatchRule.from_config(rule) for map_type, rule in rules.items()}
            for child_type, rules in raw_match_rules.items()
        }

        self._stitched_dirpath = (None if stitched_dirpath is None else Path(stitched_dirpath))
        self.validate()

    @property
    def map_types(self) -> list[str]:
        return self._map_types.copy()

    @property
    def search_strings(self) -> dict[str, str | None]:
        return self._search_strings.copy()

    @property
    def map_binnings(self) -> dict[str, int]:
        return self._map_binnings.copy()

    @property
    def match_rules(self) -> dict[str, dict[str, MatchRule]]:
        return {child_type: dict(rules) for child_type, rules in self._match_rules.items()}

    @property
    def stitched_dirpath(self) -> Path | None:
        return self._stitched_dirpath

    def _validate_map_types(self) -> None:
        if not self._map_types:
            raise ValueError("At least one map type must be defined.")
        if len(self._map_types) != len(set(self._map_types)):
            raise ValueError("map_types contains duplicate values.")

        unknown_map_types = [map_type for map_type in self._map_types if map_type not in self.MAP_TYPES]
        if unknown_map_types:
            raise ValueError(f"Unknown map types: {unknown_map_types}. Allowed map types are: {list(self.MAP_TYPES)}")

        indices = [self.MAP_TYPES.index(map_type) for map_type in self._map_types]
        if any(current != previous + 1 for previous, current in zip(indices, indices[1:])):
            raise ValueError(f"map_types must form a contiguous hierarchy. Found: {self._map_types}")

    def validate(self) -> None:
        self._validate_map_types()

        missing_search_strings = [map_type for map_type in self._map_types if map_type not in self._search_strings]
        if missing_search_strings:
            raise ValueError(f"Missing search strings for map types: {missing_search_strings}")

        extra_search_strings = [map_type for map_type in self._search_strings if map_type not in self._map_types]
        if extra_search_strings:
            raise ValueError(f"Search strings supplied for inactive map types: {extra_search_strings}")

        missing_map_binnings = [map_type for map_type in self._map_types if map_type not in self._map_binnings]
        if missing_map_binnings:
            raise ValueError(f"Missing map binnings for map types: {missing_map_binnings}")

        extra_map_binnings = [map_type for map_type in self._map_binnings if map_type not in self._map_types]
        if extra_map_binnings:
            raise ValueError(f"Map binnings supplied for inactive map types: {extra_map_binnings}")

        for map_type, binning in self._map_binnings.items():
            if not isinstance(binning, int):
                raise TypeError(f'Binning for "{map_type}" must be an integer. Found: {type(binning).__name__}')
            if binning <= 0:
                raise ValueError(f'Binning for "{map_type}" must be > 0. Found: {binning}')

    def discover_items(self, navigator: "Navigator", map_type: str) -> dict[str, dict]:
        if map_type not in self._map_types:
            raise ValueError(f'Inactive map type: "{map_type}"')

        search_string = self._search_strings[map_type]
        if search_string is None:
            raise NotImplementedError(f'Automatic discovery is not defined for map_type="{map_type}" when search_string is None.')

        return get_map_items_by_glob(navigator.nav_dict, navigator.filepath, search_string)

    def source_filepath(
        self,
        navigator: "Navigator",
        map_type: str,
        map_id: str,
        item: Mapping[str, Any],
    ) -> Path:
        return get_map_filepath_from_nav_item(navigator.filepath, item)

    def resolve_filepath(
        self,
        navigator: "Navigator",
        map_type: str,
        map_id: str,
        item: Mapping[str, Any],
        source_filepath: Path,
    ) -> Path:
        """
        Return the filepath actually used by NavigatorMap.

        The generic behavior is to use the SerialEM source filepath unchanged.
        Workflow-specific profiles may override this.
        """
        return source_filepath

    def get_section_id(self, navigator: "Navigator", map_: NavigatorMap) -> int | None:
        note = map_.note
        if note is None:
            return None

        match = re.search(r"Sec\s+(\d+)", note)
        if match is None:
            return None

        return int(match.group(1))

    def match_id(self, navigator: "Navigator", map_: NavigatorMap, rule: MatchRule) -> str | None:
        return rule.match(map_.serialem_item, nav_dict=navigator.nav_dict)
    
    def build_relationships(self, navigator: "Navigator", hierarchy: MapHierarchy) -> None:
        """
        Build parent-child relationships using MATCH_RULES.

        Each MATCH_RULES entry defines how a child map type and its
        parent map type generate a common matching identifier.
        """

        for child_type, rules in self._match_rules.items():

            if child_type not in self._map_types:
                continue

            parent_type = hierarchy.parent_type(child_type)
            if parent_type is None:
                continue

            if parent_type not in rules:
                raise ValueError(f'Missing parent match rule for "{parent_type}" -> "{child_type}".')
            if child_type not in rules:
                raise ValueError(f'Missing child match rule for "{child_type}".')

            parent_rule = rules[parent_type]
            child_rule = rules[child_type]

            parent_matches = {}

            for parent_map in navigator.iter_maps(parent_type):
                match_id = self.match_id(navigator, parent_map, parent_rule)
                if match_id is None:
                    continue
                parent_matches.setdefault(match_id, []).append(parent_map.id)

            for child_map in navigator.iter_maps(child_type):
                match_id = self.match_id(navigator, child_map, child_rule)
                if match_id is None:
                    continue

                matching_parent_ids = parent_matches.get(match_id, [])
                if len(matching_parent_ids) == 0:
                    continue

                if len(matching_parent_ids) > 1:
                    raise ValueError(f'Ambiguous parent match for {child_type} "{child_map.id}": {matching_parent_ids}')

                hierarchy.add_relation(parent_type, matching_parent_ids[0], child_type, child_map.id)


# ============================================================================
# Single-particle workflow
# ============================================================================

class SingleParticleProfile(NavigatorProfile):

    MAP_TYPES = ("grid", "search", "view", "record")

    SEARCH_STRINGS = {
        "grid": "gridmap.st",
        "search": "*_search.mrc",
        "view": "*_view.mrc",
        "record": "*_record.mrc",
    }

    MAP_BINNINGS = {
        "grid": 8,
        "search": 4,
        "view": 1,
        "record": 1,
    }

    MATCH_RULES = {
        "view": {
            "search": MatchRule(
                item_key="MapID",
                regex=r"(.*)",
            ),
            "view": MatchRule(
                item_key="Note",
                regex=r"^(\d{1,2})",
                secondary_item_key="DrawnID",
                secondary_regex=r"(.*)",
            ),
        },
        "record": {
            "view": MatchRule(
                item_key="Note",
                regex=r"^(\d{1,2})",
            ),
            "record": MatchRule(
                item_key="Note",
                regex=r"^(\d{1,2})",
            ),
        },
    }

    def get_section_id_from_item(self, item: Mapping[str, Any]) -> int:
        note = item.get("Note")

        if note is None:
            raise ValueError("Navigator item has no Note entry.")

        section_id = match_regex(note, r"Sec\s+(\d+)")
        if section_id is None:
            raise ValueError(f"Could not determine section ID from Note: {note!r}")

        return int(section_id)

    def _find_matching_filepath(self, parent_filepath: Path, pattern: str) -> Path | None:

        for candidate_filepath in parent_filepath.iterdir():
            if re.search(pattern, candidate_filepath.name):
                return candidate_filepath
        return None

    def _resolve_full_resolution_section_filepath(self, source_filepath: Path, item: Mapping[str, Any], map_type: str) -> Path:

        parent_filepath = (self._stitched_dirpath if self._stitched_dirpath is not None else source_filepath.parent)
        section_id = self.get_section_id_from_item(item) + 1

        pattern = (rf"{re.escape(source_filepath.stem)}_0*{section_id}{re.escape(source_filepath.suffix)}$")

        filepath = self._find_matching_filepath(parent_filepath, pattern)
        if filepath is not None:
            return filepath

        if source_filepath.suffix.lower() != ".mrc":

            pattern_mrc = (rf"{re.escape(source_filepath.stem)}_0*{section_id}\.mrc$")

            filepath = self._find_matching_filepath(parent_filepath, pattern_mrc)
            if filepath is not None:
                return filepath

        raise FileNotFoundError(f'{map_type.capitalize()} map file not found for "{source_filepath}"')

    def discover_items(self, navigator: "Navigator", map_type: str) -> dict[str, dict]:

        if map_type not in self._map_types:
            raise ValueError(f'Inactive map type: "{map_type}"')
        if map_type != "grid":
            return super().discover_items(navigator, map_type)

        search_string = self._search_strings["grid"]

        if search_string is None:
            key = next(iter(navigator.nav_dict["items"]))
            item = navigator.nav_dict["items"][key]

            if "MapFile" not in item:
                raise ValueError("First Navigator item does not contain MapFile.")
            
            map_file = str(item["MapFile"]).lower()
            if not (map_file.endswith(".mrc") or map_file.endswith(".st")):
                raise ValueError(f'Grid MapFile must end with ".mrc" or ".st". Found: "{item["MapFile"]}"')

            return {key: item}

        map_items = get_map_items_by_glob(navigator.nav_dict, navigator.filepath, search_string)

        if len(map_items) == 0:
            raise ValueError("No grid map item was found.")
        if len(map_items) == 1:
            return map_items

        valid_map_items = {}

        for map_id, item in map_items.items():

            source_filepath = self.source_filepath(navigator, "grid", map_id, item)
            
            filepath = self.resolve_grid_filepath(navigator, source_filepath, item, allow_not_exist=True)
            if filepath is not None:
                valid_map_items[map_id] = item

        if len(valid_map_items) != 1:
            raise ValueError(f"{len(valid_map_items)} valid grid map items were found, but exactly one is required.")

        return valid_map_items

    def resolve_filepath(self, navigator: "Navigator", map_type: str, map_id: str, item: Mapping[str, Any], source_filepath: Path) -> Path:

        if map_type == "grid":
            filepath = self.resolve_grid_filepath(navigator, source_filepath, item)
            assert filepath is not None
            return filepath
        if map_type == "search":
            return self.resolve_search_filepath(navigator, source_filepath, item)
        if map_type == "view":
            return self.resolve_view_filepath(navigator, source_filepath, item)
        if map_type == "record":
            return self.resolve_record_filepath(navigator, source_filepath, item)

        return super().resolve_filepath(navigator, map_type, map_id, item, source_filepath)

    def resolve_grid_filepath(self, navigator: "Navigator", source_filepath: Path, item: Mapping[str, Any], allow_not_exist: bool = False) -> Path | None:

        binning = self._map_binnings["grid"]
        parent_filepath = (self._stitched_dirpath if self._stitched_dirpath is not None else source_filepath.parent)

        name = match_regex(item.get("Note", ""), r"^Grid \d{2} (\S+)")
        if name is None:
            if allow_not_exist:
                return None
            raise ValueError(f'Could not determine grid-map name from Note: {item.get("Note")!r}')

        grid_filepath = (parent_filepath / f"{source_filepath.stem}_{name}_bin{binning}" f"{source_filepath.suffix}")

        if grid_filepath.exists():
            return grid_filepath

        if source_filepath.suffix.lower() != ".mrc":

            grid_filepath_mrc = (parent_filepath / f"{source_filepath.stem}_{name}_bin{binning}.mrc")
            if grid_filepath_mrc.exists():
                return grid_filepath_mrc

        if allow_not_exist:
            return None
        raise FileNotFoundError(f'Grid map file not found for "{source_filepath}"')

    def resolve_search_filepath(self, navigator: "Navigator", source_filepath: Path, item: Mapping[str, Any]) -> Path:
        binning = self._map_binnings["search"]
        parent_filepath = (self._stitched_dirpath if self._stitched_dirpath is not None else source_filepath.parent)
        section_id = self.get_section_id_from_item(item) + 1

        pattern = (rf"{re.escape(source_filepath.stem)}_0*{section_id}_bin{binning}{re.escape(source_filepath.suffix)}$")
        filepath = self._find_matching_filepath(parent_filepath, pattern)

        if filepath is not None:
            return filepath

        if source_filepath.suffix.lower() != ".mrc":

            pattern_mrc = (rf"{re.escape(source_filepath.stem)}_0*{section_id}_bin{binning}\.mrc$")

            filepath = self._find_matching_filepath(parent_filepath, pattern_mrc)
            if filepath is not None:
                return filepath

        raise FileNotFoundError(f'Search map file not found for "{source_filepath}"')

    def resolve_view_filepath(self, navigator: "Navigator", source_filepath: Path, item: Mapping[str, Any]) -> Path:

        if self._map_binnings["view"] != 1:
            raise NotImplementedError("View maps are currently only implemented at full resolution.")

        return self._resolve_full_resolution_section_filepath(source_filepath, item, map_type="view")

    def resolve_record_filepath(self, navigator: "Navigator", source_filepath: Path, item: Mapping[str, Any]) -> Path:

        if self._map_binnings["record"] != 1:
            raise NotImplementedError("Record maps are currently only implemented at full resolution.")
        return self._resolve_full_resolution_section_filepath(source_filepath, item, map_type="record")
    
    def build_relationships(self, navigator: "Navigator", hierarchy: MapHierarchy) -> None:

        if "grid" in self._map_types and "search" in self._map_types:
            grid_id = navigator.get_grid_id()

            for search_id in navigator.get_map_ids("search"):
                hierarchy.add_relation("grid", grid_id, "search", search_id)

        super().build_relationships(navigator, hierarchy)


# ============================================================================
# TomoCLEM workflow
# ============================================================================

class TomoCLEMProfile(NavigatorProfile):

    MAP_TYPES = ("grid", "lamella", "view", "tgt")

    SEARCH_STRINGS = {
        "grid": None,
        "lamella": "L_*.map",
        "view": "L??_tgt_???_view.mrc",
        "tgt": "L??_tgt_???.mrc",
    }

    MAP_BINNINGS = {
        "grid": 8,
        "lamella": 8,
        "view": 1,
        "tgt": 1,
    }

    MATCH_RULES = {
        "view": {
            "lamella": MatchRule("MapFile", r"L_(\d{2})"),
            "view": MatchRule("MapFile", r"L(\d{2})"),
        },
        "tgt": {
            "view": MatchRule("MapFile", r"L(\d{2})_tgt_(\d{3})"),
            "tgt": MatchRule("MapFile", r"L(\d{2})_tgt_(\d{3})"),
        },
    }

    def discover_items(self, navigator: "Navigator", map_type: str) -> dict[str, dict]:
        return super().discover_items(navigator, map_type)

    def resolve_filepath(
        self,
        navigator: "Navigator",
        map_type: str,
        map_id: str,
        item: Mapping[str, Any],
        source_filepath: Path,
    ) -> Path:
        pass

    def resolve_grid_filepath(
        self,
        navigator: "Navigator",
        source_filepath: Path,
    ) -> Path:
        pass

    def resolve_lamella_filepath(
        self,
        navigator: "Navigator",
        source_filepath: Path,
    ) -> Path:
        pass

    def resolve_view_filepath(
        self,
        navigator: "Navigator",
        source_filepath: Path,
    ) -> Path:
        pass

    def resolve_target_filepath(
        self,
        navigator: "Navigator",
        source_filepath: Path,
    ) -> Path:
        pass

    def build_relationships(
        self,
        navigator: "Navigator",
        hierarchy: MapHierarchy,
    ) -> None:
        pass


# ============================================================================
# Main Navigator object
# ============================================================================

class Navigator:

    def __init__(self, filepaths: str | Path | Sequence[str | Path], profile: NavigatorProfile, verbose: bool = False):
        self.verbose = verbose
        self.profile = profile

        self.filepath: Path
        self.all_filepaths: list[Path]
        self.nav_dict: dict[str, Any]

        self.maps = MapCollection()

        self._load_navigator_files(filepaths)
        self._load_maps()
        self._build_hierarchy()

    # ------------------------------------------------------------------------
    # Loading / initialization
    # ------------------------------------------------------------------------

    def _load_navigator_files(self, filepaths: str | Path | Sequence[str | Path]) -> None:
        if isinstance(filepaths, (str, Path)):
            all_filepaths = [Path(filepaths)]
        else:
            all_filepaths = [Path(filepath) for filepath in filepaths]

        if not all_filepaths:
            raise ValueError("At least one Navigator filepath must be supplied.")

        self.filepath = all_filepaths[0]
        self.all_filepaths = all_filepaths

        nav_dict = navigator_file_to_dict(self.filepath)

        for filepath in self.all_filepaths[1:]:
            nav_dict = extend_navigator_dict(filepath, nav_dict)
        self.nav_dict = nav_dict

    def _load_maps(self) -> None:
        for map_type in self.profile.map_types:
            self._load_maps_of_type(map_type)

    def _load_maps_of_type(self, map_type: str) -> None:
        items = self.profile.discover_items(self, map_type)

        for map_id, item in items.items():
            map_ = self._create_map(map_type, map_id, item)
            self.maps.add(map_)

    def _create_map(self, map_type: str, map_id: str, item: Mapping[str, Any]) -> NavigatorMap:

        source_filepath = self.profile.source_filepath(self, map_type, map_id, item)
        filepath = self.profile.resolve_filepath(self, map_type, map_id, item, source_filepath)

        map_ = NavigatorMap(
            id=map_id,
            map_type=map_type,
            serialem_item=dict(item),
            source_filepath=source_filepath,
            filepath=filepath,
            binning=self.profile.map_binnings[map_type],
        )

        map_.section_id = self.profile.get_section_id(self, map_)
        self._load_initial_metadata(map_)
        return map_

    def _load_initial_metadata(self, map_: NavigatorMap) -> None:
        map_.load_resolution()
        map_.load_shape()

    def _build_hierarchy(self) -> None:
        self.hierarchy = MapHierarchy(self.profile.map_types)

        # Every map is a hierarchy node, regardless of whether it
        # participates in a parent-child relationship.
        for map_ in self.maps:
            self.hierarchy.add_node(map_.map_type, map_.id)

        self.profile.build_relationships(self, self.hierarchy)
        self.hierarchy.validate()
        
    def refresh(self) -> None:
        pass

    # ------------------------------------------------------------------------
    # Generic map access
    # ------------------------------------------------------------------------

    def get_map(self, map_type: str, map_id: str) -> NavigatorMap:
        return self.maps.get(map_type, map_id)

    def get_maps(self, map_type: str) -> dict[str, NavigatorMap]:
        return self.maps.by_type(map_type)

    def get_map_ids(self, map_type: str) -> list[str]:
        return self.maps.ids(map_type)

    def get_grid_map(self) -> NavigatorMap:

        grid_ids = self.get_map_ids("grid")
        if len(grid_ids) == 0:
            raise ValueError("No grid map found.")
        if len(grid_ids) > 1:
            raise ValueError(f"Expected exactly one grid map, found {len(grid_ids)}: {grid_ids}")

        return self.get_map("grid", grid_ids[0])

    def get_grid_id(self) -> str:
        return self.get_grid_map().id

    def find_item(self, map_type: str, serialem_item: str, regex: str, target_value: str) -> NavigatorMap | None:
        return self.maps.find(map_type=map_type, serialem_item=serialem_item, regex=regex, target_value=target_value)
    
    # ------------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------------

    def get_parent(self, map_type: str, map_id: str) -> NavigatorMap | None:

        parent_type = self.hierarchy.parent_type(map_type)
        if parent_type is None:
            return None

        parent_id = self.hierarchy.parent_id(map_type, map_id)
        if parent_id is None:
            return None

        return self.get_map(parent_type, parent_id)

    def get_children(self, map_type: str, map_id: str) -> list[NavigatorMap]:

        child_type = self.hierarchy.child_type(map_type)
        if child_type is None:
            return []

        child_ids = self.hierarchy.children_ids(map_type, map_id)
        return [self.get_map(child_type, child_id) for child_id in child_ids]

    def get_siblings(self, map_type: str, map_id: str) -> list[NavigatorMap]:
        sibling_ids = self.hierarchy.siblings_ids(map_type, map_id)
        return [self.get_map(map_type, sibling_id) for sibling_id in sibling_ids]

    def get_ancestors(self, map_type: str, map_id: str) -> list[NavigatorMap]:
        ancestor_keys = self.hierarchy.ancestors(map_type, map_id)
        return [self.get_map(ancestor_type, ancestor_id) for ancestor_type, ancestor_id in ancestor_keys]

    def get_ancestor(self, map_type: str, map_id: str, ancestor_type: str) -> NavigatorMap | None:
        for ancestor in self.get_ancestors(map_type, map_id):
            if ancestor.map_type == ancestor_type:
                return ancestor
        return None

    def get_descendants(self, map_type: str, map_id: str) -> list[NavigatorMap]:
        descendant_keys = self.hierarchy.descendants(map_type, map_id)
        return [self.get_map(descendant_type, descendant_id) for descendant_type, descendant_id in descendant_keys]

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    def get_map_resolution(self, map_type: str, map_id: str) -> np.ndarray:
        map_ = self.get_map(map_type, map_id)
        return map_.load_resolution()

    def get_map_shape(self, map_type: str, map_id: str) -> tuple[int, ...]:
        map_ = self.get_map(map_type, map_id)
        return map_.load_shape()

    def get_map_section_id(self, map_type: str, map_id: str) -> int | None:
        map_ = self.get_map(map_type, map_id)
        return map_.section_id

    def get_contrast_limits(self, map_type: str, force: bool = False) -> dict[str, tuple[float, float]]:
        return {map_.id: map_.load_contrast_limits(force=force) for map_ in self.iter_maps(map_type)}

    # ------------------------------------------------------------------------
    # Affine transforms
    # ------------------------------------------------------------------------

    def get_map_affine(
        self,
        map_type: str,
        map_id: str,
        *,
        invert: bool = False,
        full_square: bool = False,
        flatten: bool = False,
        rotate_90: bool = False,
        is_3d: bool = False,
    ) -> np.ndarray:

        map_ = self.get_map(map_type, map_id)
        return self._build_affine(
            map_,
            invert=invert,
            full_square=full_square,
            flatten=flatten,
            rotate_90=rotate_90,
            is_3d=is_3d,
        )

    def get_map_full_affine(
        self,
        map_type: str,
        map_id: str,
        *,
        apply_affine=None,
        invert: bool = False,
        full_square: bool = False,
        flatten: bool = False,
        rotate_90: bool = False,
        is_3d: bool = False,
    ) -> np.ndarray:

        affine = self.get_map_affine(
            map_type,
            map_id,
            invert=invert,
            full_square=True,
            flatten=False,
            rotate_90=rotate_90,
            is_3d=is_3d,
        )

        if apply_affine is not None:
            apply_affine = np.asarray(apply_affine, dtype=float)
            if apply_affine.shape != (4, 4):
                raise ValueError(f"apply_affine must have shape (4, 4). Found: {apply_affine.shape}")
            affine = (apply_affine @ affine)

        if not full_square:
            affine = affine[:3]

        if flatten:
            affine = affine.flatten()

        return affine
    
    def get_map_full_affines(
        self,
        map_type: str,
        *,
        stage_coordinate_system: bool = False,
        is_3d: bool = False,
        rotate_90: bool = False,
    ) -> dict[str, np.ndarray]:

        if stage_coordinate_system:
            apply_affine = None

        else:
            grid_map = self.get_grid_map()

            apply_affine = self.get_map_affine(
                grid_map.map_type,
                grid_map.id,
                invert=False,
                full_square=True,
                flatten=False,
                is_3d=is_3d,
            )

        return {
            map_.id: self.get_map_full_affine(
                map_.map_type,
                map_.id,
                apply_affine=apply_affine,
                invert=True,
                full_square=True,
                flatten=True,
                rotate_90=rotate_90,
                is_3d=is_3d,
            )
            for map_ in self.iter_maps(map_type)
        }

    def _build_affine(
        self,
        map_: NavigatorMap,
        *,
        invert: bool = False,
        full_square: bool = False,
        flatten: bool = False,
        rotate_90: bool = False,
        is_3d: bool = False,
    ) -> np.ndarray:

        stage_xy = map_.stage_xy
        map_scale = map_.map_scale_matrix

        if stage_xy is None:
            raise ValueError(f'Map "{map_.id}" has no StageXYZ entry.')
        if map_scale is None:
            raise ValueError(f'Map "{map_.id}" has no MapScaleMat entry.')

        shape_xyz = np.asarray(map_.load_shape(), dtype=float)[::-1]
        resolution_xyz = np.asarray(map_.load_resolution(), dtype=float)

        if not np.isclose(resolution_xyz[0], resolution_xyz[1]):
            raise ValueError(f"Affine construction requires isotropic XY resolution. Found: {resolution_xyz[:2]}")

        # The MRC resolution belongs to the image actually being visualized.
        # SerialEM's MapScaleMat, however, maps stage displacement [µm] to pixels of the original map image.
        # If our visualization image has subsequently been binned, convert its pixel size back to that original-map pixel size.
        serialem_resolution_xy = resolution_xyz[:2] / map_.binning

        # Convert SerialEM's stage -> pixel matrix into
        # stage -> physical-image-coordinate [µm].
        #
        #     Δp_pixels = MapScaleMat @ Δstage
        #
        # therefore
        #
        #     Δp_um = diag(pixel_size) @ MapScaleMat @ Δstage
        #
        stage_to_image_linear = (np.diag(serialem_resolution_xy) @ map_scale)

        if rotate_90:
            rotation = np.array([[0.0, -1.0],[1.0,  0.0]])
            stage_to_image_linear = stage_to_image_linear @ rotation

        # Physical coordinates of the center of the loaded image.
        image_center_xy = (shape_xyz[:2] * resolution_xyz[:2] / 2)

        # Construct stage -> image-physical-coordinate transform:
        #
        #     q = A @ (stage - StageXYZ) + center
        #       = A @ stage + (center - A @ StageXYZ)
        #
        affine = np.eye(4, dtype=float)
        affine[:2, :2] = stage_to_image_linear
        affine[:2, 3] = (image_center_xy - stage_to_image_linear @ stage_xy)

        if is_3d:
            affine[2, 3] = shape_xyz[2] * resolution_xyz[2] / 2
        if invert:
            affine = np.linalg.inv(affine)
        if not full_square:
            affine = affine[:3]
        if flatten:
            affine = affine.flatten()

        return affine

    # ------------------------------------------------------------------------
    # Dynamic map types / external data
    # ------------------------------------------------------------------------

    def add_maps(
        self,
        map_type: str,
        maps: Iterable[NavigatorMap],
        *,
        parent_map_type: str | None = None,
    ) -> None:
        pass

    def add_tomograms(
        self,
        dirpath: str | Path,
        pattern: str = "*.mrc",
        *,
        parent_map_type: str = "tgt",
        parent_match_rule: MatchRule | None = None,
        tomo_match_rule: MatchRule | None = None,
    ) -> None:
        pass

    def remove_map_type(self, map_type: str) -> None:
        pass

    # ------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------

    def iter_maps(self, map_type: str | None = None) -> Iterator[NavigatorMap]:
        if map_type is None:
            yield from self.maps
            return
        yield from self.maps.by_type(map_type).values()

    def iter_hierarchy(self, root_type: str | None = None, root_id: str | None = None) -> Iterator[tuple[str, NavigatorMap, list[int]]]:
        for map_type, map_id, idx_path in self.hierarchy.iter_depth_first(root_type=root_type, root_id=root_id):
            yield (self.get_map(map_type, map_id), idx_path)

    def __iter__(self) -> Iterator[tuple[str, NavigatorMap, list[int]]]:
        yield from self.iter_hierarchy()


# ============================================================================
# Convenience constructors
# ============================================================================

def single_particle_navigator(
    filepaths: str | Path | Sequence[str | Path],
    *,
    search_strings: Mapping[str, str | None] | None = None,
    map_binnings: Mapping[str, int] | None = None,
    map_types: Sequence[str] | None = None,
    stitched_dirpath: str | Path | None = None,
    verbose: bool = False,
) -> Navigator:
    """Construct a Navigator configured for the single-particle workflow."""
    pass


def tomo_clem_navigator(
    filepaths: str | Path | Sequence[str | Path],
    *,
    search_strings: Mapping[str, str | None] | None = None,
    map_binnings: Mapping[str, int] | None = None,
    map_types: Sequence[str] | None = None,
    stitched_dirpath: str | Path | None = None,
    verbose: bool = False,
) -> Navigator:
    """Construct a Navigator configured for the TomoCLEM workflow."""
    pass
