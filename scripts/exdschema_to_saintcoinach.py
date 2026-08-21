#!/usr/bin/env python3
"""
Convert xivdev/EXDSchema YAML definitions to xivapi/SaintCoinach JSON definitions.

This converter uses EXDSchema's `.github/columns.yml` to translate the
offset-ordered EXDSchema field list into SaintCoinach's EXH column-definition
index space.

Dependency:
    python -m pip install PyYAML

Typical usage with the EXDSchema `latest` branch:
    python exdschema_to_saintcoinach.py ./EXDSchema ./Definitions

Single file:
    python exdschema_to_saintcoinach.py \
        ./EXDSchema/Item.yml \
        ./Definitions \
        --columns-file ./EXDSchema/.github/columns.yml

Optional:
    --generic-target Item
    --generic-target Action
    --warning-report conversion_warnings.json
    --strict
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with: python -m pip install PyYAML"
    ) from exc


Json = dict[str, Any]
_PACKED_BOOL_RE = re.compile(r"^packedbool([0-7])$", re.IGNORECASE)


@dataclass
class ConversionWarning:
    sheet: str
    path: str
    code: str
    message: str


@dataclass(frozen=True)
class LeafNaming:
    path: str
    base_name: str
    array_scopes: tuple[str, ...]
    repeat_depth: int


@dataclass
class ExpandedLeaf:
    path: str
    base_name: str
    output_base_name: str
    scope_paths: tuple[str, ...]
    scope_indices: tuple[int, ...]
    field: Json

    @property
    def suffix(self) -> str:
        # SaintCoinach RepeatDataDefinition appends the innermost repeat index
        # first, then each outer repeat index.
        return "".join(f"[{i}]" for i in reversed(self.scope_indices))

    @property
    def final_name(self) -> str:
        return f"{self.output_base_name}{self.suffix}"


class ConversionError(RuntimeError):
    pass


class ColumnIndexResolver:
    """
    Maps EXDSchema's flattened offset order to SaintCoinach's EXH column index.

    EXDSchema fields are ordered by row byte offset. SaintCoinach's `index`
    refers to the original position of a column definition inside the EXH
    header. EXDSchema's `.github/columns.yml` stores those original column
    definitions, including their byte offsets.
    """

    def __init__(self, columns_file: Path) -> None:
        self.columns_file = columns_file
        self._sheets = self._load(columns_file)

    @staticmethod
    def _load(path: Path) -> dict[str, list[Json]]:
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                value = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConversionError(
                f"{path}: columns YAML parse failed: {exc}"
            ) from exc

        if not isinstance(value, dict):
            raise ConversionError(f"{path}: columns YAML root must be an object")

        result: dict[str, list[Json]] = {}
        for raw_name, raw_columns in value.items():
            if not isinstance(raw_name, str):
                raise ConversionError(f"{path}: invalid sheet name in columns file")
            if not isinstance(raw_columns, list):
                raise ConversionError(
                    f"{path}: columns entry for {raw_name!r} must be a list"
                )

            # EXDTools writes subrow sheets as "Sheet@Subrow". The EXDSchema file
            # itself is still named "Sheet".
            name = raw_name.removesuffix("@Subrow")
            if name in result:
                raise ConversionError(
                    f"{path}: duplicate normalized columns entry for {name!r}"
                )

            columns: list[Json] = []
            for i, raw_column in enumerate(raw_columns):
                if not isinstance(raw_column, dict):
                    raise ConversionError(
                        f"{path}: {raw_name}[{i}] must be an object"
                    )

                col_type = raw_column.get("type")
                offset = raw_column.get("offset")
                if not isinstance(col_type, str) or not col_type:
                    raise ConversionError(
                        f"{path}: {raw_name}[{i}].type must be a string"
                    )
                if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                    raise ConversionError(
                        f"{path}: {raw_name}[{i}].offset must be a non-negative integer"
                    )

                columns.append({"type": col_type, "offset": offset})

            result[name] = columns

        return result

    @staticmethod
    def _offset_sort_key(column: Json, original_index: int) -> tuple[int, int]:
        """
        Match the offset ordering used by EXDSchema consumers.

        Packed bool columns share a byte offset. Their packed-bool bit number
        disambiguates their order within that byte.
        """
        offset = int(column["offset"])
        col_type = str(column["type"]).lower()
        match = _PACKED_BOOL_RE.fullmatch(col_type)
        bit = int(match.group(1)) if match else 0
        return (offset * 8 + bit, original_index)

    def mapping_for(self, sheet: str, expected_count: int) -> list[int]:
        columns = self._sheets.get(sheet)
        if columns is None:
            raise ConversionError(
                f"{sheet}: no EXH column metadata found in {self.columns_file}"
            )

        if len(columns) != expected_count:
            raise ConversionError(
                f"{sheet}: EXDSchema expands to {expected_count} physical fields, "
                f"but {self.columns_file} contains {len(columns)} EXH columns. "
                "The schema and columns file must come from the same EXDSchema commit."
            )

        # The list position is the original EXH column-definition index used by
        # SaintCoinach. Sorting by offset reconstructs the order used by EXDSchema.
        sorted_columns = sorted(
            enumerate(columns),
            key=lambda pair: self._offset_sort_key(pair[1], pair[0]),
        )
        return [original_index for original_index, _column in sorted_columns]


class Converter:
    def __init__(
        self,
        *,
        column_resolver: ColumnIndexResolver,
        generic_targets: set[str] | None = None,
        strict: bool = False,
    ) -> None:
        self.column_resolver = column_resolver
        self.generic_targets = generic_targets or set()
        self.strict = strict
        self.warnings: list[ConversionWarning] = []
        self._sheet = ""
        self._name_overrides: dict[str, str] = {}
        self._warned_paths: set[tuple[str, str]] = set()

    def warn(
        self,
        path: str,
        code: str,
        message: str,
        *,
        lossy: bool = True,
        once: bool = False,
    ) -> None:
        if once:
            token = (path, code)
            if token in self._warned_paths:
                return
            self._warned_paths.add(token)

        warning = ConversionWarning(
            sheet=self._sheet,
            path=path,
            code=code,
            message=message,
        )
        self.warnings.append(warning)
        if self.strict and lossy:
            raise ConversionError(
                f"{self._sheet}: {path}: [{code}] {message}"
            )

    def convert_sheet(self, source: Json) -> Json:
        if not isinstance(source, dict):
            raise ConversionError("YAML root must be an object")

        sheet = source.get("name")
        fields = source.get("fields")
        if not isinstance(sheet, str) or not sheet:
            raise ConversionError("Missing or invalid top-level 'name'")
        if not isinstance(fields, list) or not fields:
            raise ConversionError(f"{sheet}: missing or invalid top-level 'fields'")

        self._sheet = sheet
        self._warned_paths = set()
        self._name_overrides = self.build_name_overrides(fields)

        result: Json = {"sheet": sheet}

        display_field = source.get("displayField")
        if display_field is not None:
            if not isinstance(display_field, str) or not display_field:
                raise ConversionError(f"{sheet}: invalid 'displayField'")
            result["defaultColumn"] = display_field

        if sheet in self.generic_targets:
            result["isGenericReferenceTarget"] = True

        if source.get("relations"):
            self.warn(
                "$.relations",
                "relations-flattened",
                "EXDSchema relation metadata is not emitted. Every physical field is "
                "written as an individual SaintCoinach definition, so AoS/SoA column "
                "positioning remains correct without group/repeat layout assumptions.",
                lossy=False,
            )

        leaves = self.expand_fields(fields)
        self.attach_converters(leaves)

        column_mapping = self.column_resolver.mapping_for(sheet, len(leaves))

        definitions: list[Json] = []
        for offset_index, leaf in enumerate(leaves):
            actual_index = column_mapping[offset_index]

            definition: Json = {"name": leaf.final_name}
            converter = leaf.field.get("__saintcoinach_converter")
            if converter is not None:
                definition["converter"] = converter

            if actual_index != 0:
                definition = {"index": actual_index, **definition}

            definitions.append(definition)

        definitions.sort(key=lambda d: int(d.get("index", 0)))
        result["definitions"] = definitions

        validate_generated_sheet(result, expected_count=len(leaves))
        self.validate_default_column(result)
        return result

    def validate_default_column(self, result: Json) -> None:
        default_column = result.get("defaultColumn")
        if default_column is None:
            return
        names = {
            d.get("name")
            for d in result["definitions"]
            if isinstance(d, dict)
        }
        if default_column not in names:
            raise ConversionError(
                f"{self._sheet}: defaultColumn {default_column!r} is not present "
                "in the generated SaintCoinach column names"
            )

    def expand_fields(self, fields: list[Json]) -> list[ExpandedLeaf]:
        leaves: list[ExpandedLeaf] = []
        for i, field in enumerate(fields):
            self.expand_field(
                field,
                inherited_name=None,
                path=f"$.fields[{i}]",
                scope_paths=(),
                scope_indices=(),
                out=leaves,
            )
        return leaves

    def expand_field(
        self,
        field: Json,
        *,
        inherited_name: str | None,
        path: str,
        scope_paths: tuple[str, ...],
        scope_indices: tuple[int, ...],
        out: list[ExpandedLeaf],
    ) -> None:
        if not isinstance(field, dict):
            raise ConversionError(f"{self._sheet}: {path}: field must be an object")

        own_name = field.get("name")
        if own_name is not None and not isinstance(own_name, str):
            raise ConversionError(f"{self._sheet}: {path}: invalid field name")

        effective_name = own_name or inherited_name
        field_type = field.get("type", "scalar")

        if "comment" in field:
            self.warn(
                f"{path}.comment",
                "comment-dropped",
                "SaintCoinach definitions have no field-comment property; comment omitted.",
                once=True,
            )

        if field.get("relations"):
            self.warn(
                f"{path}.relations",
                "relations-flattened",
                "Nested EXDSchema relation metadata is not emitted. Physical fields are "
                "flattened individually and retain correct EXH indices.",
                lossy=False,
                once=True,
            )

        if field_type != "array":
            if effective_name is None:
                raise ConversionError(
                    f"{self._sheet}: {path}: unnamed non-array field has no array name to inherit"
                )

            out.append(
                ExpandedLeaf(
                    path=path,
                    base_name=effective_name,
                    output_base_name=self.output_name(path, effective_name),
                    scope_paths=scope_paths,
                    scope_indices=scope_indices,
                    field=copy.deepcopy(field),
                )
            )
            return

        if effective_name is None:
            raise ConversionError(
                f"{self._sheet}: {path}: unnamed array has no name to inherit"
            )

        count = field.get("count")
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            raise ConversionError(f"{self._sheet}: {path}: array count must be a number")
        if int(count) != count or int(count) <= 1:
            raise ConversionError(
                f"{self._sheet}: {path}: array count must be an integer greater than 1"
            )
        count = int(count)

        children = field.get("fields")

        if children is None:
            # An array without an element descriptor is a repeated scalar.
            for repeat_index in range(count):
                out.append(
                    ExpandedLeaf(
                        path=path,
                        base_name=effective_name,
                        output_base_name=self.output_name(path, effective_name),
                        scope_paths=(*scope_paths, path),
                        scope_indices=(*scope_indices, repeat_index),
                        field={"type": "scalar"},
                    )
                )
            return

        if not isinstance(children, list) or not children:
            raise ConversionError(
                f"{self._sheet}: {path}.fields: must be a non-empty list"
            )

        for repeat_index in range(count):
            child_scope_paths = (*scope_paths, path)
            child_scope_indices = (*scope_indices, repeat_index)

            if len(children) == 1:
                self.expand_field(
                    children[0],
                    inherited_name=effective_name,
                    path=f"{path}.fields[0]",
                    scope_paths=child_scope_paths,
                    scope_indices=child_scope_indices,
                    out=out,
                )
            else:
                for i, child in enumerate(children):
                    self.expand_field(
                        child,
                        inherited_name=None,
                        path=f"{path}.fields[{i}]",
                        scope_paths=child_scope_paths,
                        scope_indices=child_scope_indices,
                        out=out,
                    )

    def attach_converters(self, leaves: list[ExpandedLeaf]) -> None:
        for leaf in leaves:
            converter = self.convert_value_converter(leaf, leaves)
            if converter is not None:
                leaf.field["__saintcoinach_converter"] = converter

    def convert_value_converter(
        self,
        leaf: ExpandedLeaf,
        all_leaves: list[ExpandedLeaf],
    ) -> Json | None:
        field = leaf.field
        path = leaf.path
        field_type = field.get("type", "scalar")

        if field_type == "scalar":
            return None

        if field_type == "icon":
            return {"type": "icon"}

        if field_type == "color":
            return {"type": "color"}

        if field_type == "modelId":
            self.warn(
                path,
                "modelid-downgraded",
                "EXDSchema 'modelId' has no corresponding SaintCoinach value converter; "
                "it is emitted as a plain scalar field.",
                once=True,
            )
            return None

        if field_type != "link":
            raise ConversionError(
                f"{self._sheet}: {path}: unsupported EXDSchema field type {field_type!r}"
            )

        if "targets" in field:
            targets = normalize_targets(field["targets"], self._sheet, path)
            if len(targets) == 1:
                return {"type": "link", "target": targets[0]}
            return {"type": "multiref", "targets": targets}

        condition = field.get("condition")
        if not isinstance(condition, dict):
            raise ConversionError(
                f"{self._sheet}: {path}: link requires 'targets' or 'condition'"
            )

        switch = condition.get("switch")
        cases = condition.get("cases")
        if not isinstance(switch, str) or not switch:
            raise ConversionError(
                f"{self._sheet}: {path}.condition.switch: invalid switch field"
            )
        if not isinstance(cases, dict) or not cases:
            raise ConversionError(
                f"{self._sheet}: {path}.condition.cases: must be a non-empty object"
            )

        resolved_switch = self.resolve_condition_switch(
            switch=switch,
            current_leaf=leaf,
            all_leaves=all_leaves,
        )

        links: list[Json] = []
        for raw_case, raw_targets in cases.items():
            try:
                case_value = int(raw_case)
            except (TypeError, ValueError) as exc:
                raise ConversionError(
                    f"{self._sheet}: {path}.condition.cases: "
                    f"case key {raw_case!r} is not an integer"
                ) from exc

            targets = normalize_targets(
                raw_targets,
                self._sheet,
                f"{path}.condition.cases[{raw_case!r}]",
            )

            link: Json
            if len(targets) == 1:
                link = {"sheet": targets[0]}
            else:
                link = {"sheets": targets}

            link["when"] = {
                "key": resolved_switch,
                "value": case_value,
            }
            links.append(link)

        return {"type": "complexlink", "links": links}

    def resolve_condition_switch(
        self,
        *,
        switch: str,
        current_leaf: ExpandedLeaf,
        all_leaves: list[ExpandedLeaf],
    ) -> str:
        """
        Resolve an EXDSchema condition key to the exact flattened SaintCoinach name.

        For a link repeated inside an array, a sibling switch field must refer to
        that same array element, for example `Type[3]`. Search from the deepest
        matching array scope outward until a unique switch field is found.
        """
        candidates = [leaf for leaf in all_leaves if leaf.base_name == switch]

        current_depth = len(current_leaf.scope_paths)
        for depth in range(current_depth, -1, -1):
            wanted_paths = current_leaf.scope_paths[:depth]
            wanted_indices = current_leaf.scope_indices[:depth]

            scoped = [
                leaf
                for leaf in candidates
                if leaf.scope_paths == wanted_paths
                and leaf.scope_indices == wanted_indices
            ]
            if len(scoped) == 1:
                return scoped[0].final_name
            if len(scoped) > 1:
                raise ConversionError(
                    f"{self._sheet}: {current_leaf.path}.condition.switch: "
                    f"switch field {switch!r} is ambiguous in array scope"
                )

        raise ConversionError(
            f"{self._sheet}: {current_leaf.path}.condition.switch: "
            f"cannot resolve switch field {switch!r} to a generated SaintCoinach column"
        )

    def output_name(self, path: str, base_name: str) -> str:
        return self._name_overrides.get(path, base_name)

    def build_name_overrides(self, fields: list[Json]) -> dict[str, str]:
        """
        Prevent final SaintCoinach column-name collisions.

        RepeatDataDefinition historically appended only [index] and did not retain
        the outer EXDSchema container name. If two arrays contain a member with the
        same name, both could become e.g. Name[0]. Qualify only ambiguous leaves.
        """
        leaves: list[LeafNaming] = []
        for i, field in enumerate(fields):
            self.collect_leaf_namings(
                field,
                inherited_name=None,
                path=f"$.fields[{i}]",
                array_scopes=(),
                out=leaves,
            )

        by_shape: dict[tuple[str, int], list[LeafNaming]] = {}
        for leaf in leaves:
            by_shape.setdefault((leaf.base_name, leaf.repeat_depth), []).append(leaf)

        overrides: dict[str, str] = {}
        for (base_name, _repeat_depth), group in by_shape.items():
            if len(group) <= 1:
                continue

            used: set[str] = set()
            for leaf in group:
                if leaf.array_scopes:
                    qualifier = ".".join(leaf.array_scopes)
                else:
                    stable = (
                        leaf.path.replace("$.fields", "F")
                        .replace(".fields", "F")
                        .replace("[", "")
                        .replace("]", "")
                    )
                    qualifier = stable

                candidate = f"{base_name}{{{qualifier}}}"
                if candidate in used:
                    stable = (
                        leaf.path.replace("$.fields", "F")
                        .replace(".fields", "F")
                        .replace("[", "")
                        .replace("]", "")
                    )
                    candidate = f"{base_name}{{{qualifier}.{stable}}}"

                used.add(candidate)
                overrides[leaf.path] = candidate
                self.warn(
                    leaf.path,
                    "column-name-disambiguated",
                    f"SaintCoinach would generate a duplicate column name from "
                    f"'{base_name}'. Emitting '{candidate}' to keep names unique.",
                    lossy=False,
                )

        return overrides

    def collect_leaf_namings(
        self,
        field: Json,
        *,
        inherited_name: str | None,
        path: str,
        array_scopes: tuple[str, ...],
        out: list[LeafNaming],
    ) -> None:
        if not isinstance(field, dict):
            raise ConversionError(f"{self._sheet}: {path}: field must be an object")

        own_name = field.get("name")
        if own_name is not None and not isinstance(own_name, str):
            raise ConversionError(f"{self._sheet}: {path}: invalid field name")

        effective_name = own_name or inherited_name
        field_type = field.get("type", "scalar")

        if field_type != "array":
            if effective_name is None:
                raise ConversionError(
                    f"{self._sheet}: {path}: unnamed non-array field has no array name to inherit"
                )
            out.append(
                LeafNaming(
                    path=path,
                    base_name=effective_name,
                    array_scopes=array_scopes,
                    repeat_depth=len(array_scopes),
                )
            )
            return

        if effective_name is None:
            raise ConversionError(
                f"{self._sheet}: {path}: unnamed array has no name to use as its scope"
            )

        scopes = (*array_scopes, effective_name)
        children = field.get("fields")

        if children is None:
            out.append(
                LeafNaming(
                    path=path,
                    base_name=effective_name,
                    array_scopes=scopes,
                    repeat_depth=len(scopes),
                )
            )
            return

        if not isinstance(children, list) or not children:
            raise ConversionError(
                f"{self._sheet}: {path}.fields: must be a non-empty list"
            )

        if len(children) == 1:
            self.collect_leaf_namings(
                children[0],
                inherited_name=effective_name,
                path=f"{path}.fields[0]",
                array_scopes=scopes,
                out=out,
            )
            return

        for i, child in enumerate(children):
            self.collect_leaf_namings(
                child,
                inherited_name=None,
                path=f"{path}.fields[{i}]",
                array_scopes=scopes,
                out=out,
            )


def normalize_targets(value: Any, sheet: str, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConversionError(f"{sheet}: {path}: targets must be a non-empty list")

    targets: list[str] = []
    for target in value:
        if not isinstance(target, str) or not target:
            raise ConversionError(f"{sheet}: {path}: invalid target {target!r}")
        targets.append(target)
    return targets


def validate_generated_sheet(sheet: Json, *, expected_count: int) -> None:
    sheet_name = sheet.get("sheet", "<unknown>")
    definitions = sheet.get("definitions")
    if not isinstance(definitions, list):
        raise ConversionError(f"{sheet_name}: invalid generated definitions")

    if len(definitions) != expected_count:
        raise ConversionError(
            f"{sheet_name}: generated {len(definitions)} definitions, "
            f"expected {expected_count}"
        )

    seen_names: dict[str, int] = {}
    seen_indices: dict[int, str] = {}

    for definition in definitions:
        if not isinstance(definition, dict):
            raise ConversionError(f"{sheet_name}: invalid generated definition")

        index = definition.get("index", 0)
        name = definition.get("name")

        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ConversionError(
                f"{sheet_name}: invalid generated definition index {index!r}"
            )
        if not isinstance(name, str) or not name:
            raise ConversionError(
                f"{sheet_name}: invalid generated definition name {name!r}"
            )

        previous_name = seen_indices.get(index)
        if previous_name is not None:
            raise ConversionError(
                f"{sheet_name}: generated duplicate EXH column index {index}: "
                f"{previous_name!r} and {name!r}"
            )
        seen_indices[index] = name

        previous_index = seen_names.get(name)
        if previous_index is not None:
            raise ConversionError(
                f"{sheet_name}: generated duplicate SaintCoinach column name "
                f"{name!r} at indices {previous_index} and {index}"
            )
        seen_names[name] = index

    if sorted(seen_indices) != list(range(expected_count)):
        missing = sorted(set(range(expected_count)) - set(seen_indices))
        extra = sorted(set(seen_indices) - set(range(expected_count)))
        raise ConversionError(
            f"{sheet_name}: generated column-index coverage is invalid; "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )


def load_yaml(path: Path) -> Json:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            value = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConversionError(f"{path}: YAML parse failed: {exc}") from exc

    if not isinstance(value, dict):
        raise ConversionError(f"{path}: YAML root must be an object")
    return value


def discover_yaml_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {".yml", ".yaml"}:
            raise ConversionError(f"input file must be .yml or .yaml: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ConversionError(f"input path does not exist: {input_path}")

    # On EXDSchema's `latest` branch the sheet YAMLs are repository-root files.
    direct = sorted(
        p
        for pattern in ("*.yml", "*.yaml")
        for p in input_path.glob(pattern)
    )
    if direct:
        return direct

    # On EXDSchema `main`, schemas/latest is a submodule.
    latest = input_path / "schemas" / "latest"
    if latest.is_dir():
        latest_direct = sorted(
            p
            for pattern in ("*.yml", "*.yaml")
            for p in latest.glob(pattern)
        )
        if latest_direct:
            return latest_direct

    recursive = sorted(
        {
            p
            for pattern in ("*.yml", "*.yaml")
            for p in input_path.rglob(pattern)
            if ".github" not in p.parts
        }
    )
    if not recursive:
        raise ConversionError(
            f"no EXDSchema .yml/.yaml definitions found under {input_path}"
        )
    return recursive


def discover_columns_file(
    input_path: Path,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise ConversionError(f"columns file does not exist: {explicit}")
        return explicit

    start = input_path if input_path.is_dir() else input_path.parent
    candidates: list[Path] = []

    # Most common case: checkout of the `latest` branch.
    candidates.append(start / ".github" / "columns.yml")

    # If caller points at schemas/latest from a main checkout.
    candidates.append(start / "schemas" / "latest" / ".github" / "columns.yml")

    # Also search a few parents for single-file invocation.
    for parent in [start, *list(start.parents)[:4]]:
        candidates.append(parent / ".github" / "columns.yml")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    raise ConversionError(
        "Could not find EXDSchema .github/columns.yml. "
        "Use an EXDSchema `latest` checkout as input or pass "
        "--columns-file /path/to/EXDSchema/.github/columns.yml. "
        "Accurate SaintCoinach indices cannot be generated from sheet YAML alone."
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def convert_paths(
    input_path: Path,
    output_dir: Path,
    *,
    columns_file: Path,
    generic_targets: set[str],
    strict: bool,
) -> tuple[int, list[ConversionWarning]]:
    files = discover_yaml_files(input_path)
    resolver = ColumnIndexResolver(columns_file)
    converter = Converter(
        column_resolver=resolver,
        generic_targets=generic_targets,
        strict=strict,
    )

    converted = 0
    output_names: set[str] = set()

    for source_path in files:
        source = load_yaml(source_path)
        result = converter.convert_sheet(source)
        sheet = result["sheet"]

        if sheet in output_names:
            raise ConversionError(
                f"duplicate sheet name {sheet!r}; input discovery found multiple versions. "
                "Point the script at one EXDSchema version, preferably the `latest` checkout."
            )
        output_names.add(sheet)

        write_json(output_dir / f"{sheet}.json", result)
        converted += 1

    return converted, converter.warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert xivdev/EXDSchema YAML to SaintCoinach Definitions JSON "
            "using EXDSchema EXH column metadata."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "An EXDSchema YAML file, a checkout of the `latest` branch, "
            "or a repo containing schemas/latest."
        ),
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for SaintCoinach .json definitions.",
    )
    parser.add_argument(
        "--columns-file",
        type=Path,
        help=(
            "Path to EXDSchema .github/columns.yml. Auto-detected when input "
            "is an EXDSchema checkout."
        ),
    )
    parser.add_argument(
        "--generic-target",
        action="append",
        default=[],
        metavar="SHEET",
        help=(
            "Mark this sheet as isGenericReferenceTarget=true. "
            "May be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--warning-report",
        type=Path,
        help="Write conversion warnings to this JSON file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on metadata that cannot be represented losslessly.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        columns_file = discover_columns_file(args.input, args.columns_file)
        count, warnings = convert_paths(
            args.input,
            args.output,
            columns_file=columns_file,
            generic_targets=set(args.generic_target),
            strict=args.strict,
        )
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.warning_report:
        write_json(args.warning_report, [asdict(w) for w in warnings])

    for warning in warnings:
        print(
            f"warning: {warning.sheet}: {warning.path}: "
            f"[{warning.code}] {warning.message}",
            file=sys.stderr,
        )

    print(f"columns metadata: {columns_file}")
    print(
        f"converted {count} sheet(s) to {args.output} "
        f"with {len(warnings)} warning(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
