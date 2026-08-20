#!/usr/bin/env python3
"""
Convert xivdev/EXDSchema YAML definitions to xivapi/SaintCoinach JSON definitions.

Dependency:
    pip install PyYAML

Typical usage:
    python exdschema_to_saintcoinach.py ./EXDSchema ./Definitions
    python exdschema_to_saintcoinach.py ./EXDSchema/schemas/latest ./Definitions
    python exdschema_to_saintcoinach.py ./GCSupplyDuty.yml ./Definitions

Optional:
    --generic-target Item
    --generic-target Action
    --warning-report conversion_warnings.json
    --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with: python -m pip install PyYAML"
    ) from exc


Json = dict[str, Any]


@dataclass
class ConversionWarning:
    sheet: str
    path: str
    code: str
    message: str


class ConversionError(RuntimeError):
    pass


class Converter:
    def __init__(
        self,
        *,
        generic_targets: set[str] | None = None,
        strict: bool = False,
    ) -> None:
        self.generic_targets = generic_targets or set()
        self.strict = strict
        self.warnings: list[ConversionWarning] = []
        self._sheet = ""
        self._conditional_switches: list[tuple[str, str]] = []

    def warn(self, path: str, code: str, message: str, *, lossy: bool = True) -> None:
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
        self._conditional_switches = []

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
                "relations-dropped",
                "EXDSchema relations have no direct SaintCoinach JSON equivalent; "
                "the physical field layout is preserved and relation metadata is omitted.",
            )

        definitions: list[Json] = []
        column_index = 0

        for i, field in enumerate(fields):
            path = f"$.fields[{i}]"
            definition = self.convert_field(field, inherited_name=None, path=path)
            length = definition_length(definition)

            positioned = definition if column_index == 0 else {"index": column_index, **definition}
            definitions.append(positioned)
            column_index += length

        result["definitions"] = definitions

        # SaintCoinach ComplexLinkConverter resolves a "when.key" by searching
        # top-level positioned definitions and comparing InnerDefinition.GetName(0).
        # A top-level scalar-like field is safe. A top-level repeat/group generally
        # exposes a suffixed or nested first name and may not resolve the source key.
        scalar_top_level_names = {
            f.get("name")
            for f in fields
            if isinstance(f, dict)
            and isinstance(f.get("name"), str)
            and f.get("type", "scalar") != "array"
        }
        for switch, path in self._conditional_switches:
            if switch not in scalar_top_level_names:
                self.warn(
                    path,
                    "conditional-key-may-not-resolve",
                    f"conditional link uses switch field '{switch}', which is not a "
                    "top-level scalar-like definition. SaintCoinach resolves complex-link "
                    "condition keys against top-level definition names, so this case may "
                    "need a manual definition rewrite.",
                )

        return result

    def convert_field(
        self,
        field: Json,
        *,
        inherited_name: str | None,
        path: str,
    ) -> Json:
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
            )

        if field.get("relations"):
            self.warn(
                f"{path}.relations",
                "relations-dropped",
                "EXDSchema array relations have no direct SaintCoinach JSON equivalent; "
                "the physical field layout is preserved and relation metadata is omitted.",
            )

        if field_type == "array":
            return self.convert_array(
                field,
                effective_name=effective_name,
                path=path,
            )

        if effective_name is None:
            raise ConversionError(
                f"{self._sheet}: {path}: unnamed non-array field has no array name to inherit"
            )

        definition: Json = {"name": effective_name}

        converter = self.convert_value_converter(field, path=path)
        if converter is not None:
            definition["converter"] = converter

        return definition

    def convert_array(
        self,
        field: Json,
        *,
        effective_name: str | None,
        path: str,
    ) -> Json:
        count = field.get("count")
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            raise ConversionError(f"{self._sheet}: {path}: array count must be a number")
        if int(count) != count or int(count) <= 1:
            raise ConversionError(
                f"{self._sheet}: {path}: array count must be an integer greater than 1"
            )
        count = int(count)

        children = field.get("fields")

        # EXDSchema allows an array without an explicit element descriptor.
        # Its element is treated as a plain scalar carrying the array's own name.
        if children is None:
            if effective_name is None:
                raise ConversionError(
                    f"{self._sheet}: {path}: unnamed array without fields has no name to inherit"
                )
            repeated: Json = {"name": effective_name}

        else:
            if not isinstance(children, list) or not children:
                raise ConversionError(
                    f"{self._sheet}: {path}.fields: must be a non-empty list"
                )

            if len(children) == 1:
                # In EXDSchema, a one-field array uses an unnamed field descriptor.
                # Its semantic name is the containing array's name.
                repeated = self.convert_field(
                    children[0],
                    inherited_name=effective_name,
                    path=f"{path}.fields[0]",
                )
            else:
                members: list[Json] = []
                for i, child in enumerate(children):
                    member = self.convert_field(
                        child,
                        inherited_name=None,
                        path=f"{path}.fields[{i}]",
                    )
                    members.append(member)
                repeated = {"type": "group", "members": members}

        return {
            "type": "repeat",
            "count": count,
            "definition": repeated,
        }

    def convert_value_converter(self, field: Json, *, path: str) -> Json | None:
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

        links: list[Json] = []
        for raw_case, raw_targets in cases.items():
            try:
                case_value = int(raw_case)
            except (TypeError, ValueError) as exc:
                raise ConversionError(
                    f"{self._sheet}: {path}.condition.cases: "
                    f"case key {raw_case!r} is not an integer"
                ) from exc

            # EXDSchema's live `latest` definitions may use 0 as a
            # conditional discriminator (for example MKDRelicGrowth2ContentList).
            # SaintCoinach's ComplexLinkConverter compares when.value directly
            # and imposes no positive-integer restriction, so preserve any
            # integer case value verbatim.

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

            link["when"] = {"key": switch, "value": case_value}
            links.append(link)

        self._conditional_switches.append(
            (switch, f"{path}.condition.switch")
        )

        return {"type": "complexlink", "links": links}


def normalize_targets(value: Any, sheet: str, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConversionError(f"{sheet}: {path}: targets must be a non-empty list")
    targets: list[str] = []
    for target in value:
        if not isinstance(target, str) or not target:
            raise ConversionError(f"{sheet}: {path}: invalid target {target!r}")
        targets.append(target)
    return targets


def definition_length(definition: Json) -> int:
    """
    Return the number of physical EXH columns consumed by a SaintCoinach IDataDefinition.
    """
    kind = definition.get("type")

    if kind == "repeat":
        count = definition.get("count")
        child = definition.get("definition")
        if not isinstance(count, int) or not isinstance(child, dict):
            raise ConversionError("invalid generated repeat definition")
        return count * definition_length(child)

    if kind == "group":
        members = definition.get("members")
        if not isinstance(members, list):
            raise ConversionError("invalid generated group definition")
        return sum(definition_length(member) for member in members)

    # SingleDataDefinition
    if "name" not in definition:
        raise ConversionError(f"invalid generated single definition: {definition!r}")
    return 1


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

    # On the EXDSchema "latest" branch, definitions are in the repository root.
    direct = sorted(
        p for pattern in ("*.yml", "*.yaml") for p in input_path.glob(pattern)
    )
    if direct:
        return direct

    # On EXDSchema main, "schemas/latest" is the preferred submodule.
    latest = input_path / "schemas" / "latest"
    if latest.is_dir():
        latest_direct = sorted(
            p for pattern in ("*.yml", "*.yaml") for p in latest.glob(pattern)
        )
        if latest_direct:
            return latest_direct

    # Fallback for a custom checkout layout.
    recursive = sorted(
        {
            p
            for pattern in ("*.yml", "*.yaml")
            for p in input_path.rglob(pattern)
        }
    )
    if not recursive:
        raise ConversionError(f"no .yml/.yaml definitions found under {input_path}")
    return recursive


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def convert_paths(
    input_path: Path,
    output_dir: Path,
    *,
    generic_targets: set[str],
    strict: bool,
) -> tuple[int, list[ConversionWarning]]:
    files = discover_yaml_files(input_path)
    converter = Converter(
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
                "Point the script at one EXDSchema version, preferably schemas/latest."
            )
        output_names.add(sheet)

        output_path = output_dir / f"{sheet}.json"
        write_json(output_path, result)
        converted += 1

    return converted, converter.warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert xivdev/EXDSchema YAML to SaintCoinach Definitions JSON."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="An EXDSchema YAML file, the latest branch directory, or the EXDSchema repo root.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for SaintCoinach .json definitions.",
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
        help="Write lossy-conversion warnings to this JSON file.",
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
        count, warnings = convert_paths(
            args.input,
            args.output,
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

    print(
        f"converted {count} sheet(s) to {args.output} "
        f"with {len(warnings)} warning(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
