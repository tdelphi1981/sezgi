#!/usr/bin/env python3
"""Rewrite Cargo manifests inside a source-embedded r-sezgi tree.

Usage: cran_rewrite.py PKG_DIR

PKG_DIR is a copy of r-sezgi/ into which the six sibling workspace crates
(core, components, problems, bench, stats, bias) have already been copied
under src/rust/vendor-workspace/<crate>/. This script performs the three
manifest edits that make that copy buildable on its own, without the
repository's root Cargo.toml or workspace:

1. De-inherit workspace fields. Each embedded crate's Cargo.toml declares
   `version.workspace = true`, `edition.workspace = true` and
   `license.workspace = true`, which only resolve inside the monorepo's
   workspace. Outside it, cargo fails with "error inheriting ... from
   workspace root manifest". Replace each with the literal value taken from
   the repository's root Cargo.toml ([workspace.package]).

2. Drop [dev-dependencies] from each embedded crate. src/rust/Cargo.toml
   keeps a bare `[workspace]` table so the embedded crates are visible as
   path dependencies; cargo then treats them as workspace members and would
   fold their dev-dependencies (test-only crates, e.g. tempfile) into
   `cargo vendor`, bloating vendor.tar.xz with packages never compiled from
   the tarball. Section removal is done line-by-line, keyed on lines that
   are themselves a bare "[section]" header -- a regex that instead matches
   "[dev-dependencies]...next [section]" as one span is unsafe here because
   at least one embedded crate has a dependency value containing a bracketed
   list, e.g. `serde_json = { version = "1", features =
   ["float_roundtrip"] }`, which is not a section header and must survive.

3. Repoint the bridge crate's six sibling-crate path dependencies. In the
   repository, src/rust/Cargo.toml depends on
   `{ path = "../../../crates/<name>" }`; in the embedded tree those crates
   live at src/rust/vendor-workspace/<name>/ instead, so rewrite the path.

The values used to de-inherit (WS_VERSION/WS_EDITION/WS_LICENSE) are read
directly from the repository's root Cargo.toml [workspace.package] table
(see `read_workspace_package` below), so a version bump there does not
require a matching edit here.
"""
import re
import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def read_workspace_package(repo_root: pathlib.Path) -> dict[str, str]:
    """Parse the [workspace.package] table of the repo's root Cargo.toml.

    Line-oriented, not a full TOML parser, matching this file's own
    manifest-editing style elsewhere: the table has a fixed, hand-written
    shape (one `key = "value"` per line) and is never machine-generated.
    """
    manifest = repo_root / "Cargo.toml"
    text = manifest.read_text()
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "[workspace.package]"),
        None,
    )
    if start is None:
        raise SystemExit(f"error: {manifest}: no [workspace.package] table found")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("["):
            end = i
            break
    block = "\n".join(lines[start:end])

    def extract(key: str) -> str:
        m = re.search(rf'(?m)^{key}\s*=\s*"([^"]*)"', block)
        if m is None:
            raise SystemExit(
                f"error: {manifest}: [workspace.package] has no '{key}' key"
            )
        return m.group(1)

    return {
        "version": extract("version"),
        "edition": extract("edition"),
        "license": extract("license"),
    }


_ws = read_workspace_package(REPO_ROOT)
WS_VERSION = _ws["version"]
WS_EDITION = _ws["edition"]
WS_LICENSE = _ws["license"]


def deinherit_and_strip_dev_deps(manifest_path: pathlib.Path) -> None:
    text = manifest_path.read_text()
    text = text.replace("version.workspace = true", f'version = "{WS_VERSION}"')
    text = text.replace("edition.workspace = true", f'edition = "{WS_EDITION}"')
    text = text.replace("license.workspace = true", f'license = "{WS_LICENSE}"')

    kept_lines = []
    in_dev_deps = False
    for line in text.splitlines():
        if line.startswith("["):
            in_dev_deps = line.strip() == "[dev-dependencies]"
        if not in_dev_deps:
            kept_lines.append(line)
    new_text = "\n".join(kept_lines).rstrip("\n") + "\n"
    manifest_path.write_text(new_text)


def repoint_bridge_paths(bridge_manifest: pathlib.Path) -> None:
    text = bridge_manifest.read_text()
    rewritten, count = re.subn(
        r'path\s*=\s*"\.\./\.\./\.\./crates/([a-z]+)"',
        r'path = "vendor-workspace/\1"',
        text,
    )
    if count == 0:
        raise SystemExit(
            f"error: {bridge_manifest}: no '../../../crates/<name>' path "
            "dependency found to rewrite -- has the bridge manifest shape "
            "changed?"
        )
    bridge_manifest.write_text(rewritten)
    print(f"rewrote {count} path dependencies in {bridge_manifest}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} PKG_DIR", file=sys.stderr)
        return 2

    pkg = pathlib.Path(argv[1])
    rust = pkg / "src" / "rust"
    vendor_workspace = rust / "vendor-workspace"

    manifests = sorted(vendor_workspace.glob("*/Cargo.toml"))
    if not manifests:
        raise SystemExit(
            f"error: no Cargo.toml found under {vendor_workspace} -- embed "
            "the sibling crates before running this script"
        )
    for manifest in manifests:
        deinherit_and_strip_dev_deps(manifest)
        print(f"rewrote {manifest.relative_to(pkg)}")

    repoint_bridge_paths(rust / "Cargo.toml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
