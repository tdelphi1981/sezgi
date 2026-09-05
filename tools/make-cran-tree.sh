#!/bin/sh -e
# tools/make-cran-tree.sh [--build] [--no-vendor] [--in-place DIR]
#
# Produces a self-contained r-sezgi package tree: the six sibling workspace
# crates (core, components, problems, bench, stats, bias) are copied in
# under src/rust/vendor-workspace/, their manifests and the bridge crate's
# manifest are rewritten to work without the monorepo's ../../../crates path
# dependencies (tools/cran_rewrite.py), the crates.io dependencies are
# vendored into a reproducible src/rust/vendor.tar.xz, and inst/AUTHORS +
# LICENSE.note are (re)generated to credit every dependency crate.
#
# Modes:
#   (default)          Wipes dist/cran and derives a self-contained copy of
#                       r-sezgi/ there, at dist/cran/sezgi. Nothing under
#                       r-sezgi/ or crates/ is ever modified. This is the
#                       CRAN release path.
#   --in-place DIR      Performs the same embed+rewrite directly inside DIR
#                       (which must already be an r-sezgi-shaped tree, e.g.
#                       obtained by checking out the whole monorepo) instead
#                       of copying into dist/cran first. Used by
#                       r-sezgi/.prepare, which R-universe runs against its
#                       own scratch checkout before `R CMD build`.
#   --no-vendor         Skips cargo vendor / vendor.tar.xz entirely: the six
#                       sibling crates are still embedded and rewritten, but
#                       the crates.io dependencies are left to be resolved
#                       online at build time (configure's
#                       `test -f src/rust/vendor.tar.xz` switch then keeps
#                       the build unvendored and parallel). Intended for
#                       R-universe, whose build servers have network access
#                       and a warm cargo cache, so vendoring there would only
#                       add a redundant local copy of the same crates.
#   --build             After preparing the tree, also run `R CMD build` on
#                       it (produces the .tar.gz next to the prepared tree).
#
# Idempotent: every run starts by removing whatever generated output would
# collide with what it is about to produce, then re-derives everything from
# the current working tree. Nothing under crates/ is ever modified by this
# script; embedding is always a copy.

# The `-e` in the shebang only takes effect when this file is invoked
# directly (./tools/make-cran-tree.sh); running it as `sh tools/make-cran-
# tree.sh` ignores the shebang's own flags entirely, so set it here too.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)

PKG_DIR=""
DO_VENDOR=yes
DO_BUILD=no

while [ $# -gt 0 ]; do
  case "$1" in
    --in-place)
      PKG_DIR=$2
      shift 2
      ;;
    --no-vendor)
      DO_VENDOR=no
      shift
      ;;
    --build)
      DO_BUILD=yes
      shift
      ;;
    *)
      echo "make-cran-tree.sh: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$PKG_DIR" ]; then
  OUT="$ROOT/dist/cran"
  rm -rf "$OUT"
  mkdir -p "$OUT"
  # 1. package tree: everything r-sezgi ships, minus generated/build-local
  #    files that configure/Makevars regenerate at build time and must
  #    never travel as stale copies.
  rsync -a \
    --exclude 'src/rust/target/' \
    --exclude 'src/rust/vendor/' \
    --exclude 'src/rust/vendor-workspace/' \
    --exclude 'src/rust/vendor.tar.xz' \
    --exclude 'src/Makevars' \
    --exclude 'src/Makevars.win' \
    --exclude 'src/*.o' \
    --exclude 'src/*.so' \
    --exclude 'src/*.dll' \
    "$ROOT/r-sezgi/" "$OUT/sezgi/"
  PKG_DIR="$OUT/sezgi"
else
  # in-place mode: PKG_DIR is already an r-sezgi tree (typically r-sezgi/
  # itself, inside a full checkout of the monorepo). Only clear what this
  # script itself is about to regenerate, so re-running it is safe.
  rm -rf "$PKG_DIR/src/rust/vendor-workspace" \
         "$PKG_DIR/src/rust/vendor" \
         "$PKG_DIR/src/rust/vendor.tar.xz"
fi

# 2. embed the six sibling crates, without their tests/benches/target -- the
#    CRAN tree never builds or runs those.
for c in core components problems bench stats bias; do
  rsync -a --exclude 'tests/' --exclude 'benches/' --exclude 'target/' \
    "$ROOT/crates/$c/" "$PKG_DIR/src/rust/vendor-workspace/$c/"
done

# 3. rewrite manifests: de-inherit workspace fields, drop [dev-dependencies],
#    and repoint the bridge crate's six path dependencies at the embedded
#    copies. See tools/cran_rewrite.py for why this must be line-wise rather
#    than a single regex over the whole file.
python3 "$ROOT/tools/cran_rewrite.py" "$PKG_DIR"

if [ "$DO_VENDOR" = "yes" ]; then
  # 4. vendor the crates.io dependencies, reproducibly, with xz.
  #    Cargo.lock travels unchanged: path dependencies carry no
  #    source/checksum entry, so rewriting their `path=` above does not
  #    invalidate the lockfile.
  #
  #    GNU tar's --sort=name/--mtime/--owner=/--group= make the archive
  #    byte-for-byte reproducible across runs and machines. Prefer GNU tar
  #    (as `gtar`, the common Homebrew name, or as `tar` itself on Linux);
  #    fall back to a bsdtar-compatible invocation on hosts that only have
  #    that (e.g. stock macOS) -- still deterministic member order and
  #    zeroed uid/gid, but each member's mtime is whatever it was on disk,
  #    so the resulting archive is not guaranteed byte-identical run to run.
  (
    cd "$PKG_DIR/src/rust"
    cargo vendor --locked vendor >/dev/null
    if command -v gtar >/dev/null 2>&1; then
      TAR_BIN=gtar
    else
      TAR_BIN=tar
    fi
    if "$TAR_BIN" --version 2>/dev/null | grep -q "GNU tar"; then
      "$TAR_BIN" --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 \
          --numeric-owner --xz --create --file=vendor.tar.xz vendor
    else
      echo "make-cran-tree.sh: warning: no GNU tar found (only '$("$TAR_BIN" --version | head -1)')," >&2
      echo "  vendor.tar.xz will be deterministic in content and member order but not in" >&2
      echo "  embedded mtimes. Install GNU tar (e.g. 'brew install gnu-tar' -> gtar) for a" >&2
      echo "  fully byte-reproducible release tarball." >&2
      find vendor -print | LC_ALL=C sort | \
        "$TAR_BIN" --no-recursion --numeric-owner --uid 0 --gid 0 \
            --create --xz --file=vendor.tar.xz --files-from -
    fi
    rm -rf vendor
  )
fi

# 5. (re)generate inst/AUTHORS + LICENSE.note, crediting every dependency
#    crate. Works whether or not step 4 ran: see tools/cran_authors.R's
#    header for its vendor/ -> vendor.tar.xz -> `cargo metadata` fallback
#    order.
Rscript "$ROOT/tools/cran_authors.R" "$PKG_DIR"

if [ "$DO_BUILD" = "yes" ]; then
  (cd "$(dirname "$PKG_DIR")" && R CMD build "$(basename "$PKG_DIR")")
fi
