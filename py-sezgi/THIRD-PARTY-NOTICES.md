# Third-party notices

sezgi's own source code is MIT-licensed (see `LICENSE`). This file inventories
every externally-sourced artifact that sezgi's Rust source quotes, mirrors, or
vendors -- either as short excerpts inside documentation comments (used to
pin sezgi's own, independently-written implementation against the reference
behaviour it reimplements; the excerpts are never compiled) or as vendored
data files. For each source: what is used, where it lives in the tree, its
license status as found, and the attribution given. See also
`r-sezgi/inst/COPYRIGHTS`, the R-package-local copy of this inventory
referenced by `r-sezgi/DESCRIPTION`'s `Copyright` field.

None of the material below is a dependency of the compiled sezgi library at
build time; it is either quoted documentation or vendored benchmark data.

## 1. KanGAL / Dr. Kalyanmoy Deb -- NSGA-II reference C (2005)

- **Used:** ~340 lines of reference C quoted verbatim in documentation
  comments (`check_dominance`, `evaluate_ind`, `realcross`,
  `real_mutate_ind`, `bincross`, `assign_crowding_distance`,
  `initialize.c`, `tourselect.c`, and related routines), pinning the
  constrained-domination truth table, the crowding-distance degenerate
  case, SBX/polynomial-mutation draw order, and the `EPS = 1.0e-14`
  constant.
- **Where:** `crates/components/src/nsga2.rs`, `crates/components/src/bin_ops.rs`.
- **Source:** Kanpur Genetic Algorithms Laboratory (KanGAL, IIT Kanpur)
  original C implementation, via the GitHub mirror
  `https://github.com/darnir/nsga2` (verified against the repository's own
  blob SHAs).
- **License status:** no OSI license; the repository carries a file named
  `Notice` with a non-commercial, academic-use notice, reproduced verbatim
  below.
- **Attribution given:** sezgi's Rust NSGA-II is an independently written,
  differently structured implementation (its own function signatures,
  control flow, and documented deviations from the C); the quoted C is
  inert documentation, not compiled code. The `Notice` text is reproduced
  here in full, per its own request for acknowledgement:

  ```
  /*************************************************************************
   * Copyright Notice:                                                     *
   * Source code for random number generator (files rand.h & rand.c) has   *
   * been taken from : sga.c (C) David E. Goldberg 1986.                   *
   * Entire source code (other than mentioned above) present in this       *
   * directory has been developed (from scratch) at Kanpur Genetic         *
   * Algorithms Laboratory (KanGAL, IIT Kanpur) and is the property of its *
   * authors. (C) Dr. Kalyanmoy Deb 2005.                                  *
   *                                                                       *
   * Disclaimer Notice:                                                    *
   * These codes have been developed for research purpose and are in a     *
   * process of continous change, and therefore bugs may exist.            *
   * In any case, developers of the codes do not take any responsibility   *
   * of any malfunction, although they have been tested on many test       *
   * problems. These codes have been tested on Mandrake and Gentoo linux   *
   * (kernel version 2.6.x and gcc version 3.3.x). Any bug or error may    *
   * kindly be communicated to deb@iitk.ac.in. Commercial use of these     *
   * codes is strictly prohibited without the knowledge of developers. For *
   * academic use, these can be used or modified  at will, however an      *
   * acknowledgement of developers at appropriate places would be highly   *
   * appreciated.                                                          *
   *************************************************************************/
  ```

## 2. Yarpiz / Mostapha Kalami Heris -- TLBO reference MATLAB (2015)

- **Used:** ~53 lines of `tlbo.m` quoted verbatim in documentation comments
  (population-mean, teacher-selection, teacher-phase and learner-phase
  loops), used as the verified secondary/fallback reference since Rao's own
  TLBO code could not be independently located.
- **Where:** `crates/components/src/tlbo.rs`.
- **Source:** Mostapha Kalami Heris, "Teaching-Learning-based Optimization
  in MATLAB", Project Code YPEA111, Yarpiz
  (`https://yarpiz.com/83/ypea111-teaching-learning-based-optimization`),
  mirrored at `https://github.com/smkalami/ypea111-teaching-learning-based-optimization`.
- **License status:** BSD-2-Clause-style, with an explicit
  redistribution-notice-retention condition. Reproduced verbatim below, per
  that condition:

  ```
  Copyright (c) 2015, Mostapha Kalami Heris & Yarpiz (www.yarpiz.com)
  All rights reserved.

  Cite as:
  Mostapha Kalami Heris, Teaching-Learning-based Optimization in MATLAB (URL: https://yarpiz.com/83/ypea111-teaching-learning-based-optimization), Yarpiz, 2015.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the distribution

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
  ```

## 3. Karaboga & Basturk / Erciyes University -- ABC reference MATLAB (2009)

- **Used:** ~63 lines of `ABCorig.m` quoted verbatim in documentation
  comments (control-parameter section, employed/onlooker/scout phases).
- **Where:** `crates/components/src/abc.rs`.
- **Source:** Dervis Karaboga and Bahriye Basturk's own reference MATLAB
  (`ABCorig.m`), authored at Erciyes University, Intelligent Systems
  Research Group, Dept. of Computer Engineering. The official homepage
  (`abc.erciyes.edu.tr/software.htm`) gates its downloads behind
  registration; the file was fetched via a third-party mirror,
  `https://github.com/tianxiangyi/MATLAB-ABC-Algorithm`, whose own file
  header carries Karaboga & Basturk's original copyright notice ("Copyright
  (c) 2009 Erciyes University, Intelligent Systems Research Group, The
  Dept. of Computer Engineering").
- **License status:** no license file accompanies the file; the header
  identifies the authors and their institution but states no redistribution
  terms.
- **Attribution given:** quoted for provenance documentation, with the
  authors, institution, and originating publications (Karaboga 2005,
  Karaboga & Basturk 2007/2008/2009) named in the module documentation.

## 4. Mirjalili / Yang / Rashedi / Heidari -- author-published MATLAB reference implementations

- **Used:** 16-57 lines each, quoted verbatim in documentation comments,
  pinning update equations and loop structure for algorithms whose authors
  published their own MATLAB source on MATLAB Central File Exchange.
- **Where:**
  - `crates/components/src/gwo.rs` -- Grey Wolf Optimizer (Mirjalili,
    Mirjalili & Lewis 2014).
  - `crates/components/src/woa.rs` -- Whale Optimization Algorithm
    (`WOA.m`, File Exchange #55667; Mirjalili & Lewis 2016).
  - `crates/components/src/mfo.rs` -- Moth-Flame Optimization (`MFO.m`,
    Mirjalili 2015).
  - `crates/components/src/alo.rs` -- Ant Lion Optimizer (`antlion.m` /
    `RouletteWheelSelection.m`, File Exchange #49920; Mirjalili 2015).
  - `crates/components/src/ssa.rs` -- Salp Swarm Algorithm (`SSA.m`,
    Mirjalili et al. 2017).
  - `crates/components/src/fa.rs` -- Firefly Algorithm (Xin-She Yang's own
    File Exchange submission #29693).
  - `crates/components/src/ba.rs` -- Bat Algorithm (`bat_algorithm.m`,
    Xin-She Yang's own File Exchange submission #37582; Yang 2010).
  - `crates/components/src/fpa.rs` -- Flower Pollination Algorithm
    (`fpa_demo.m`, File Exchange #45112; Yang 2012).
  - `crates/components/src/gsa.rs` -- Gravitational Search Algorithm
    (`Gfield.m`, `move.m`, Esmat Rashedi's own File Exchange submission
    #27756; Rashedi, Nezamabadi-pour & Saryazdi 2009).
  - `crates/components/src/hho.rs` -- Harris Hawks Optimization (`HHO.m`,
    Ali Asghar Heidari's own reference implementation; Heidari et al. 2019).
- **Source:** each author's own MATLAB Central File Exchange submission
  (linked from, or explicitly identified as accompanying, the cited
  peer-reviewed paper), fetched via MathWorks File Exchange preview mirrors
  or the authors' own GitHub accounts.
- **License status:** File Exchange submissions carry MathWorks' standard
  submission terms; none of the quoted files carries a separate,
  independently-hosted OSI license. Each is quoted for provenance
  documentation only.
- **Attribution given:** author name, File Exchange submission number, and
  the originating publication are named in each module's documentation.
  These algorithms are additionally labeled "labeled metaphor presets" in
  sezgi's own documentation (`examples/README.md`), alongside the
  equivalence-critique literature (Camacho-Villalón, Dorigo & Stützle) that
  questions their novelty relative to established metaheuristics.

## 5. The Walking Fish Group (WFG) -- WFG toolkit C++ (2005/2006)

- **Used:** ~96 lines of the official WFG toolkit C++ quoted verbatim in
  documentation comments (shape/transformation function signatures and the
  `D`-scaling divergence between the EMO2005 paper and the 2006 toolkit).
- **Where:** `crates/problems/src/wfg.rs`.
- **Source:** the official WFG Toolkit C++ (`WFG_v2006.03.28/`), cloned from
  `https://github.com/richardeverson/wfg` (canonical host
  `wfg.csse.uwa.edu.au` is dead; corroborated by a Wayback Machine capture
  of the same archive from the original site).
- **License status:** the wrapper repository's root `LICENSE` (GPLv2)
  covers only the mirror maintainer's own Python wrapper; every file under
  `WFG_v2006.03.28/` instead carries the WFG authors' own permissive
  academic notice:

  ```
  Copyright (c) 2005 The Walking Fish Group (WFG). This material is
  provided "as is", with no warranty expressed or implied. Any use is at
  your own risk. Permission to use or copy this software for any purpose
  is hereby granted without fee, provided this notice is retained on all
  copies. Permission to modify the code and to distribute modified code is
  granted, provided a notice that the code was modified is included with
  the above copyright notice.
  ```
- **Attribution given:** reproduced above per its own retention condition;
  no line of the C++ enters sezgi's compiled code (used oracle-only, to
  cross-check an independent Rust reimplementation written from the
  published paper).

## 6. CEC 2014 / 2017 / 2022 -- reference C and benchmark data tables

- **Used:** reference C excerpts quoted in documentation comments (173,
  179 and 31 lines respectively), plus vendored shift-vector,
  rotation-matrix and shuffle-index data tables used at runtime by the
  benchmark suites.
- **Where:** `crates/problems/src/cec2014/`, `crates/problems/src/cec2017/`,
  `crates/problems/src/cec2022/`, and their data under
  `crates/problems/data/{cec2014,cec2017,cec2022}/`.
- **Source:** the CEC Special Session and Competition organizers' own
  official repositories -- `github.com/P-N-Suganthan/CEC2014`,
  `github.com/P-N-Suganthan/CEC2017-BoundContrained`, and
  `github.com/P-N-Suganthan/2022-SO-BO` (Liang, Qu, Suganthan and
  coauthors).
- **License status:** none. The GitHub API reports `license: null` for all
  three repositories, and none carries a `LICENSE`/`LICENSE.md`/`LICENSE.txt`
  file at its root (checked directly). There is consequently no license
  grant, but also no restriction asserted by the organizers.
- **Attribution given:** vendored per an established community norm, not a
  license -- IOHexperimenter (BSD-3-Clause) ships the byte-identical CEC
  2022 data files under its own license with no separate grant; the
  CEC2014/CEC2017 repositories' own readmes advertise third-party CRAN
  packages built on this data; the CRAN packages `cecs` and `cec2013`
  (both since removed for unrelated technical reasons) redistributed this
  same family of data with only their own R authors as copyright holders.
  Full per-suite source URLs, hashes and report-vs-C divergence rulings are
  documented in each module's own doc comments and in `README.md`.

## 7. TSPLIB -- benchmark instances

- **Used:** three `EUC_2D` instances (`berlin52`, `eil51`, `st70`) and
  their published `.opt.tour` optimal tours, vendored verbatim.
- **Where:** `crates/problems/data/tsplib/`; reader in
  `crates/problems/src/tsp.rs`.
- **Source:** Gerhard Reinelt's TSPLIB95, fetched via two independent
  GitHub mirrors (`github.com/mastqe/tsplib`,
  `github.com/pdrozdowski/TSPLib.Net`), confirmed byte-identical to each
  other.
- **License status:** none. The canonical Heidelberg distribution site
  (`comopt.ifi.uni-heidelberg.de/software/TSPLIB95/`, checked directly)
  carries no copyright notice, license statement, terms of use, or
  redistribution policy of any kind -- it does not describe the data as
  "public domain" or as anything else. The data is nonetheless universally
  redistributed and parsed by TSP-adjacent software without a license
  grant from Reinelt (e.g. CRAN's `TSP` package, PyPI's `tsplib95`).
- **Attribution given:** vendored on the same community-norm basis as the
  CEC data, with prominent attribution in `crates/problems/src/tsp.rs`'s
  module documentation rather than the data being withheld.
