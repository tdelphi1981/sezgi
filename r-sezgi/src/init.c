
// clang-format sorts includes unless SortIncludes: Never. However, the ordering
// does matter here. So, we need to disable clang-format for safety.

// clang-format off
#include <stdint.h>
#include <Rinternals.h>
#include <R_ext/Parse.h>
// clang-format on

#include "rust/api.h"

static uintptr_t TAGGED_POINTER_MASK = (uintptr_t)1;

SEXP handle_result(SEXP res_) {
    uintptr_t res = (uintptr_t)res_;

    // An error is indicated by tag.
    if ((res & TAGGED_POINTER_MASK) == 1) {
        // Remove tag
        SEXP res_aligned = (SEXP)(res & ~TAGGED_POINTER_MASK);

        // Currently, there are two types of error cases:
        //
        //   1. Error from Rust code
        //   2. Error from R's C API, which is caught by R_UnwindProtect()
        //
        if (TYPEOF(res_aligned) == CHARSXP) {
            // In case 1, the result is an error message that can be passed to
            // Rf_errorcall() directly.
            Rf_errorcall(R_NilValue, "%s", CHAR(res_aligned));
        } else {
            // In case 2, the result is the token to restart the
            // cleanup process on R's side.
            R_ContinueUnwind(res_aligned);
        }
    }

    return (SEXP)res;
}

SEXP savvy_sezgi_version__impl(void) {
    SEXP res = savvy_sezgi_version__ffi();
    return handle_result(res);
}

SEXP savvy_sz_bayesian_plackett_luce_raw__impl(SEXP c_arg__rankings, SEXP c_arg__samples, SEXP c_arg__burn_in, SEXP c_arg__seed) {
    SEXP res = savvy_sz_bayesian_plackett_luce_raw__ffi(c_arg__rankings, c_arg__samples, c_arg__burn_in, c_arg__seed);
    return handle_result(res);
}

SEXP savvy_sz_coco_export__impl(SEXP c_arg__log_root, SEXP c_arg__out_dir) {
    SEXP res = savvy_sz_coco_export__ffi(c_arg__log_root, c_arg__out_dir);
    return handle_result(res);
}

SEXP savvy_sz_ecdf_raw__impl(SEXP c_arg__log_root, SEXP c_arg__per_algo, SEXP c_arg__targets) {
    SEXP res = savvy_sz_ecdf_raw__ffi(c_arg__log_root, c_arg__per_algo, c_arg__targets);
    return handle_result(res);
}

SEXP savvy_sz_per_budget_packages_raw__impl(SEXP c_arg__algo, SEXP c_arg__fid, SEXP c_arg__dim, SEXP c_arg__instance, SEXP c_arg__seed, SEXP c_arg__budget_col, SEXP c_arg__best_f, SEXP c_arg__f_opt, SEXP c_arg__evals, SEXP c_arg__rope, SEXP c_arg__samples, SEXP c_arg__master_seed, SEXP c_arg__aggregate) {
    SEXP res = savvy_sz_per_budget_packages_raw__ffi(c_arg__algo, c_arg__fid, c_arg__dim, c_arg__instance, c_arg__seed, c_arg__budget_col, c_arg__best_f, c_arg__f_opt, c_arg__evals, c_arg__rope, c_arg__samples, c_arg__master_seed, c_arg__aggregate);
    return handle_result(res);
}

SEXP savvy_sz_preset_cmaes__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_cmaes__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_cmaes_ipop__impl(SEXP c_arg__dim, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_cmaes_ipop__ffi(c_arg__dim, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_de_best_1__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_de_best_1__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_de_rand_1__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_de_rand_1__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_es_mu_plus_lambda_raw__impl(SEXP c_arg__pop_size, SEXP c_arg__budget, SEXP c_arg__dist, SEXP c_arg__mean, SEXP c_arg__sigma, SEXP c_arg__loc, SEXP c_arg__scale, SEXP c_arg__alpha, SEXP c_arg__nu) {
    SEXP res = savvy_sz_preset_es_mu_plus_lambda_raw__ffi(c_arg__pop_size, c_arg__budget, c_arg__dist, c_arg__mean, c_arg__sigma, c_arg__loc, c_arg__scale, c_arg__alpha, c_arg__nu);
    return handle_result(res);
}

SEXP savvy_sz_preset_ga_real__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_ga_real__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_gwo__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_gwo__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_harmony_search__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_harmony_search__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_jde__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_jde__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_lshade__impl(SEXP c_arg__dim, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_lshade__ffi(c_arg__dim, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_nelder_mead__impl(SEXP c_arg__dim, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_nelder_mead__ffi(c_arg__dim, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_pso__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_pso__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_random_search__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_random_search__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_sa__impl(SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_sa__ffi(c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_shade__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_shade__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_woa__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_woa__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_read_ioh_records__impl(SEXP c_arg__log_root, SEXP c_arg__budgets) {
    SEXP res = savvy_sz_read_ioh_records__ffi(c_arg__log_root, c_arg__budgets);
    return handle_result(res);
}

SEXP savvy_sz_results_matrix_raw__impl(SEXP c_arg__algo, SEXP c_arg__fid, SEXP c_arg__dim, SEXP c_arg__instance, SEXP c_arg__seed, SEXP c_arg__budget_col, SEXP c_arg__best_f, SEXP c_arg__f_opt, SEXP c_arg__evals, SEXP c_arg__budget, SEXP c_arg__aggregate) {
    SEXP res = savvy_sz_results_matrix_raw__ffi(c_arg__algo, c_arg__fid, c_arg__dim, c_arg__instance, c_arg__seed, c_arg__budget_col, c_arg__best_f, c_arg__f_opt, c_arg__evals, c_arg__budget, c_arg__aggregate);
    return handle_result(res);
}

SEXP savvy_sz_run_experiment_raw__impl(SEXP c_arg__spec_toml, SEXP c_arg__parallel, SEXP c_arg__journal, SEXP c_arg__threads, SEXP c_arg__log_dir) {
    SEXP res = savvy_sz_run_experiment_raw__ffi(c_arg__spec_toml, c_arg__parallel, c_arg__journal, c_arg__threads, c_arg__log_dir);
    return handle_result(res);
}

SEXP savvy_sz_solve_bbob__impl(SEXP c_arg__spec_json, SEXP c_arg__fid, SEXP c_arg__dim, SEXP c_arg__instance, SEXP c_arg__master_seed, SEXP c_arg__run_id) {
    SEXP res = savvy_sz_solve_bbob__ffi(c_arg__spec_json, c_arg__fid, c_arg__dim, c_arg__instance, c_arg__master_seed, c_arg__run_id);
    return handle_result(res);
}

SEXP savvy_sz_stats_bayesian_signed_rank_raw__impl(SEXP c_arg__a, SEXP c_arg__b, SEXP c_arg__rope, SEXP c_arg__samples, SEXP c_arg__seed) {
    SEXP res = savvy_sz_stats_bayesian_signed_rank_raw__ffi(c_arg__a, c_arg__b, c_arg__rope, c_arg__samples, c_arg__seed);
    return handle_result(res);
}

SEXP savvy_sz_stats_cliffs_delta__impl(SEXP c_arg__a, SEXP c_arg__b) {
    SEXP res = savvy_sz_stats_cliffs_delta__ffi(c_arg__a, c_arg__b);
    return handle_result(res);
}

SEXP savvy_sz_stats_cliffs_magnitude__impl(SEXP c_arg__delta) {
    SEXP res = savvy_sz_stats_cliffs_magnitude__ffi(c_arg__delta);
    return handle_result(res);
}

SEXP savvy_sz_stats_friedman__impl(SEXP c_arg__m) {
    SEXP res = savvy_sz_stats_friedman__ffi(c_arg__m);
    return handle_result(res);
}

SEXP savvy_sz_stats_paper_package_raw__impl(SEXP c_arg__algo_names, SEXP c_arg__problem_names, SEXP c_arg__m, SEXP c_arg__rope, SEXP c_arg__samples, SEXP c_arg__seed) {
    SEXP res = savvy_sz_stats_paper_package_raw__ffi(c_arg__algo_names, c_arg__problem_names, c_arg__m, c_arg__rope, c_arg__samples, c_arg__seed);
    return handle_result(res);
}

SEXP savvy_sz_stats_plackett_luce__impl(SEXP c_arg__rankings) {
    SEXP res = savvy_sz_stats_plackett_luce__ffi(c_arg__rankings);
    return handle_result(res);
}

SEXP savvy_sz_stats_wilcoxon__impl(SEXP c_arg__a, SEXP c_arg__b) {
    SEXP res = savvy_sz_stats_wilcoxon__ffi(c_arg__a, c_arg__b);
    return handle_result(res);
}

SEXP savvy_EvalSession_best__impl(SEXP self__) {
    SEXP res = savvy_EvalSession_best__ffi(self__);
    return handle_result(res);
}

SEXP savvy_EvalSession_budget__impl(SEXP self__) {
    SEXP res = savvy_EvalSession_budget__ffi(self__);
    return handle_result(res);
}

SEXP savvy_EvalSession_evals_used__impl(SEXP self__) {
    SEXP res = savvy_EvalSession_evals_used__ffi(self__);
    return handle_result(res);
}

SEXP savvy_EvalSession_evaluate__impl(SEXP self__, SEXP c_arg__x) {
    SEXP res = savvy_EvalSession_evaluate__ffi(self__, c_arg__x);
    return handle_result(res);
}

SEXP savvy_EvalSession_f_opt__impl(SEXP self__) {
    SEXP res = savvy_EvalSession_f_opt__ffi(self__);
    return handle_result(res);
}

SEXP savvy_EvalSession_finish__impl(SEXP self__) {
    SEXP res = savvy_EvalSession_finish__ffi(self__);
    return handle_result(res);
}

SEXP savvy_EvalSession_new__impl(SEXP c_arg__fid, SEXP c_arg__dim, SEXP c_arg__instance, SEXP c_arg__budget, SEXP c_arg__algo_name, SEXP c_arg__seed, SEXP c_arg__log_dir) {
    SEXP res = savvy_EvalSession_new__ffi(c_arg__fid, c_arg__dim, c_arg__instance, c_arg__budget, c_arg__algo_name, c_arg__seed, c_arg__log_dir);
    return handle_result(res);
}


static const R_CallMethodDef CallEntries[] = {
    {"savvy_sezgi_version__impl", (DL_FUNC) &savvy_sezgi_version__impl, 0},
    {"savvy_sz_bayesian_plackett_luce_raw__impl", (DL_FUNC) &savvy_sz_bayesian_plackett_luce_raw__impl, 4},
    {"savvy_sz_coco_export__impl", (DL_FUNC) &savvy_sz_coco_export__impl, 2},
    {"savvy_sz_ecdf_raw__impl", (DL_FUNC) &savvy_sz_ecdf_raw__impl, 3},
    {"savvy_sz_per_budget_packages_raw__impl", (DL_FUNC) &savvy_sz_per_budget_packages_raw__impl, 13},
    {"savvy_sz_preset_cmaes__impl", (DL_FUNC) &savvy_sz_preset_cmaes__impl, 2},
    {"savvy_sz_preset_cmaes_ipop__impl", (DL_FUNC) &savvy_sz_preset_cmaes_ipop__impl, 2},
    {"savvy_sz_preset_de_best_1__impl", (DL_FUNC) &savvy_sz_preset_de_best_1__impl, 2},
    {"savvy_sz_preset_de_rand_1__impl", (DL_FUNC) &savvy_sz_preset_de_rand_1__impl, 2},
    {"savvy_sz_preset_es_mu_plus_lambda_raw__impl", (DL_FUNC) &savvy_sz_preset_es_mu_plus_lambda_raw__impl, 9},
    {"savvy_sz_preset_ga_real__impl", (DL_FUNC) &savvy_sz_preset_ga_real__impl, 2},
    {"savvy_sz_preset_gwo__impl", (DL_FUNC) &savvy_sz_preset_gwo__impl, 2},
    {"savvy_sz_preset_harmony_search__impl", (DL_FUNC) &savvy_sz_preset_harmony_search__impl, 2},
    {"savvy_sz_preset_jde__impl", (DL_FUNC) &savvy_sz_preset_jde__impl, 2},
    {"savvy_sz_preset_lshade__impl", (DL_FUNC) &savvy_sz_preset_lshade__impl, 2},
    {"savvy_sz_preset_nelder_mead__impl", (DL_FUNC) &savvy_sz_preset_nelder_mead__impl, 2},
    {"savvy_sz_preset_pso__impl", (DL_FUNC) &savvy_sz_preset_pso__impl, 2},
    {"savvy_sz_preset_random_search__impl", (DL_FUNC) &savvy_sz_preset_random_search__impl, 2},
    {"savvy_sz_preset_sa__impl", (DL_FUNC) &savvy_sz_preset_sa__impl, 1},
    {"savvy_sz_preset_shade__impl", (DL_FUNC) &savvy_sz_preset_shade__impl, 2},
    {"savvy_sz_preset_woa__impl", (DL_FUNC) &savvy_sz_preset_woa__impl, 2},
    {"savvy_sz_read_ioh_records__impl", (DL_FUNC) &savvy_sz_read_ioh_records__impl, 2},
    {"savvy_sz_results_matrix_raw__impl", (DL_FUNC) &savvy_sz_results_matrix_raw__impl, 11},
    {"savvy_sz_run_experiment_raw__impl", (DL_FUNC) &savvy_sz_run_experiment_raw__impl, 5},
    {"savvy_sz_solve_bbob__impl", (DL_FUNC) &savvy_sz_solve_bbob__impl, 6},
    {"savvy_sz_stats_bayesian_signed_rank_raw__impl", (DL_FUNC) &savvy_sz_stats_bayesian_signed_rank_raw__impl, 5},
    {"savvy_sz_stats_cliffs_delta__impl", (DL_FUNC) &savvy_sz_stats_cliffs_delta__impl, 2},
    {"savvy_sz_stats_cliffs_magnitude__impl", (DL_FUNC) &savvy_sz_stats_cliffs_magnitude__impl, 1},
    {"savvy_sz_stats_friedman__impl", (DL_FUNC) &savvy_sz_stats_friedman__impl, 1},
    {"savvy_sz_stats_paper_package_raw__impl", (DL_FUNC) &savvy_sz_stats_paper_package_raw__impl, 6},
    {"savvy_sz_stats_plackett_luce__impl", (DL_FUNC) &savvy_sz_stats_plackett_luce__impl, 1},
    {"savvy_sz_stats_wilcoxon__impl", (DL_FUNC) &savvy_sz_stats_wilcoxon__impl, 2},
    {"savvy_EvalSession_best__impl", (DL_FUNC) &savvy_EvalSession_best__impl, 1},
    {"savvy_EvalSession_budget__impl", (DL_FUNC) &savvy_EvalSession_budget__impl, 1},
    {"savvy_EvalSession_evals_used__impl", (DL_FUNC) &savvy_EvalSession_evals_used__impl, 1},
    {"savvy_EvalSession_evaluate__impl", (DL_FUNC) &savvy_EvalSession_evaluate__impl, 2},
    {"savvy_EvalSession_f_opt__impl", (DL_FUNC) &savvy_EvalSession_f_opt__impl, 1},
    {"savvy_EvalSession_finish__impl", (DL_FUNC) &savvy_EvalSession_finish__impl, 1},
    {"savvy_EvalSession_new__impl", (DL_FUNC) &savvy_EvalSession_new__impl, 7},
    {NULL, NULL, 0}
};

void R_init_sezgi(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);

    // Functions for initialization, if any.

}
