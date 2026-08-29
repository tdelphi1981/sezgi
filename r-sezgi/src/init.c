
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

SEXP savvy_sz_preset_de_best_1__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_de_best_1__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_preset_de_rand_1__impl(SEXP c_arg__pop_size, SEXP c_arg__budget) {
    SEXP res = savvy_sz_preset_de_rand_1__ffi(c_arg__pop_size, c_arg__budget);
    return handle_result(res);
}

SEXP savvy_sz_solve_bbob__impl(SEXP c_arg__spec_json, SEXP c_arg__fid, SEXP c_arg__dim, SEXP c_arg__instance, SEXP c_arg__master_seed, SEXP c_arg__run_id) {
    SEXP res = savvy_sz_solve_bbob__ffi(c_arg__spec_json, c_arg__fid, c_arg__dim, c_arg__instance, c_arg__master_seed, c_arg__run_id);
    return handle_result(res);
}


static const R_CallMethodDef CallEntries[] = {
    {"savvy_sezgi_version__impl", (DL_FUNC) &savvy_sezgi_version__impl, 0},
    {"savvy_sz_preset_de_best_1__impl", (DL_FUNC) &savvy_sz_preset_de_best_1__impl, 2},
    {"savvy_sz_preset_de_rand_1__impl", (DL_FUNC) &savvy_sz_preset_de_rand_1__impl, 2},
    {"savvy_sz_solve_bbob__impl", (DL_FUNC) &savvy_sz_solve_bbob__impl, 6},
    {NULL, NULL, 0}
};

void R_init_sezgi(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);

    // Functions for initialization, if any.

}
