#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

extern "C" {
#include "slu_zdefs.h"
}

namespace nb = nanobind;

namespace {

using IndexArray =
    nb::ndarray<const int_t, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ComplexArrayRO = nb::ndarray<const std::complex<double>,
                                   nb::numpy,
                                   nb::ndim<1>,
                                   nb::c_contig,
                                   nb::device::cpu>;
using VecArrayRW = nb::ndarray<std::complex<double>,
                               nb::numpy,
                               nb::ndim<1>,
                               nb::c_contig,
                               nb::device::cpu>;
using MatArrayRW = nb::ndarray<std::complex<double>,
                               nb::numpy,
                               nb::ndim<2>,
                               nb::c_contig,
                               nb::device::cpu>;

constexpr bool kComplexLayoutCompatible =
    sizeof(std::complex<double>) == sizeof(doublecomplex) &&
    alignof(std::complex<double>) == alignof(doublecomplex);

void validate_csc_pattern(size_t n,
                          const IndexArray &colptr,
                          const IndexArray &rowind,
                          size_t nnz_expected) {
    if (colptr.ndim() != 1 || rowind.ndim() != 1) {
        throw std::runtime_error("Expected 1D CSC index arrays.");
    }
    if (colptr.shape(0) != n + 1) {
        throw std::runtime_error("colptr length must be n + 1.");
    }
    if (rowind.shape(0) != nnz_expected) {
        throw std::runtime_error("rowind length mismatch.");
    }
    if (colptr.data()[0] != static_cast<int_t>(0)) {
        throw std::runtime_error("colptr[0] must be 0.");
    }
    for (size_t i = 1; i < n + 1; ++i) {
        if (colptr.data()[i] < colptr.data()[i - 1]) {
            throw std::runtime_error("colptr must be non-decreasing.");
        }
    }
    if (colptr.data()[n] != static_cast<int_t>(nnz_expected)) {
        throw std::runtime_error("colptr[n] must equal nnz.");
    }
    for (size_t i = 0; i < nnz_expected; ++i) {
        const int_t v = rowind.data()[i];
        if (v < static_cast<int_t>(0) || static_cast<size_t>(v) >= n) {
            throw std::runtime_error("rowind entry out of range.");
        }
    }
}

void fill_superlu_values(const ComplexArrayRO &src, std::vector<doublecomplex> &dst) {
    const size_t nnz = src.shape(0);
    dst.resize(nnz);

    if constexpr (kComplexLayoutCompatible) {
        std::memcpy(dst.data(), src.data(), nnz * sizeof(doublecomplex));
        return;
    }

    for (size_t i = 0; i < nnz; ++i) {
        const std::complex<double> c = src.data()[i];
        dst[i].r = c.real();
        dst[i].i = c.imag();
    }
}

class ZFactor {
public:
    explicit ZFactor(size_t n) : n_(n), ready_(false) {
        std::memset(&L_, 0, sizeof(SuperMatrix));
        std::memset(&U_, 0, sizeof(SuperMatrix));
    }

    ~ZFactor() { destroy_lu_if_allocated(); }

    ZFactor(const ZFactor &) = delete;
    ZFactor &operator=(const ZFactor &) = delete;

    void factorize(const IndexArray &colptr,
                   const IndexArray &rowind,
                   const ComplexArrayRO &nzval) {
        if (nzval.ndim() != 1) {
            throw std::runtime_error("nzval must be 1D.");
        }
        validate_csc_pattern(n_, colptr, rowind, nzval.shape(0));

        destroy_lu_if_allocated();

        std::vector<doublecomplex> nzval_buf;
        fill_superlu_values(nzval, nzval_buf);

        std::vector<int> perm_c(n_);
        std::vector<int> perm_r(n_);
        std::vector<int> etree(n_);

        superlu_options_t options;
        set_default_options(&options);

        SuperMatrix A;
        SuperMatrix AC;
        std::memset(&A, 0, sizeof(SuperMatrix));
        std::memset(&AC, 0, sizeof(SuperMatrix));

        zCreate_CompCol_Matrix(&A,
                               static_cast<int>(n_),
                               static_cast<int>(n_),
                               static_cast<int_t>(nzval.shape(0)),
                               nzval_buf.data(),
                               const_cast<int_t *>(rowind.data()),
                               const_cast<int_t *>(colptr.data()),
                               SLU_NC,
                               SLU_Z,
                               SLU_GE);

        get_perm_c(options.ColPerm, &A, perm_c.data());
        sp_preorder(&options, &A, perm_c.data(), etree.data(), &AC);

        SuperLUStat_t stat;
        StatInit(&stat);

        GlobalLU_t glu;
        std::memset(&glu, 0, sizeof(GlobalLU_t));
        int_t info = 0;

        zgstrf(&options,
               &AC,
               sp_ienv(2),
               sp_ienv(1),
               etree.data(),
               nullptr,
               0,
               perm_c.data(),
               perm_r.data(),
               &L_,
               &U_,
               &glu,
               &stat,
               &info);

        StatFree(&stat);
        Destroy_CompCol_Permuted(&AC);
        Destroy_SuperMatrix_Store(&A);

        if (info != 0) {
            destroy_lu_if_allocated();
            throw std::runtime_error("SuperLU factorization failed, info=" +
                                     std::to_string(static_cast<long long>(info)));
        }

        perm_c_ = std::move(perm_c);
        perm_r_ = std::move(perm_r);
        ready_ = true;
    }

    void solve_inplace(const VecArrayRW &rhs, bool transpose = false) const {
        if (!ready_) {
            throw std::runtime_error("Factor is not initialized.");
        }
        if (rhs.shape(0) != n_) {
            throw std::runtime_error("rhs length mismatch.");
        }
        solve_raw(reinterpret_cast<doublecomplex *>(rhs.data()),
                  1,
                  transpose ? TRANS : NOTRANS);
    }

    void solve_inplace(const MatArrayRW &rhs, bool transpose = false) const {
        if (!ready_) {
            throw std::runtime_error("Factor is not initialized.");
        }
        const size_t batch = rhs.shape(0);
        const size_t width = rhs.shape(1);
        if (width != n_) {
            throw std::runtime_error("rhs second dimension must be n.");
        }
        if (batch == 0) {
            return;
        }

        auto *base = reinterpret_cast<doublecomplex *>(rhs.data());
        for (size_t i = 0; i < batch; ++i) {
            solve_raw(base + i * n_, 1, transpose ? TRANS : NOTRANS);
        }
    }

    size_t n() const { return n_; }

private:
    void destroy_lu_if_allocated() {
        if (L_.Store != nullptr) {
            Destroy_SuperNode_Matrix(&L_);
            L_.Store = nullptr;
        }
        if (U_.Store != nullptr) {
            Destroy_CompCol_Matrix(&U_);
            U_.Store = nullptr;
        }
        ready_ = false;
    }

    void solve_raw(doublecomplex *rhs_ptr, int nrhs, trans_t trans) const {
        SuperMatrix B;
        std::memset(&B, 0, sizeof(SuperMatrix));
        zCreate_Dense_Matrix(&B,
                             static_cast<int>(n_),
                             nrhs,
                             rhs_ptr,
                             static_cast<int>(n_),
                             SLU_DN,
                             SLU_Z,
                             SLU_GE);

        SuperLUStat_t stat;
        StatInit(&stat);
        int info = 0;

        zgstrs(trans,
               const_cast<SuperMatrix *>(&L_),
               const_cast<SuperMatrix *>(&U_),
               perm_c_.data(),
               perm_r_.data(),
               &B,
               &stat,
               &info);

        StatFree(&stat);
        Destroy_SuperMatrix_Store(&B);

        if (info != 0) {
            throw std::runtime_error("SuperLU solve failed, info=" + std::to_string(info));
        }
    }

private:
    size_t n_;
    bool ready_;
    std::vector<int> perm_c_;
    std::vector<int> perm_r_;
    SuperMatrix L_;
    SuperMatrix U_;
};

class ZReusableFactor {
public:
    ZReusableFactor(const IndexArray &colptr, const IndexArray &rowind, size_t n)
        : n_(n),
          nnz_(rowind.shape(0)),
          colptr_(n + 1),
          rowind_(nnz_),
          perm_c_(n),
          perm_r_(n),
          etree_(n),
          R_(n),
          C_(n),
          ferr_(1, 1.0),
          berr_(1, 1.0),
          values_(nnz_),
          has_factor_(false) {
        std::memset(&L_, 0, sizeof(SuperMatrix));
        std::memset(&U_, 0, sizeof(SuperMatrix));
        std::memset(&glu_, 0, sizeof(GlobalLU_t));

        validate_csc_pattern(n_, colptr, rowind, nnz_);

        for (size_t i = 0; i < n + 1; ++i) {
            colptr_[i] = colptr.data()[i];
        }
        for (size_t i = 0; i < nnz_; ++i) {
            rowind_[i] = rowind.data()[i];
        }

        set_default_options(&options_);
        options_.Equil = NO;
        options_.IterRefine = NOREFINE;
        options_.PivotGrowth = NO;
        options_.ConditionNumber = NO;
        options_.PrintStat = NO;
    }

    ~ZReusableFactor() { destroy_lu_if_allocated(); }

    ZReusableFactor(const ZReusableFactor &) = delete;
    ZReusableFactor &operator=(const ZReusableFactor &) = delete;

    void refactorize(const ComplexArrayRO &nzval) {
        if (nzval.ndim() != 1 || nzval.shape(0) != nnz_) {
            throw std::runtime_error("nzval length mismatch for sprefactorize.");
        }

        fill_superlu_values(nzval, values_);

        options_.Fact = has_factor_ ? SamePattern_SameRowPerm : DOFACT;

        SuperMatrix A;
        std::memset(&A, 0, sizeof(SuperMatrix));
        zCreate_CompCol_Matrix(&A,
                               static_cast<int>(n_),
                               static_cast<int>(n_),
                               static_cast<int_t>(nnz_),
                               values_.data(),
                               rowind_.data(),
                               colptr_.data(),
                               SLU_NC,
                               SLU_Z,
                               SLU_GE);

        auto *rhsb = doublecomplexMalloc(1);
        auto *rhsx = doublecomplexMalloc(1);
        if (rhsb == nullptr || rhsx == nullptr) {
            if (rhsb != nullptr) {
                SUPERLU_FREE(rhsb);
            }
            if (rhsx != nullptr) {
                SUPERLU_FREE(rhsx);
            }
            Destroy_SuperMatrix_Store(&A);
            throw std::runtime_error("Failed to allocate temporary RHS buffer.");
        }

        SuperMatrix B;
        SuperMatrix X;
        std::memset(&B, 0, sizeof(SuperMatrix));
        std::memset(&X, 0, sizeof(SuperMatrix));
        zCreate_Dense_Matrix(&B,
                             static_cast<int>(n_),
                             0,
                             rhsb,
                             static_cast<int>(n_),
                             SLU_DN,
                             SLU_Z,
                             SLU_GE);
        zCreate_Dense_Matrix(&X,
                             static_cast<int>(n_),
                             0,
                             rhsx,
                             static_cast<int>(n_),
                             SLU_DN,
                             SLU_Z,
                             SLU_GE);

        SuperLUStat_t stat;
        StatInit(&stat);

        mem_usage_t mem_usage;
        double recip_pivot_growth = 0.0;
        double rcond = 0.0;
        int_t info = 0;

        zgssvx(&options_,
               &A,
               perm_c_.data(),
               perm_r_.data(),
               etree_.data(),
               equed_,
               R_.data(),
               C_.data(),
               &L_,
               &U_,
               nullptr,
               0,
               &B,
               &X,
               &recip_pivot_growth,
               &rcond,
               ferr_.data(),
               berr_.data(),
               &glu_,
               &mem_usage,
               &stat,
               &info);

        StatFree(&stat);
        Destroy_SuperMatrix_Store(&A);
        Destroy_SuperMatrix_Store(&B);
        Destroy_SuperMatrix_Store(&X);

        if (info != 0) {
            throw std::runtime_error("SuperLU refactorize failed, info=" +
                                     std::to_string(static_cast<long long>(info)));
        }

        has_factor_ = true;
    }

    void solve_inplace(const VecArrayRW &rhs, bool transpose = false) const {
        if (!has_factor_) {
            throw std::runtime_error("Factor is not initialized. Call sprefactorize first.");
        }
        if (rhs.shape(0) != n_) {
            throw std::runtime_error("rhs length mismatch.");
        }
        solve_raw(reinterpret_cast<doublecomplex *>(rhs.data()),
                  1,
                  transpose ? TRANS : NOTRANS);
    }

    void solve_inplace(const MatArrayRW &rhs, bool transpose = false) const {
        if (!has_factor_) {
            throw std::runtime_error("Factor is not initialized. Call sprefactorize first.");
        }
        const size_t batch = rhs.shape(0);
        const size_t width = rhs.shape(1);
        if (width != n_) {
            throw std::runtime_error("rhs second dimension must be n.");
        }
        if (batch == 0) {
            return;
        }

        auto *base = reinterpret_cast<doublecomplex *>(rhs.data());
        for (size_t i = 0; i < batch; ++i) {
            solve_raw(base + i * n_, 1, transpose ? TRANS : NOTRANS);
        }
    }

    size_t n() const { return n_; }
    size_t nnz() const { return nnz_; }

private:
    void destroy_lu_if_allocated() {
        if (L_.Store != nullptr) {
            Destroy_SuperNode_Matrix(&L_);
            L_.Store = nullptr;
        }
        if (U_.Store != nullptr) {
            Destroy_CompCol_Matrix(&U_);
            U_.Store = nullptr;
        }
        has_factor_ = false;
    }

    void solve_raw(doublecomplex *rhs_ptr, int nrhs, trans_t trans) const {
        SuperMatrix B;
        std::memset(&B, 0, sizeof(SuperMatrix));
        zCreate_Dense_Matrix(&B,
                             static_cast<int>(n_),
                             nrhs,
                             rhs_ptr,
                             static_cast<int>(n_),
                             SLU_DN,
                             SLU_Z,
                             SLU_GE);

        SuperLUStat_t stat;
        StatInit(&stat);
        int info = 0;

        zgstrs(trans,
               const_cast<SuperMatrix *>(&L_),
               const_cast<SuperMatrix *>(&U_),
               perm_c_.data(),
               perm_r_.data(),
               &B,
               &stat,
               &info);

        StatFree(&stat);
        Destroy_SuperMatrix_Store(&B);

        if (info != 0) {
            throw std::runtime_error("SuperLU solve failed, info=" + std::to_string(info));
        }
    }

private:
    size_t n_;
    size_t nnz_;
    std::vector<int_t> colptr_;
    std::vector<int_t> rowind_;

    std::vector<int> perm_c_;
    std::vector<int> perm_r_;
    std::vector<int> etree_;
    std::vector<double> R_;
    std::vector<double> C_;
    std::vector<double> ferr_;
    std::vector<double> berr_;
    std::vector<doublecomplex> values_;

    superlu_options_t options_;
    char equed_[1] = {'N'};

    SuperMatrix L_;
    SuperMatrix U_;
    GlobalLU_t glu_;
    bool has_factor_;
};

VecArrayRW copy_vector(const VecArrayRW &rhs) {
    auto *data = new std::complex<double>[rhs.shape(0)];
    std::memcpy(data, rhs.data(), rhs.shape(0) * sizeof(std::complex<double>));
    nb::capsule owner(data,
                      [](void *p) noexcept {
                          delete[] static_cast<std::complex<double> *>(p);
                      });
    return VecArrayRW(data, {rhs.shape(0)}, owner);
}

MatArrayRW copy_matrix(const MatArrayRW &rhs) {
    const size_t n = rhs.shape(0) * rhs.shape(1);
    auto *data = new std::complex<double>[n];
    std::memcpy(data, rhs.data(), n * sizeof(std::complex<double>));
    nb::capsule owner(data,
                      [](void *p) noexcept {
                          delete[] static_cast<std::complex<double> *>(p);
                      });
    return MatArrayRW(data, {rhs.shape(0), rhs.shape(1)}, owner);
}

} // namespace

NB_MODULE(_superlu_nb, m) {
    m.doc() = "nanobind wrappers for local SuperLU";
    m.attr("index_size_bytes") = nb::int_(sizeof(int_t));
    m.attr("index_is_64bit") = nb::bool_(sizeof(int_t) == 8);

    nb::class_<ZFactor>(m, "ZFactor")
        .def_prop_ro("n", &ZFactor::n)
        .def("solve_inplace",
             nb::overload_cast<const VecArrayRW &, bool>(&ZFactor::solve_inplace, nb::const_),
             nb::arg("rhs"),
             nb::arg("transpose") = false,
             nb::call_guard<nb::gil_scoped_release>())
        .def("solve_inplace",
             nb::overload_cast<const MatArrayRW &, bool>(&ZFactor::solve_inplace, nb::const_),
             nb::arg("rhs"),
             nb::arg("transpose") = false,
             nb::call_guard<nb::gil_scoped_release>());

    nb::class_<ZReusableFactor>(m, "ZReusableFactor")
        .def_prop_ro("n", &ZReusableFactor::n)
        .def_prop_ro("nnz", &ZReusableFactor::nnz)
        .def("refactorize",
             &ZReusableFactor::refactorize,
             nb::arg("nzval"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("solve_inplace",
             nb::overload_cast<const VecArrayRW &, bool>(&ZReusableFactor::solve_inplace,
                                                          nb::const_),
             nb::arg("rhs"),
             nb::arg("transpose") = false,
             nb::call_guard<nb::gil_scoped_release>())
        .def("solve_inplace",
             nb::overload_cast<const MatArrayRW &, bool>(&ZReusableFactor::solve_inplace,
                                                          nb::const_),
             nb::arg("rhs"),
             nb::arg("transpose") = false,
             nb::call_guard<nb::gil_scoped_release>());

    m.def("spfactorize",
          [](const IndexArray &colptr,
             const IndexArray &rowind,
             const ComplexArrayRO &nzval,
             size_t n) {
              auto factor = std::make_unique<ZFactor>(n);
              factor->factorize(colptr, rowind, nzval);
              return factor.release();
          },
          nb::arg("colptr"),
          nb::arg("rowind"),
          nb::arg("nzval"),
          nb::arg("n"),
          nb::rv_policy::take_ownership,
          nb::call_guard<nb::gil_scoped_release>());

    m.def("spanalyze",
          [](const IndexArray &colptr, const IndexArray &rowind, size_t n) {
              auto factor = std::make_unique<ZReusableFactor>(colptr, rowind, n);
              return factor.release();
          },
          nb::arg("colptr"),
          nb::arg("rowind"),
          nb::arg("n"),
          nb::rv_policy::take_ownership,
          nb::call_guard<nb::gil_scoped_release>());

    m.def("sprefactorize",
          [](ZReusableFactor &factor, const ComplexArrayRO &nzval) {
              factor.refactorize(nzval);
          },
          nb::arg("factor"),
          nb::arg("nzval"),
          nb::call_guard<nb::gil_scoped_release>());

    m.def("spsolve",
          [](ZFactor &factor, const VecArrayRW &rhs, bool overwrite_b, bool transpose) {
              VecArrayRW out = overwrite_b ? rhs : copy_vector(rhs);
              factor.solve_inplace(out, transpose);
              return out;
          },
          nb::arg("factor"),
          nb::arg("rhs"),
          nb::arg("overwrite_b") = true,
          nb::arg("transpose") = false,
          nb::call_guard<nb::gil_scoped_release>());

    m.def("spsolve",
          [](ZFactor &factor, const MatArrayRW &rhs, bool overwrite_b, bool transpose) {
              MatArrayRW out = overwrite_b ? rhs : copy_matrix(rhs);
              factor.solve_inplace(out, transpose);
              return out;
          },
          nb::arg("factor"),
          nb::arg("rhs"),
          nb::arg("overwrite_b") = true,
          nb::arg("transpose") = false,
          nb::call_guard<nb::gil_scoped_release>());

    m.def("spsolve",
          [](ZReusableFactor &factor,
             const VecArrayRW &rhs,
             bool overwrite_b,
             bool transpose) {
              VecArrayRW out = overwrite_b ? rhs : copy_vector(rhs);
              factor.solve_inplace(out, transpose);
              return out;
          },
          nb::arg("factor"),
          nb::arg("rhs"),
          nb::arg("overwrite_b") = true,
          nb::arg("transpose") = false,
          nb::call_guard<nb::gil_scoped_release>());

    m.def("spsolve",
          [](ZReusableFactor &factor,
             const MatArrayRW &rhs,
             bool overwrite_b,
             bool transpose) {
              MatArrayRW out = overwrite_b ? rhs : copy_matrix(rhs);
              factor.solve_inplace(out, transpose);
              return out;
          },
          nb::arg("factor"),
          nb::arg("rhs"),
          nb::arg("overwrite_b") = true,
          nb::arg("transpose") = false,
          nb::call_guard<nb::gil_scoped_release>());
}
