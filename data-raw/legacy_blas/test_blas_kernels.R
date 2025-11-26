#
# # test blas kernels
#
# library(tinytest)
#
# a <- c(.1,.2,.3)
# b <- c(.2,.4,.5)
# A <- matrix(c(1,2,3,2,5,6,3,6,5), 3, 3)
#
# B <- matrix(c(1,2,3,2,4,6,3,6,5,1,2,3), 3, 4)
# d <- c(.1,.2,.3,.4)
#
# C <- matrix(c(2,5,9,2,3,6,5,9,3,1,2,3), 3, 4) / 10
# D <- matrix(c(2,5,9,2,3,6,5,9,3,1,2,3), 4, 3) / 10
#
# E <- matrix(c(4,5,6,5,3,1,6,1,2), 3, 3)
#
# ################################################################
# # utilities
# ################################################################
#
# expect_equal(test_vec_at(a, 1), a[2])
# expect_equal(test_mat_at(A, 1, 1), A[2, 2])
# # expect_error(test_vec_at(A, 1))
# # expect_error(test_mat_at(a, 1, 1))
#
# expect_equal(test_diag(a), diag(a))
#
# ################################################################
# # Overloading Operators
# ################################################################
#
# expect_equal(test_div1(a, 2), a / 2)
# expect_equal(test_div2(2, a), 2 / a)
# expect_equal(test_times1(a, 2), a * 2)
# expect_equal(test_times2(2, a), 2 * a)
#
#
# ################################################################
# # Extending BLAS: Hadamard stuff
# ################################################################
#
# # TODO: test the matrix Hadamard prod/div
# # implement power function as well
#
# # expect_equal(test_dhmpd(a, b), a * b)
# # expect_equal(test_dhmdv(a, b), a / b)
# # expect_equal(test_dhmpw(a, b), a ^ b)
#
# expect_equal(test_pow(a, 2), a ^ 2)
# expect_equal(test_exp(a), exp(a))
# expect_equal(test_log(a), log(a))
#
#
# ################################################################
# # Level 1 BLAS: daxpy, ddot, dnrm2, dasum, idamax
# ################################################################
#
# expect_equal(test_daxpy(b, a, 2.5), 2.5 * a + b)
# expect_equal(test_dscal(a, 2), a * 2)
# expect_equal(test_ddot(a, b), sum(a * b))
# expect_equal(test_dnrm2(a), sqrt(sum(a ^ 2)))
# expect_equal(test_dasum(a), sum(abs(a)))
# expect_equal(test_idamax(a), which(abs(a) == max(abs(a))))
#
#
# ################################################################
# # Level 2 BLAS: dgemv, dsymv, dtrmv, dtrsv
# ################################################################
#
# expect_equal(test_dgemv(d, D, a, FALSE, 1, 0), c(D %*% a))
#
# expect_equal(test_dgemv(a, D, d, TRUE, 1, 0), c(t(D) %*% d))
#
# expect_equal(test_dgemv(a, C, d, FALSE, 1, 0), c(C %*% d))
#
#
# # test dgemv
# expect_equal(test_dgemv(b, A, a, FALSE, 1, 0), c(A %*% a))
# expect_equal(test_dgemv(b, A, a, FALSE, 2, 0), c(A %*% (2 * a)))
# expect_equal(test_dgemv(b, A, a, FALSE, 3, .5), c(A %*% (3 * a) + .5 * b))
# expect_equal(test_dgemv(b, A, a, TRUE, 1, 0), c(t(A) %*% a))
# expect_equal(test_dgemv(b, A, a, TRUE, 2, 0), c(t(A) %*% (2 * a)))
# expect_equal(test_dgemv(b, A, a, TRUE, 3, .5), c(t(A) %*% (3 * a) + .5 * b))
#
# expect_equal(test_dgemv(b, B, d, FALSE, 1, 0), c(B %*% d))
# expect_equal(test_dgemv(b, B, d, FALSE, 2, 0), c(B %*% (2*d)))
# expect_equal(test_dgemv(b, B, d, FALSE, 3, .5), c(B %*% (3*d) + .5 * b))
# # expect_error(test_dgemv(b, B, a, TRUE, 1, 0))
# # expect_error(test_dgemv(b, B, a, TRUE, 2, 0))
# # expect_error(test_dgemv(b, B, a, TRUE, 3, .5))
#
# # test dsymv
# expect_equal(test_dsymv(b, A, a, FALSE, 1, 0), c(A %*% a))
# expect_equal(test_dsymv(b, A, a, FALSE, 2, 0), c(A %*% (2 * a)))
# expect_equal(test_dsymv(b, A, a, FALSE, 3, .5), c(A %*% (3 * a) + .5 * b))
# expect_equal(test_dsymv(b, A, a, TRUE, 1, 0), c(t(A) %*% a))
# expect_equal(test_dsymv(b, A, a, TRUE, 2, 0), c(t(A) %*% (2 * a)))
# expect_equal(test_dsymv(b, A, a, TRUE, 3, .5), c(t(A) %*% (3 * a) + .5 * b))
#
# # test dtrmv
# expect_equal(test_dtrmv(b, diag(a), TRUE, FALSE), a * b)
#
# # test dtrmv
# expect_equal(test_dtrsv(b, diag(a), TRUE, FALSE), solve(diag(a), b))
#
#
# ################################################################
# # Level 3 BLAS: dgemm, dsymm, dtrmm, dtrsm
# ################################################################
#
# # test dgemm
#
# expect_equal(test_dgemm(matrix(d, ncol = 1), D, matrix(a, ncol = 1), F, F, 1, 0), D %*% a)
#
# expect_equal(test_dgemm(matrix(a, ncol = 1), D, matrix(d, ncol = 1), T, F, 1, 0), t(D) %*% d)
#
# expect_equal(test_dgemm(C, A, B, FALSE, FALSE, 1, 0), A %*% B)
# expect_equal(test_dgemm(C, A, B, FALSE, FALSE, 2, 0), A %*% (2*B))
# expect_equal(test_dgemm(C, A, B, FALSE, FALSE, 3, 0.5), A %*% (3*B) + .5 * C)
#
# expect_equal(test_dgemm(C, A, B, TRUE, FALSE, 1, 0), t(A) %*% B)
# expect_equal(test_dgemm(C, A, B, TRUE, FALSE, 2, 0), t(A) %*% (2*B))
# expect_equal(test_dgemm(C, A, B, TRUE, FALSE, 3, .5), t(A) %*% (3*B) + .5 * C)
#
# expect_equal(test_dgemm(t(C), D, A, FALSE, TRUE, 1, 0), D %*% t(A))
# expect_equal(test_dgemm(t(C), D, A, FALSE, TRUE, 2, 0), D %*% t(2*A))
# expect_equal(test_dgemm(t(C), D, A, FALSE, TRUE, 3, .5), D %*% t(3*A) + t(.5*C))
#
# expect_equal(test_dgemm(t(C), B, A, TRUE, TRUE, 1, 0), t(B) %*% t(A))
# expect_equal(test_dgemm(t(C), B, A, TRUE, TRUE, 2, 0), t(B) %*% t(2*A))
# expect_equal(test_dgemm(t(C), B, A, TRUE, TRUE, 3, .5), t(B) %*% t(3*A) + t(.5*C))
#
#
# # test dsymm
# expect_equal(test_dsymm(C, A, B, TRUE, TRUE, 1, 0), A %*% B)
# expect_equal(test_dsymm(C, A, B, TRUE, TRUE, 2, 0), A %*% (2*B))
# expect_equal(test_dsymm(C, A, B, TRUE, TRUE, 3, .5), A %*% (3*B) + .5 * C)
#
# expect_equal(test_dsymm(C, A, B, TRUE, FALSE, 1, 0), A %*% B)
# expect_equal(test_dsymm(C, A, B, TRUE, FALSE, 2, 0), A %*% (2*B))
# expect_equal(test_dsymm(C, A, B, TRUE, FALSE, 3, .5), A %*% (3*B) + .5 * C)
#
# expect_equal(test_dsymm(t(C), A, D, FALSE, TRUE, 1, 0), D %*% A)
# expect_equal(test_dsymm(t(C), A, D, FALSE, TRUE, 2, 0), D %*% (2*A))
# expect_equal(test_dsymm(t(C), A, D, FALSE, TRUE, 3, .5), D %*% (3*A) + t(.5*C))
#
# expect_equal(test_dsymm(t(C), A, D, FALSE, FALSE, 1, 0), D %*% A)
# expect_equal(test_dsymm(t(C), A, D, FALSE, FALSE, 2, 0), D %*% (2*A))
# expect_equal(test_dsymm(t(C), A, D, FALSE, FALSE, 3, .5), D %*% (3*A) + t(.5*C))
#
#
# # test dtrmm
# expect_equal(test_dtrmm(B, diag(a), TRUE, TRUE, FALSE, 1), diag(a) %*% B)
# expect_equal(test_dtrmm(B, diag(a), TRUE, TRUE, FALSE, 2), 2 * diag(a) %*% B)
#
# expect_equal(test_dtrmm(B, diag(a), TRUE, FALSE, FALSE, 1), diag(a) %*% B)
# expect_equal(test_dtrmm(B, diag(a), TRUE, FALSE, FALSE, 2), 2 * diag(a) %*% B)
#
# expect_equal(test_dtrmm(D, diag(a), FALSE, TRUE, FALSE, 1), D %*% diag(a))
# expect_equal(test_dtrmm(D, diag(a), FALSE, TRUE, FALSE, 2), 2 * D %*% diag(a))
#
# expect_equal(test_dtrmm(D, diag(a), FALSE, FALSE, FALSE, 1), D %*% diag(a))
# expect_equal(test_dtrmm(D, diag(a), FALSE, FALSE, FALSE, 2), 2 * D %*% diag(a))
#
#
# # test dtrsm
# expect_equal(test_dtrsm(B, diag(a), TRUE, TRUE, FALSE, 1), solve(diag(a), B))
# expect_equal(test_dtrsm(B, diag(a), TRUE, TRUE, FALSE, 2), 2 * solve(diag(a), B))
#
# expect_equal(test_dtrsm(B, diag(a), TRUE, FALSE, FALSE, 1), solve(diag(a), B))
# expect_equal(test_dtrsm(B, diag(a), TRUE, FALSE, FALSE, 2), 2 * solve(diag(a), B))
#
# expect_equal(test_dtrsm(D, diag(a), FALSE, TRUE, FALSE, 1), D %*% solve(diag(a)))
# expect_equal(test_dtrsm(D, diag(a), FALSE, TRUE, FALSE, 2), 2 * D %*% solve(diag(a)))
#
# expect_equal(test_dtrsm(D, diag(a), FALSE, FALSE, FALSE, 1), D %*% solve(diag(a)))
# expect_equal(test_dtrsm(D, diag(a), FALSE, FALSE, FALSE, 2), 2 * D %*% solve(diag(a)))
