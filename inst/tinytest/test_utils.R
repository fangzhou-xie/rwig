library(tinytest)

# euclidean distance matrix between embedding rows
set.seed(21)
E <- matrix(rnorm(12 * 4), 12, 4)
E[5, ] <- E[2, ] # exact duplicate row
D <- rwig:::euclidean_cpp(E)
expect_equal(D, as.matrix(dist(E)), tolerance = 1e-12, check.attributes = FALSE)
expect_equal(D[2, 5], 0)
expect_true(isSymmetric(D))

# document-to-distribution: counts normalized per document; unknown tokens
# are counted under the last dictionary entry
dict <- c("a", "b", "c", "</s>")
docs <- list(c("a", "a", "b"), c("c"), c("zzz", "b"))
X <- rwig:::doc2dist_cpp(docs, dict)
expect_equal(dim(X), c(4L, 3L))
expect_equal(colSums(X), rep(1, 3))
expect_equal(X[, 1], c(2 / 3, 1 / 3, 0, 0))
expect_equal(X[, 3], c(0, 1 / 2, 0, 1 / 2))

# truncated SVD reconstructs the leading singular directions
set.seed(22)
Mx <- matrix(rnorm(30 * 6), 30, 6)
sv <- svd(Mx)
for (fs in c("auto", "sklearn", "none")) {
  Tk <- rwig::tsvd(Mx, k = 3, flip_sign = fs)
  expect_equal(dim(Tk), c(30L, 3L))
  # each column equals +/- u_k * s_k
  for (k in 1:3) {
    ref <- sv$u[, k] * sv$d[k]
    expect_true(isTRUE(all.equal(Tk[, k], ref, tolerance = 1e-10)) ||
                  isTRUE(all.equal(Tk[, k], -ref, tolerance = 1e-10)),
                info = paste("tsvd", fs, "column", k))
  }
}
# sklearn convention: the largest-magnitude entry of every column is positive
Tk <- rwig::tsvd(Mx, k = 3, flip_sign = "sklearn")
expect_true(all(apply(Tk, 2, function(col) col[which.max(abs(col))] > 0)))
