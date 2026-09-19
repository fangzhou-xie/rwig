library(tinytest)

# CPU vs CUDA parity: only runs where the package was built with CUDA
if (rwig::check_cuda()) {
  tol <- 1e-8

  set.seed(11)
  M <- 40; N <- 30
  a <- runif(M); a <- a / sum(a)
  b <- runif(N); b <- b / sum(b)
  C <- matrix(runif(M * N), M, N)

  for (wg in c(FALSE, TRUE)) {
    ctl <- list(reg = .5, method = "vanilla", with_grad = wg, verbose = 0L, max_iter = 500L)
    cpu <- sinkhorn(a, b, C, c(ctl, use_cuda = FALSE))
    gpu <- sinkhorn(a, b, C, c(ctl, use_cuda = TRUE))
    expect_equal(gpu$P, cpu$P, tolerance = tol, info = paste("P with_grad =", wg))
    expect_equal(gpu$u, cpu$u, tolerance = tol)
    expect_equal(gpu$v, cpu$v, tolerance = tol)
    expect_equal(gpu$loss, cpu$loss, tolerance = tol)
    expect_equal(gpu$iter, cpu$iter)
    if (wg) expect_equal(gpu$grad_a, cpu$grad_a, tolerance = tol)
  }

  S <- 3
  A <- matrix(runif(M * S), M, S); A <- sweep(A, 2, colSums(A), "/")
  w <- runif(S); w <- w / sum(w)
  b_ext <- runif(N); b_ext <- b_ext / sum(b_ext)

  for (wg in c(FALSE, TRUE)) {
    ctl <- list(reg = .5, method = "parallel", with_grad = wg, verbose = 0L, max_iter = 500L)
    cpu <- barycenter(A, C, w, if (wg) b_ext else NULL, c(ctl, use_cuda = FALSE))
    gpu <- barycenter(A, C, w, if (wg) b_ext else NULL, c(ctl, use_cuda = TRUE))
    expect_equal(gpu$b, cpu$b, tolerance = tol, info = paste("b with_grad =", wg))
    expect_equal(gpu$U, cpu$U, tolerance = tol)
    expect_equal(gpu$V, cpu$V, tolerance = tol)
    expect_equal(gpu$iter, cpu$iter)
    if (wg) {
      expect_equal(gpu$grad_A, cpu$grad_A, tolerance = tol)
      expect_equal(gpu$grad_w, cpu$grad_w, tolerance = tol)
      expect_equal(gpu$loss, cpu$loss, tolerance = tol)
    }
  }
} else {
  expect_false(rwig::check_cuda())
}
