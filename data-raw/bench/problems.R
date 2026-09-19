# Shared problem generator for golden.R / compare.R.
# Every case is a closure so that RNG state is deterministic per case.

rand_prob <- function(n) { x <- stats::runif(n); x / sum(x) }
rand_prob_zeros <- function(n, nz) { x <- stats::runif(n); x[sample(n, nz)] <- 0; x / sum(x) }
rand_cost <- function(M, N, symmetric = FALSE) {
  if (symmetric) {
    E <- matrix(stats::rnorm(M * 5), M, 5)
    as.matrix(stats::dist(E))
  } else {
    matrix(stats::runif(M * N), M, N)
  }
}

make_cases <- function(cuda = FALSE, skip_crash = TRUE) {
  cases <- list()
  add <- function(name, fn) cases[[name]] <<- fn

  # ---- sinkhorn ----
  for (sz in list(c(50, 40, FALSE), c(300, 300, TRUE))) {
    M <- sz[1]; N <- sz[2]; symm <- as.logical(sz[3])
    for (method in c("vanilla", "log")) for (wg in c(FALSE, TRUE)) for (nt in c(0L, 4L)) {
      if (method == "vanilla" && nt > 0) next
      for (zeros in c(FALSE, TRUE)) {
        nm <- sprintf("sinkhorn_%s_M%d_N%d_grad%d_nt%d_zeros%d", method, M, N, wg, nt, zeros)
        add(nm, local({
          M <- M; N <- N; symm <- symm; method <- method; wg <- wg; nt <- nt; zeros <- zeros
          function() {
            set.seed(1)
            a <- if (zeros) rand_prob_zeros(M, 5) else rand_prob(M)
            b <- if (zeros) rand_prob_zeros(N, 3) else rand_prob(N)
            C <- rand_cost(M, N, symm)
            reg <- if (method == "log") 0.05 else 0.5
            rwig::sinkhorn(a, b, C, list(reg = reg, method = method, with_grad = wg,
                                         n_threads = nt, use_cuda = FALSE, max_iter = 300L, verbose = 0L))
          }
        }))
      }
    }
  }

  # ---- barycenter ----
  for (sz in list(c(40, 60, 3), c(200, 200, 4), c(60, 40, 3))) {
    M <- sz[1]; N <- sz[2]; S <- sz[3]
    for (method in c("parallel", "log")) for (wg in c(FALSE, TRUE)) for (nt in c(0L, 4L)) {
      if (method == "parallel" && nt > 0) next
      # baseline build aborts (std::out_of_range) for N > M, log, threaded
      if (skip_crash && method == "log" && nt > 0 && N > M) next
      nm <- sprintf("barycenter_%s_M%d_N%d_S%d_grad%d_nt%d", method, M, N, S, wg, nt)
      add(nm, local({
        M <- M; N <- N; S <- S; method <- method; wg <- wg; nt <- nt
        function() {
          set.seed(2)
          A <- matrix(stats::runif(M * S), M, S); A <- sweep(A, 2, colSums(A), "/")
          C <- rand_cost(M, N, M == N)
          w <- rand_prob(S)
          b_ext <- if (wg) rand_prob(N) else NULL
          reg <- if (method == "log") 0.05 else 0.5
          rwig::barycenter(A, C, w, b_ext, list(reg = reg, method = method, with_grad = wg,
                                                 n_threads = nt, use_cuda = FALSE, max_iter = 300L, verbose = 0L))
        }
      }))
    }
  }

  # ---- wdl (direct C++ entry, CPU) ----
  for (mode in c(1L, 2L)) {
    add(sprintf("wdl_mode%d", mode), local({ mode <- mode; function() {
      set.seed(3)
      N <- 60; M <- 40; S <- 3
      Y <- matrix(stats::runif(N * M), N, M); Y <- sweep(Y, 2, colSums(Y), "/")
      C <- rand_cost(N, N, TRUE)
      set.seed(4)
      rwig:::wdl_cpp(Y, C, 0.5, S, 0L, 16L, 2L, mode, FALSE, 50L, 1e-6, 2L,
                     .005, .01, .9, .999, 1e-8, FALSE, 42L)
    }}))
  }

  # ---- utilities ----
  for (fs in c("auto", "sklearn", "none")) {
    add(paste0("tsvd_", fs), local({ fs <- fs; function() {
      set.seed(5); Mx <- matrix(stats::rnorm(80 * 12), 80, 12)
      rwig::tsvd(Mx, k = 4, flip_sign = fs)
    }}))
  }
  add("euclidean", function() {
    set.seed(6); E <- matrix(stats::rnorm(300 * 10), 300, 10)
    E[7, ] <- E[3, ] # exact duplicate row -> distance 0
    rwig:::euclidean_cpp(E)
  })
  add("doc2dist", function() {
    dict <- c("alpha", "beta", "gamma", "delta", "</s>")
    docs <- list(c("alpha", "beta", "beta", "zzz_unknown"), c("gamma", "gamma", "delta"), c("delta"))
    rwig:::doc2dist_cpp(docs, dict)
  })

  # ---- CUDA variants ----
  if (cuda) {
    add("cuda_sinkhorn_vanilla_grad1", function() {
      set.seed(1); M <- 300; N <- 300
      rwig::sinkhorn(rand_prob(M), rand_prob(N), rand_cost(M, N, TRUE),
                     list(reg = 0.5, method = "vanilla", with_grad = TRUE, use_cuda = TRUE, max_iter = 300L, verbose = 0L))
    })
    add("cuda_sinkhorn_vanilla_grad0", function() {
      set.seed(1); M <- 300; N <- 300
      rwig::sinkhorn(rand_prob(M), rand_prob(N), rand_cost(M, N, TRUE),
                     list(reg = 0.5, method = "vanilla", with_grad = FALSE, use_cuda = TRUE, max_iter = 300L, verbose = 0L))
    })
    for (wg in c(TRUE, FALSE)) {
      add(sprintf("cuda_barycenter_parallel_grad%d", wg), local({ wg <- wg; function() {
        set.seed(2); M <- 200; N <- 200; S <- 4
        A <- matrix(stats::runif(M * S), M, S); A <- sweep(A, 2, colSums(A), "/")
        rwig::barycenter(A, rand_cost(M, N, TRUE), rand_prob(S), if (wg) rand_prob(N) else NULL,
                         list(reg = 0.5, method = "parallel", with_grad = wg, use_cuda = TRUE, max_iter = 300L, verbose = 0L))
      }}))
    }
  }
  cases
}
