# Timing benchmark. Usage: Rscript data-raw/bench/bench.R [label]
args <- commandArgs(trailingOnly = TRUE)
label <- if (length(args)) args[1] else "current"
suppressPackageStartupMessages(library(rwig))
source("data-raw/bench/problems.R")
set.seed(10)
N <- 2000
a <- rand_prob(N); b <- rand_prob(N); C <- rand_cost(N, N, TRUE)
S <- 4; A <- matrix(runif(1000 * S), 1000, S); A <- sweep(A, 2, colSums(A), "/")
Cb <- rand_cost(1000, 1000, TRUE); w <- rand_prob(S); bx <- rand_prob(1000)
Nw <- 500; Mw <- 128; Y <- matrix(runif(Nw * Mw), Nw, Mw); Y <- sweep(Y, 2, colSums(Y), "/")
Cw <- rand_cost(Nw, Nw, TRUE)
E <- matrix(rnorm(5000 * 10), 5000, 10)
dict <- sprintf("w%05d", 1:5000); toks <- replicate(200, sample(dict, 50, TRUE), simplify = FALSE)

ctl <- function(...) modifyList(list(reg = 0.05, max_iter = 100L, zero_tol = 0, use_cuda = FALSE), list(...))
res <- bench::mark(
  sinkhorn_log_nt0   = sinkhorn(a, b, C, ctl(method = "log", with_grad = TRUE, n_threads = 0L)),
  sinkhorn_log_nt8   = sinkhorn(a, b, C, ctl(method = "log", with_grad = TRUE, n_threads = 8L)),
  sinkhorn_vanilla   = sinkhorn(a, b, C, ctl(method = "vanilla", reg = 0.5, with_grad = TRUE)),
  barycenter_log_nt0 = barycenter(A, Cb, w, bx, ctl(method = "log", with_grad = TRUE, n_threads = 0L, max_iter = 50L)),
  barycenter_log_nt8 = barycenter(A, Cb, w, bx, ctl(method = "log", with_grad = TRUE, n_threads = 8L, max_iter = 50L)),
  barycenter_par     = barycenter(A, Cb, w, bx, ctl(method = "parallel", reg = 0.5, with_grad = TRUE)),
  wdl_cpu            = rwig:::wdl_cpp(Y, Cw, 0.5, S, 0L, 32L, 1L, 1L, FALSE, 100L, 1e-6, 2L, .005, .01, .9, .999, 1e-8, FALSE, 42L),
  euclidean          = rwig:::euclidean_cpp(E),
  doc2dist           = rwig:::doc2dist_cpp(toks, dict),
  check = FALSE, iterations = 3, memory = FALSE
)
out <- data.frame(label = label, expr = as.character(res$expression),
                  median_s = round(as.numeric(res$median), 3), mem = "")
print(out, row.names = FALSE)
f <- "data-raw/bench/results.csv"
write.table(out, f, sep = ",", row.names = FALSE, col.names = !file.exists(f), append = file.exists(f))
