# R -d "valgrind --leak-check=full --track-origins=yes" -f test_memory.R

RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::blas_get_num_procs()

# a <- c(.3, .4, .1, .1, .1)
# b <- c(.4, .5, .1)
# C <- rbind(
#   c(.1, .2, .3),
#   c(.2, .3, .4),
#   c(.4, .3, .2),
#   c(.3, .2, .1),
#   c(.5, .5, .4)
# )
# reg <- .1
#
# # devtools::document()

# wig:::sinkhorn_vanilla_cpp(a, b, C, reg, FALSE, 1, 1e-6, 0)
# wig:::sinkhorn_log_cpp(a, b, C, reg, FALSE, 2, 1, 1e-6, 0)

# # wig:::sinkhorn_vanilla_adjoint(a, b, C, reg, TRUE, 1000, 1e-6, 0)$err
#
# # wig:::sinkhorn_vanilla_cpp(a, b, C, reg, TRUE, 1000, 1e-6, 0)$grad_a
# # wig:::sinkhorn_log_cpp(a, b, C, reg, TRUE, 1000, 1e-6, 0)$grad_a
#
# Rcpp::sourceCpp("../src/test_threading.cpp")
#
# X <- matrix(rnorm(30), nrow = 3, ncol = 10)
# y <- rnorm(10)
#
# test_threading(X, y)
# test_serial(X, y)

# devtools::document()

# library(wig)

# headlines_df <- wig::headlines |>
#   dplyr::filter(dplyr::row_number() <= 1000)
# toks <- tokenizers::tokenize_words(headlines_df$headlines)
# model <- word2vec::word2vec(toks, min_count = 5)
# emb <- as.matrix(model)
# distmat <- wig:::euclidean(emb)
# docdist <- wig:::doc2dist(toks, rownames(emb))[, 1:10]

A <- rbind(
  c(.3, .2),
  c(.2, .1),
  c(.1, .2),
  c(.1, .1),
  c(.3, .4)
)
C <- rbind(
  c(.1, .2, .3, .4, .5),
  c(.2, .3, .4, .3, .2),
  c(.4, .3, .2, .1, .2),
  c(.3, .2, .1, .2, .5),
  c(.5, .5, .4, .0, .2)
)
w <- c(.4, .6)
b <- c(.2, .2, .2, .2, .2)
reg <- .1

# wig:::barycenter_parallel_cpp(A, C, w, reg, b, TRUE, 1, 1e-6, 0)
# wig:::barycenter_log_cpp(A, C, w, reg, b, 0, TRUE, 1, 1e-6, 0)
# wig:::barycenter_log_cpp(A, C, w, reg, b, 4, TRUE, 1, 1e-6, 0)

wig:::wdl_cpp(
  A,
  C,
  reg,
  4,
  2, # threads
  4,
  2,
  1,
  1000,
  1e-6,
  0,
  .001,
  .01,
  .9,
  .999,
  1e-8,
  FALSE
)
