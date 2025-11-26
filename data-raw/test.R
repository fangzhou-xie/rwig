# test arma usage in package

# R -d "valgrind --leak-check=full --track-origins=yes" -f test_memory.R

#############################################################
# test WDL
#############################################################

sentences <- c("this is a sentence", "this is another one")
wdl_fit <- wdl(sentences)

wdl_fit
summary(wdl_fit)


wigdf <- data.frame(
  ref_date = as.Date(c("2012-01-01", "2012-02-01")),
  docs = c("this is a sentence", "this is another sentence")
)
wigfit <- wig(wigdf, ref_date, docs)

wigfit |> class()
wigfit |> print()
wigfit |> summary()

dts_vec <- as.Date(c("2012-01-01", "2012-02-01", "2012-03-02", "2012-03-03"))
wig_doc_scores <- c(1, 2, 3, 4)

# dts_vec_chr <- as.character(dts_vec)
dts_vec <- gsub("[[:digit:]]{2}$", "01", as.character(dts_vec))
wig_df <- data.frame(ref_date = dts_vec, WIG = wig_doc_scores)

# base R solution to summarize
wig_df_by <- by(wig_df, wig_df$ref_date, function(df) {
  with(df, data.frame(ref_date = ref_date[[1]], WIG = sum(WIG)))
})
wig_df <- do.call(rbind, wig_df_by)
rownames(wig_df) <- NULL
wig_df

# next: standardize to mean 100 and sd 1

#############################################################
# test SKH
#############################################################

a <- c(.3, .4, .1, .1, .1)
b <- c(.4, .5, .1)
C <- rbind(
  c(.1, .2, .3),
  c(.2, .3, .4),
  c(.4, .3, .2),
  c(.3, .2, .1),
  c(.5, .5, .4)
)
reg <- .1

sinkhorn(
  a,
  b,
  C,
  list(reg = reg, verbose = 1L)
)

sinkhorn(
  a,
  b,
  C,
  list(reg = reg, with_grad = TRUE, method = "auto", verbose = 1)
)
sinkhorn(
  a,
  b,
  C,
  list(reg = reg, with_grad = TRUE, method = "vanilla", verbose = 1)
)
sinkhorn(
  a,
  b,
  C,
  list(reg = reg, with_grad = TRUE, method = "log", verbose = 1)
)

#############################################################
# test BRC
#############################################################

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

sol <- barycenter(A, C, w, barycenter_control = list(reg = reg))

barycenter(
  A,
  C,
  w,
  b,
  list(reg = reg, with_grad = TRUE, method = "auto", verbose = 1)
)
barycenter(
  A,
  C,
  w,
  b,
  list(reg = reg, with_grad = TRUE, method = "parallel", verbose = 1)
)
barycenter(
  A,
  C,
  w,
  b,
  list(reg = reg, with_grad = TRUE, method = "log", verbose = 1)
)


#############################################################
# test WDL
#############################################################

RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::blas_get_num_procs()

# devtools::document()
# library(wig)

tinytest::test_all()


headlines_df <- wig::headlines |>
  dplyr::filter(dplyr::row_number() <= 1000)
headlines_df <- wig::headlines
toks <- tokenizers::tokenize_words(headlines_df$headlines)
model <- word2vec::word2vec(toks, min_count = 5)
emb <- as.matrix(model)
distmat <- wig:::euclidean(emb)
docdist <- wig:::doc2dist(toks, rownames(emb))[, 1:10]

distmat |> dim()
docdist |> dim()

set.seed(1)
sol <- wig:::wdl_cpp(
  docdist,
  distmat,
  .1,
  4, # S
  2, # threads
  5, # batch size
  1,
  1,
  10,
  1e-6,
  0,
  0.001,
  .01,
  .9,
  .999,
  1e-8,
  TRUE
)

Rcpp::evalCpp("Rcpp::message(Rf_mkString(\"lksjdflsj\".c_str()))")

Rcpp::sourceCpp(here::here("src/test_message.cpp"))

test_msg()

#############################################################
# test threading
#############################################################

# Rcpp::compileAttributes()
# devtools::document()

RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::blas_get_num_procs()

Rcpp::sourceCpp(here::here("data-raw/legacy_cpp/test_threading.cpp"))

X <- matrix(rnorm(100000000), nrow = 10000, ncol = 10000)
y <- rnorm(10000)

# test_threading(X, y)
# test_serial(X, y)

bench::mark(
  test_threading(X, y),
  test_threading2(X, y, 10),
  # test_serial(X, y)
)

# microbenchmark::microbenchmark({
#   for (i in 1:1000) {}
# })

#############################################################
# test adjoint sinkhorn
#############################################################

RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::blas_get_num_procs()


a <- c(.3, .4, .1, .1, .1)
b <- c(.4, .5, .1)
C <- rbind(
  c(.1, .2, .3),
  c(.2, .3, .4),
  c(.4, .3, .2),
  c(.3, .2, .1),
  c(.5, .5, .4)
)
reg <- .1

sinkhorn(
  a,
  b,
  C,
  list(reg = reg, with_grad = TRUE, method = "auto", verbose = 1)
)
sinkhorn(
  a,
  b,
  C,
  list(reg = reg, with_grad = TRUE, method = "vanilla", verbose = 1)
)
sinkhorn(
  a,
  b,
  C,
  list(reg = reg, with_grad = TRUE, method = "log", verbose = 1)
)


# cpp11::cpp_register()
devtools::document()


sinkhorn_log_cpp(a, b, C, reg, TRUE, 0, 2, 1e-6, 0)$grad_a
sinkhorn_log_cpp(a, b, C, reg, TRUE, 2, 2, 1e-6, 0)$grad_a


sinkhorn_vanilla_cpp(a, b, C, reg, TRUE, 1000, 1e-6, 1)$grad_a
sinkhorn_log_cpp(a, b, C, reg, TRUE, 0, 1000, 1e-6, 0)$grad_a
sinkhorn_log_cpp(a, b, C, reg, TRUE, 2, 1000, 1e-6, 0)$grad_a

tinytest::test_all()


bench::mark(
  sinkhorn_vanilla_cpp(a, b, C, reg, TRUE, 1000, 1e-6, 0)$grad_a,
  sinkhorn_log_cpp(a, b, C, reg, TRUE, 1000, 1e-6, 0)$grad_a,
  check = FALSE
)


set.seed(1)
N <- 4000
a <- rnorm(N) |> abs()
b <- rnorm(N) |> abs()
a <- a / sum(a)
b <- b / sum(b)
C <- matrix(rnorm(N * N) * 10, N, N) |> abs()
reg <- .1


# cpp11::cpp_register()
devtools::document()
# tinytest::test_all()

# NOTE: if recording W and X will make the computation faster?
# currently bugs

sol <- sinkhorn_log_cpp(a, b, C, reg, TRUE, 0, 1000, 1e-6, 0)
sol <- sinkhorn_log_cpp(a, b, C, reg, TRUE, 4, 1000, 1e-6, 0)
sol <- sinkhorn_log_cpp(a, b, C, reg, TRUE, 10, 1000, 1e-6, 0)
sol <- sinkhorn_log_cpp(a, b, C, reg, TRUE, 20, 1000, 1e-6, 0)
sinkhorn_vanilla_cpp(a, b, C, reg, TRUE, 1000, 1e-6, 0)$grad_a[1:20]


bench::mark(
  sinkhorn_vanilla_cpp(a, b, C, reg, TRUE, 1000, 1e-6, 0),
  sinkhorn_log_cpp(a, b, C, reg, 0, FALSE, 1000, 1e-6, 0),
  sinkhorn_log_cpp(a, b, C, reg, 10, FALSE, 1000, 1e-6, 0),
  sinkhorn_log_cpp(a, b, C, reg, 20, FALSE, 1000, 1e-6, 0),
  check = FALSE
)


#############################################################
# test adjoint barycenter
#############################################################

RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::blas_get_num_procs()

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


# cpp11::cpp_register()
devtools::document()
# tinytest::test_all()

barycenter_parallel_cpp(A, C, w, b, reg, TRUE, 1000, 1e-6, 0)$grad_A
barycenter_log_cpp(A, C, w, b, reg, 0, TRUE, 1, 1e-6, 0)$grad_A
barycenter_log_cpp(A, C, w, b, reg, 2, TRUE, 1, 1e-6, 0)$grad_A


softmax <- function(A) {
  # each col
  for (j in 1:ncol(A)) {
    expAj <- exp(A[, j] - max(A[, j]))
    A[, j] <- expAj / sum(expAj)
  }
  A
}

set.seed(1)
M <- 1000
N <- 1000
S <- 4
A <- matrix(rnorm(M * S), M, S) |> softmax()
C <- matrix(rnorm(M * N), M, N) |> abs()
w <- rnorm(S) |> abs()
w <- w / sum(w)
b <- rep(1, N) / N
reg <- .1

cpp11::cpp_register()
devtools::document()
# tinytest::test_all()

sol1 <- barycenter_parallel_cpp(A, C, w, reg, b, TRUE, 1000, 1e-6, 0)
sol2 <- barycenter_log_cpp(A, C, w, reg, b, TRUE, 0, 1000, 1e-6, 0)
sol3 <- barycenter_log_cpp(A, C, w, reg, b, TRUE, 4, 1000, 1e-6, 0)

bench::mark(
  sol1 <- barycenter_parallel_cpp(A, C, w, reg, b, TRUE, 1000, 1e-6, 0),
  sol2 <- barycenter_log_cpp(A, C, w, reg, b, TRUE, 0, 1000, 1e-6, 0),
  sol3 <- barycenter_log_cpp(A, C, w, reg, b, TRUE, 4, 1000, 1e-6, 0),
  sol4 <- barycenter_log_cpp(A, C, w, reg, b, TRUE, 10, 1000, 1e-6, 0),
  sol5 <- barycenter_log_cpp(A, C, w, reg, b, TRUE, 20, 1000, 1e-6, 0),
  check = FALSE
)


Rcpp::cppFunction(
  "
void tt(arma::mat& X) {
  std::cout << sum(X, 0) << std::endl; // sum each col
  std::cout << sum(X, 1) << std::endl; // sum each row
}
",
  depends = "RcppArmadillo"
)

Rcpp::cppFunction(
  "
arma::mat tt(int i, int j) {
  arma::mat X(i, j, arma::fill::zeros);
  return X;
}
",
  depends = "RcppArmadillo"
)
tt(2, 3)

headlines_df <- wig::headlines |>
  dplyr::filter(dplyr::row_number() <= 1000)
toks <- tokenizers::tokenize_words(headlines_df$headlines)
model <- word2vec::word2vec(toks, min_count = 5)
emb <- as.matrix(model)
distmat <- euclidean(emb)
docdist <- doc2dist(toks, rownames(emb))

softmax <- function(A) {
  # each col
  for (j in 1:ncol(A)) {
    expAj <- exp(A[, j] - max(A[, j]))
    A[, j] <- expAj / sum(expAj)
  }
  A
}

# dims
N <- dim(docdist)[1]
S <- 4

set.seed(1)
A <- matrix(rnorm(N * S), ncol = S) |>
  softmax()
C <- distmat
w <- rnorm(S)
w <- exp(w - max(w)) / sum(exp(w - max(w)))
b <- rep(1, N) / N
reg <- .1

cpp11::cpp_register()
devtools::document()
tinytest::test_all()


sol1 <- barycenter_parallel_cpp(A, C, w, b, reg, TRUE, 1000, 1e-6, 0)
sol2 <- barycenter_log_cpp(A, C, w, b, reg, TRUE, 1000, 1e-6, 0)

bench::mark(
  barycenter_parallel_cpp(A, C, w, b, reg, TRUE, 1000, 1e-6, 0),
  # barycenter_log_cpp(A, C, w, b, reg, TRUE, 1000, 1e-6, 0),
  check = FALSE
)


#############################################################
# test vectorization
#############################################################

Rcpp::cppFunction(
  "
arma::mat softmax1(arma::mat& X) {
  arma::mat out{arma::mat(arma::size(X))};
  arma::vec exps{arma::vec(X.n_rows)};
  for (int s = 0; s < X.n_cols; ++s) {
    exps = exp(X.col(s) - X.col(s).max());
    out.col(s) = exps / accu(exps);
  }
  return out;
}
",
  depends = "RcppArmadillo"
)

Rcpp::cppFunction(
  "
arma::mat softmax2(arma::mat& X) {
  //arma::mat out{arma::mat(arma::size(X))};
  arma::rowvec max_vals = arma::max(X, 0);
  arma::mat A = exp(X.each_row() - max_vals);
  A = A.each_row() / sum(A, 0);
  return A;
}
",
  depends = "RcppArmadillo"
)

Rcpp::cppFunction(
  "
arma::mat softmax3(arma::mat& X) {
  //arma::mat out{arma::mat(arma::size(X))};
  arma::rowvec max_vals = arma::max(X, 0);
  arma::mat A = exp(X.each_row() - max_vals);
  A.each_row() /= sum(A, 0);
  return A;
}
",
  depends = "RcppArmadillo"
)

Rcpp::cppFunction(
  "
arma::mat softmax4(arma::mat& X) {
  //arma::mat out{arma::mat(arma::size(X))};
  arma::rowvec max_vals = arma::max(X, 0);
  arma::mat A = exp(X.each_row() - max_vals);
  //A.each_row() /= sum(A, 0);
  A = normalise(A, 1, 0);
  return A;
}
",
  depends = "RcppArmadillo"
)


X <- matrix(rnorm(10000), ncol = 10)
# softmax1(X) |> colSums()

bench::mark(
  softmax1(X),
  softmax2(X),
  softmax3(X),
  softmax4(X),
)

#############################################################
# test BLAS barycenter
#############################################################

softmax <- function(A) {
  # each col
  for (j in 1:ncol(A)) {
    expAj <- exp(A[, j] - max(A[, j]))
    A[, j] <- expAj / sum(expAj)
  }
  A
}

set.seed(1)
M <- 1000
N <- 1000
S <- 4
A <- matrix(rnorm(M * S), M, S) |> softmax()
C <- matrix(rnorm(M * N), M, N) |> abs()
w <- rnorm(S) |> abs()
w <- w / sum(w)
reg <- .1

cpp11::cpp_register()
devtools::document()

sol <- barycenter_parallel_cpp(A, C, w, reg, TRUE, 1, 1e-6, 0)

barycenter_parallel_blas(A, C, w, reg, FALSE, 1, 1e-6, 0)
sol <- barycenter_parallel_cpp(A, C, w, reg, FALSE, 1, 1e-6, 0)

tictoc::tic()
sol <- barycenter_parallel_cpp(A, C, w, reg, TRUE, 1000, 1e-6, 0)
tictoc::toc()


#############################################################
# test BLAS sinkhorn
#############################################################

a <- c(.3, .4, .1, .1, .1)
b <- c(.4, .5, .1)
C <- rbind(
  c(.1, .2, .3),
  c(.2, .3, .4),
  c(.4, .3, .2),
  c(.3, .2, .1),
  c(.5, .5, .4)
)
reg <- .1


set.seed(1)
N <- 1000
a <- rnorm(N) |> abs()
b <- rnorm(N) |> abs()
a <- a / sum(a)
b <- b / sum(b)
C <- matrix(rnorm(N * N) * 10, N, N) |> abs()
reg <- .1
# symm
C[lower.tri(C)] <- t(C)[lower.tri(C)]
# isSymmetric(C)

headlines_df <- wig::headlines |>
  dplyr::filter(dplyr::row_number() <= 1000)
toks <- tokenizers::tokenize_words(headlines_df$headlines)
model <- word2vec::word2vec(toks, min_count = 5)
emb <- as.matrix(model)
distmat <- euclidean(emb)
docdist <- doc2dist(toks, rownames(emb))

docdist |> dim()

a <- docdist[, 1] |> unname()
b <- docdist[, 2] |> unname()
C <- distmat |> unname()
reg <- 10
# isSymmetric(C)

cpp11::cpp_register()
devtools::document()
# tinytest::test_all()

sinkhorn_vanilla_cpp(a, b, C, reg, TRUE, 1000, 1e-6, 0)$err
sinkhorn_log_cpp(a, b, C, reg, TRUE, 1000, 1e-6, 0)$err

# test_dsymv(b, distmat, a, TRUE, 1, 0)
# test_dsymv(b, distmat, a, FALSE, 1, 0)
# c(distmat %*% a)

# TODO: test barycenter

# FIXME: calling `dsymv` is not correct!

sinkhorn_vanilla_blas(a, b, C, reg, FALSE, 1000, 1e-6, 1)$err
sinkhorn_vanilla_cpp(a, b, C, reg, FALSE, 1000, 1e-6, 0)$err

sinkhorn_vanilla_blas(a, b, C, reg, FALSE, 1, 1e-6, 0)
sinkhorn_vanilla_cpp(a, b, C, reg, FALSE, 1, 1e-6, 0)

sinkhorn_vanilla_blas(a, b, C, reg, FALSE, 1000, 1e-6, 0)$err
sinkhorn_vanilla_cpp(a, b, C, reg, FALSE, 1000, 1e-6, 0)$err

bench::mark(
  sinkhorn_vanilla_blas(a, b, C, reg, FALSE, 1000, 1e-6, 0),
  sinkhorn_vanilla_cpp(a, b, C, reg, FALSE, 1000, 1e-6, 0),
  check = FALSE
)

profvis::profvis({
  for (i in 1:10000) {
    sinkhorn_vanilla_blas(a, b, C, reg, FALSE, 1000, 1e-6, 0)
    # sinkhorn_vanilla_cpp(a, b, C, reg, FALSE, 1, 1e-6, 0)
    # test_dgemv(d, D, a, FALSE, 1, 0)
  }
})


diag(d) %*% D %*% diag(a)

test_dtrmm(D, diag(d), T, T, F, 1)
diag(d) %*% D

test_dtrmm(D, diag(a), F, T, F, 1)

D %*% diag(a)

test_dtrmm(
  test_dtrmm(D, diag(d), T, T, F, 1),
  diag(a),
  F,
  T,
  F,
  1
)

diag(d) %*% D %*% diag(a)


bench::mark(
  test_dgemv(d, D, a, FALSE, 1, 0),
  c(D %*% a)
)

bench::mark(test_dsymv(b, A, a, TRUE, 3, .5), c(t(A) %*% (3 * a) + .5 * b))


expect_equal(
  test_dgemm(matrix(d, ncol = 1), D, matrix(a, ncol = 1), F, F, 1, 0),
  D %*% a
)

expect_equal(
  test_dgemm(matrix(a, ncol = 1), D, matrix(d, ncol = 1), T, F, 1, 0),
  t(D) %*% d
)


a <- c(.3, .4, .1, .1, .1)
b <- c(.4, .5, .1)
C <- rbind(
  c(.1, .2, .3, .4, .5),
  c(.2, .3, .4, .3, .2),
  c(.4, .3, .2, .1, .2),
  c(.3, .2, .1, .2, .5),
  c(.5, .5, .4, .0, .2)
)
reg <- .1

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
w <- c(.5, .5)
reg <- 1


devtools::document()


headlines_df <- wig::headlines |>
  dplyr::filter(dplyr::row_number() <= 1000)
toks <- tokenizers::tokenize_words(headlines_df$headlines)
model <- word2vec::word2vec(toks, min_count = 20)
emb <- as.matrix(model)
distmat <- euclidean(emb)
docdist <- doc2dist(toks, rownames(emb))

softmax <- function(A) {
  # each col
  for (j in 1:ncol(A)) {
    expAj <- exp(A[, j] - max(A[, j]))
    A[, j] <- expAj / sum(expAj)
  }
  A
}
set.seed(1)
A <- matrix(rnorm(dim(docdist)[1] * 4), ncol = 4) |>
  softmax()
w <- rnorm(4)
w <- exp(w - max(w)) / sum(exp(w - max(w)))

system.time(
  wdl_cpp(
    A,
    distmat,
    .1,
    5,
    2,
    1,
    1,
    1000,
    1e-6,
    2,
    .001,
    .01,
    .9,
    .999,
    1e-8,
    123,
    0
  )
)

# openBLAS: 3.35s
# BLIS: 1.86

sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)


cpp11::cpp_register()
devtools::document()

sinkhorn_vanilla_cpp(a, b, C, .1, TRUE, 1000, 1e-6, 5)
barycenter_parallel_cpp(A, C, w, .1, TRUE, 1000, 1e-6, 10)
wdl_cpp(A, C, .1, 5, 2, 1, 1, 1000, 1e-6, 2, .001, .01, .9, .999, 1e-8, 123, 2)


# test sinkhorn
sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)
sinkhorn_vanilla_withgrad_cpp(a, b, C, reg, 1000, 1e-6)

sinkhorn_parallel_withoutjac_cpp(A, B, C, reg, 1000, 1e-6)
sinkhorn_parallel_withjac_cpp(A, B, C, reg, 1000, 1e-6)

sinkhorn_log_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)
sinkhorn_log_withgrad_cpp(a, b, C, reg, 1000, 1e-6)


# test barycenter
barycenter_parallel_withoutjac_cpp(A, C, w, reg, 1000, 1e-6)
barycenter_parallel_withjac_cpp(A, C, w, reg, 1000, 1e-6)

barycenter_log_withoutjac_cpp(A, C, w, reg, 1000, 1e-6)
barycenter_log_withjac_cpp(A, C, w, reg, 1000, 1e-6)


# test WDL
A <- cbind(
  c(.2, .2, .5, .1),
  c(.3, .4, .2, .1),
  c(.5, .4, .05, .05)
)
C <- cbind(c(1, 2, 3, 4), c(2, 3, 4, 5), c(3, 4, 5, 6), c(4, 5, 6, 7))


cpp11::cpp_register()
devtools::document()


cpp11::cpp_source(here::here("data-raw", "test_string.cpp"))

findstring("this", tok[[1]])
findstring("is", tok[[1]])
findstring("this is another sentence", sen)


cpp11::cpp_register()
devtools::document()


sen <- c("this is a sentence", "this is another sentence")
tok <- tokenizers::tokenize_words(sen)
model <- word2vec::word2vec(tok, dim = 4, min_count = 1)

emb <- as.matrix(model)

dict <- rownames(emb)
doc2dist(tok, dict)


cpp11::cpp_register()
devtools::document()


m <- wdl(
  c("this is a sentence", "this is another sentence"),
  specs = wdl_spec(
    word2vec.control = list(min_count = 1),
    sinkhorn.control = list(sinkhorn_mode = "parallel")
    # sinkhorn.control = list(sinkhorn_mode = "log")
  )
)
m$topics
m$weights
m$docs_dist
m$docs_pred


# test wig

cpp11::cpp_register()
devtools::document()

dates <- seq.Date(as.Date("2019-01-01"), as.Date("2019-04-05"))
xts::endpoints(dates, "months")


doc_df <- data.frame(
  ref_date = as.Date(c("2018-01-01", "2019-01-02")),
  titles = c("this is a sentence", "this is another sentence")
)

wig(doc_df, ref_date, titles)
wig(doc_df, doc_df$ref_date, doc_df$titles)
sloop::s3_dispatch(wig(doc_df, ref_date, titles))


cpp11::cpp_register()
devtools::document()


# test on the WIG paper dataset
headlines_df <- wig::headlines |>
  dplyr::filter(dplyr::row_number() <= 1000)
headlines_df |> names()


wig(
  headlines_df,
  Date,
  headlines,
  specs = wig_spec(
    wdl.control = list(verbose = TRUE)
  )
)


# library(tidyverse)

cpp11::cpp_register()
devtools::document()

# library(wig)
headlines_df <- wig::headlines |>
  dplyr::filter(dplyr::row_number() <= 1000)
toks <- tokenizers::tokenize_words(headlines_df$headlines)
model <- word2vec::word2vec(toks, min_count = 5)
emb <- as.matrix(model)
distmat <- wig:::euclidean(emb)
docdist <- wig:::doc2dist(toks, rownames(emb))

distmat |> dim()
docdist |> dim()

# distmat |> isSymmetric()
# exp(-distmat/.1) |> isSymmetric()
# K multiplication can use symmetric BLAS?

tictoc::tic()
# sol <- sinkhorn_vanilla(docdist[,1], docdist[,2], distmat, .1, TRUE)
sol <- sinkhorn_vanilla(docdist[, 1], docdist[, 2], distmat, .1)
tictoc::toc()

tictoc::tic()
sol <- barycenter_parallel(docdist[, 1:5], distmat, rep(.2, 5), .1)
tictoc::toc()

sol$b |>
  sort(decreasing = TRUE) |>
  head(10)

sol$P |>
  is.na() |>
  sum()
sum(sol$P == 0)
sum(sol$P != 0)
sum(rowSums(sol$P) - docdist[, 1])
sum(colSums(sol$P) - docdist[, 2])


C <- distmat[1:4, 1:5] |> unname()

f1 <- function(C, v, eps = 1) {
  K <- exp(-C / eps)
  out <- rep(0, nrow(C))
  for (i in 1:nrow(C)) {
    for (j in 1:ncol(C)) {
      out[i] <- out[i] + C[i, j] * K[i, j] * v[j]
    }
  }
  out
}
f2 <- function(C, v, eps = 1) {
  K <- exp(-C / eps)
  # diagv <- diag(v)
  c((C * K) %*% v)
}
f1(C, rep(1, ncol(C)), .1)
f2(C, rep(1, ncol(C)), .1)

g1 <- function(C, u, eps = 1) {
  K <- exp(-C / eps)
  out <- rep(0, ncol(C))
  for (j in 1:ncol(C)) {
    for (i in 1:nrow(C)) {
      out[j] <- out[j] + u[i] * C[i, j] * K[i, j]
    }
  }
  out
}
g2 <- function(C, u, eps = 1) {
  K <- exp(-C / eps)
  c(u %*% (C * K))
}
g1(C, rep(1, nrow(C)), .1)
g2(C, rep(1, nrow(C)), .1)


C <- distmat[1:4, 1:5] |> unname()
u <- c(1, 2, 3, 4)
v <- c(1, 2, 3, 4, 5)


cpp11::cpp_source(here::here("data-raw", "test_speed.cpp"))


library(tidyverse)
mtcars |> distinct(cyl, am)
mtcars |> filter(cyl == 6, am == 1)

mtcars |>
  summarize(n = n_distinct(am)) |>
  bind_cols(
    mtcars |>
      summarize(n2 = n_distinct(cyl))
  )

mtcars |>
  count(cyl == 6, cyl)


# Rcpp::sourceCpp(here::here("data-raw", "test_arma.cpp"))

a <- docdist[, 1]
C <- distmat

f1(a, C, .1)
f2(a, C, .1)

bench::mark(
  f1(a, C, .1),
  f2(a, C, .1)
)


# test sinkhorn wrapped in class

cpp11::cpp_register()
devtools::document()


headlines_df <- wig::headlines |>
  dplyr::filter(dplyr::row_number() <= 1000)
toks <- tokenizers::tokenize_words(headlines_df$headlines)
model <- word2vec::word2vec(toks, min_count = 10)
emb <- as.matrix(model)
distmat <- euclidean(emb)
docdist <- doc2dist(toks, rownames(emb))

docdist |> dim()


sol <- wdl_cpp(
  docdist,
  distmat,
  1,
  4,
  2,
  1,
  1,
  1000,
  1e-6,
  2,
  .001,
  .01,
  .9,
  .999,
  1e-8,
  123,
  2
)
wdl_cpp(A, C, .1, 5, 2, 1, 1, 1000, 1e-6, 2, .001, .01, .9, .999, 1e-8, 123, 2)


cpp11::cpp_register()
devtools::document()

softmax <- function(A) {
  # each col
  for (j in 1:ncol(A)) {
    expAj <- exp(A[, j] - max(A[, j]))
    A[, j] <- expAj / sum(expAj)
  }
  A
}
set.seed(1)
A <- matrix(rnorm(dim(docdist)[1] * 4), ncol = 4) |>
  softmax()
w <- rnorm(4)
w <- exp(w - max(w)) / sum(exp(w - max(w)))


cpp11::cpp_register()
devtools::document()


sol1 <- barycenter_parallel_cpp(A, distmat, w, .1, TRUE, 1000, 1e-6, FALSE) # 1.77s
sol2 <- barycenter_log_cpp(A, distmat, w, .1, TRUE, 1000, 1e-6)

sol1$JbA[1:10, 1:5]
sol2$JbA[1:10, 1:5]

sol <- barycenter_parallel_cpp(A, distmat, w, .1, FALSE, 1000, 1e-6)
sol <- barycenter_log_cpp(A, distmat, w, .1, FALSE, 1000, 1e-6)


# improve performance!!!
tictoc::tic()
sol <- barycenter_parallel_cpp(A, distmat, w, .1, TRUE, 1000, 1e-6, 10)
tictoc::toc()

tictoc::tic()
sol <- barycenter_log_cpp(A, distmat, w, .1, TRUE, 1000, 1e-6)
tictoc::toc()


b1 <- barycenter_parallel_cpp(A, distmat, w, 10, TRUE, 1000, 1e-6)$b
b2 <- barycenter_parallel_cpp(A, distmat, w, .1, TRUE, 1000, 1e-6)$b


# plot the histogram of both distributions
library(tidyverse)

bdf <- tibble(b1 = b1, b2 = b2) |>
  pivot_longer(cols = everything(), names_to = "b", values_to = "val")

ggplot(bdf) +
  geom_histogram(aes(x = val, fill = b), alpha = .2)


tictoc::tic()
sol <- barycenter_parallel_cpp(A, distmat, w, .1, FALSE, 1000, 1e-6)
tictoc::toc()

tictoc::tic()
sol <- barycenter_parallel_withoutjac_cpp(A, distmat, w, .1, 1000, 1e-6, FALSE)
tictoc::toc()

tictoc::tic()
sol <- barycenter_parallel_cpp(A, distmat, w, .1, TRUE, 1000, 1e-6)
tictoc::toc()

tictoc::tic()
sol <- barycenter_parallel_withjac_cpp(A, distmat, w, .1, 1000, 1e-6, FALSE)
tictoc::toc()


tictoc::tic()
sol <- barycenter_log_cpp(A, distmat, w, .1, FALSE, 1000, 1e-6)
tictoc::toc()

tictoc::tic()
sol <- barycenter_log_withoutjac_cpp(A, distmat, w, .1, 1000, 1e-6, FALSE)
tictoc::toc()

tictoc::tic()
sol <- barycenter_log_cpp(A, distmat, w, .1, TRUE, 1000, 1e-6)
tictoc::toc()

tictoc::tic()
sol <- barycenter_log_withjac_cpp(A, distmat, w, .1, 1000, 1e-6, FALSE)
tictoc::toc()


sol <- wdl_cpp(
  docdist,
  distmat,
  1,
  4,
  32,
  2,
  1,
  1000,
  1e-6,
  2,
  .001,
  .01,
  .9,
  .999,
  1e-8,
  123,
  TRUE
)


a <- docdist[, 1] |> unname()
b <- docdist[, 2] |> unname()
C <- distmat |> unname()
reg <- 10


asub <- a[a != 0]
bsub <- b[b != 0]
Csub <- C[which(a != 0), which(b != 0)]

sinkhorn_test(a, b, C, reg)
C[c(1, 6, 10), c(7, 10)]

sinkhorn_vanilla_withoutgrad_cpp(
  a[c(1, 6, 10)],
  b[c(7, 10)],
  C[c(1, 6, 10), c(7, 10)],
  reg,
  10000,
  1e-6
)$P

sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)$P[
  c(1, 6, 10),
  c(7, 10)
]


sinkhorn_vanilla_withoutgrad_cpp(asub, bsub, Csub, reg, 10000, 1e-6)$P
sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)$P[
  which(a != 0),
  which(b != 0)
]

sinkhorn_vanilla_withoutgrad_cpp(asub, bsub, Csub, reg, 10000, 1e-6)$err
sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)$err

bench::mark(
  sinkhorn_vanilla_withoutgrad_cpp(asub, bsub, Csub, reg, 10000, 1e-6),
  sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6),
  check = FALSE
)


# TODO:
# [x]: only use reduced form to compute sinkhorn
# [x]: add log-sinkhorn with reduced form
# [ ]: barycenter with reduced form

A <- docdist[, 1:4] |> unname()
C <- distmat |> unname()
w <- rep(.25, 4)
reg <- 1

wdl_cpp(
  docdist,
  distmat,
  1,
  4,
  32,
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
  123,
  FALSE
)

cpp11::cpp_register()
devtools::document()


wdl_legacy_cpp(
  docdist,
  distmat,
  1,
  4,
  32,
  2,
  1,
  .1,
  1000,
  1e-6,
  0,
  .001,
  .01,
  .9,
  .999,
  1e-8,
  123,
  FALSE
)$W
wdl_legacy_cpp(
  docdist,
  distmat,
  1,
  4,
  32,
  2,
  1,
  .1,
  1000,
  1e-6,
  1,
  .001,
  .01,
  .9,
  .999,
  1e-8,
  123,
  FALSE
)$W
wdl_legacy_cpp(
  docdist,
  distmat,
  1,
  4,
  32,
  2,
  1,
  .1,
  1000,
  1e-6,
  2,
  .001,
  .01,
  .9,
  .999,
  1e-8,
  123,
  FALSE
)$W

wdl_cpp(
  docdist,
  distmat,
  1,
  4,
  32,
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
  123,
  FALSE
)$W
wdl_cpp(
  docdist,
  distmat,
  1,
  4,
  32,
  2,
  1,
  1000,
  1e-6,
  1,
  .001,
  .01,
  .9,
  .999,
  1e-8,
  123,
  FALSE
)$W
wdl_cpp(
  docdist,
  distmat,
  1,
  4,
  32,
  2,
  1,
  1000,
  1e-6,
  2,
  .001,
  .01,
  .9,
  .999,
  1e-8,
  123,
  FALSE
)$W


sol <- wdl_cpp(
  A,
  C,
  1,
  4,
  32,
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
  123,
  TRUE
)
sol$A |> colSums()
sol$W |> colSums()
sol$Yhat


bench::mark(
  wdl_legacy_cpp(
    docdist,
    distmat,
    1,
    4,
    32,
    2,
    1,
    .1,
    1000,
    1e-6,
    0,
    .001,
    .01,
    .9,
    .999,
    1e-8,
    123,
    FALSE
  ),
  wdl_cpp(
    docdist,
    distmat,
    1,
    4,
    32,
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
    123,
    FALSE
  ),
  check = FALSE
)

wdl_legacy_cpp(
  docdist,
  distmat,
  1,
  4,
  32,
  2,
  1,
  .1,
  1000,
  1e-6,
  0,
  .001,
  .01,
  .9,
  .999,
  1e-8,
  123,
  TRUE
)

sol <- wdl_cpp(
  docdist,
  distmat,
  1,
  4,
  32,
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
  123,
  TRUE
)
docdist
sol$Yhat


sol$A |> dim()
sol$W |> dim()
sol$Yhat |> dim()

sol$A[, 1] |> sum()
sol$W[, 1] |> sum()
sol$Yhat[, 1]


bench::mark(
  barycenter_parallel_cpp(A, C, w, reg, FALSE, 1000, 1e-6),
  barycenter_log_cpp(A, C, w, reg, FALSE, 1000, 1e-6),
  check = FALSE
)
bench::mark(
  barycenter_parallel_cpp(A, C, w, reg, TRUE, 1000, 1e-6),
  barycenter_log_cpp(A, C, w, reg, TRUE, 1000, 1e-6),
  check = FALSE
)


barycenter_parallel_cpp(A, C, w, reg, FALSE, 1000, 1e-6)
barycenter_log_cpp(A, C, w, reg, FALSE, 1000, 1e-6)


barycenter_parallel_cpp(A, C, w, 2, TRUE, 1000, 1e-6)
barycenter_log_cpp(A, C, w, 2, TRUE, 1000, 1e-6)


barycenter_log_withjac_cpp(A, C, w, reg, 1000, 1e-6, FALSE)
barycenter_log_cpp(A, C, w, reg, TRUE, 1000, 1e-6)


barycenter_log_withjac_cpp(A, C, w, reg, 1000, 1e-6, FALSE)[c(
  "b",
  "JbA",
  "Jbw",
  "iter",
  "err"
)]
barycenter_log_cpp(A, C, w, reg, TRUE, 1000, 1e-6)[c(
  "b",
  "JbA",
  "Jbw",
  "iter",
  "err"
)]

barycenter_log_cpp(A, C, w, reg, FALSE, 1, 1e-6)
barycenter_parallel_cpp(A, C, w, reg, FALSE, 1000, 1e-6)


barycenter_parallel_withoutjac_cpp(A, C, w, reg, 1000, 1e-6, FALSE)
barycenter_parallel_cpp(A, C, w, reg, FALSE, 1000, 1e-6)


bench::mark(
  barycenter_log_withjac_cpp(A, C, w, reg, 1000, 1e-6, FALSE),
  barycenter_log_cpp(A, C, w, reg, TRUE, 1000, 1e-6),
  check = FALSE
)


bench::mark(
  barycenter_parallel_cpp(A, C, w, reg, FALSE, 1000, 1e-6),
  barycenter_parallel_cpp(A, C, w, reg, TRUE, 1000, 1e-6),
  barycenter_parallel_withjac_cpp(A, C, w, reg, 1000, 1e-6, FALSE),
  check = FALSE
)

bench::mark(
  barycenter_parallel_cpp(A, C, w, reg, FALSE, 1000, 1e-6),
  barycenter_parallel_withoutjac_cpp(A, C, w, reg, 1000, 1e-6, FALSE),
  check = FALSE
)

bench::mark(
  barycenter_parallel_cpp(A, C, w, reg, TRUE, 1000, 1e-6),
  barycenter_parallel_withjac_cpp(A, C, w, reg, 1000, 1e-6, FALSE),
  check = FALSE
)


(barycenter_parallel_cpp(A, C, w, reg, TRUE, 1000, 1e-6)$JbA -
  barycenter_parallel_withjac_cpp(A, C, w, reg, 1000, 1e-6, FALSE)$JbA) |>
  abs() |>
  sum()


barycenter_parallel_cpp(A, C, w, reg, TRUE, 1000, 1e-6)
barycenter_parallel_cpp(A, C, w, reg, FALSE, 1000, 1e-6)


A <- cbind(c(.3, .2, .1, .0, .4), c(.2, .1, .2, .0, .5))
C <- cbind()


sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)
sinkhorn_vanilla_class(a, b, C, reg, FALSE, 1000, 1e-6)
sinkhorn_log_cpp(a, b, C, reg, FALSE, 1000, 1e-6)

sinkhorn_vanilla_cpp(asub, bsub, Csub, reg, FALSE, 1000, 1e-6)$P
sinkhorn_vanilla_cpp(a, b, C, reg, FALSE, 1000, 1e-6)$P[
  which(a != 0),
  which(b != 0)
]
sinkhorn_log_cpp(a, b, C, reg, FALSE, 1000, 1e-6)$P[
  which(a != 0),
  which(b != 0)
]
sinkhorn_log_cpp(asub, bsub, Csub, reg, FALSE, 1000, 1e-6)$P


sinkhorn_vanilla_class(a, b, C, reg, TRUE, 1000, 1e-6)
sinkhorn_log_class(a, b, C, reg, TRUE, 1000, 1e-6)

bench::mark(
  sinkhorn_vanilla_class(a, b, C, reg, FALSE, 1000, 1e-6),
  sinkhorn_vanilla_class(a, b, C, reg, TRUE, 1000, 1e-6),
  sinkhorn_log_class(a, b, C, reg, FALSE, 1000, 1e-6),
  sinkhorn_log_class(a, b, C, reg, TRUE, 1000, 1e-6),
  check = FALSE
)


sinkhorn_vanilla_class(a, b, C, reg, FALSE, 1000, 1e-6)
sinkhorn_vanilla_class(a, b, C, reg, TRUE, 1000, 1e-6)


sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)

sinkhorn_vanilla_withoutgrad_cpp(asub, bsub, Csub, reg, 1000, 1e-6)
sinkhorn_vanilla_class(asub, bsub, Csub, reg, FALSE, 1000, 1e-6)
sinkhorn_vanilla_withgrad_cpp(asub, bsub, Csub, reg, 1000, 1e-6)
sinkhorn_vanilla_class(asub, bsub, Csub, reg, TRUE, 1000, 1e-6)


sinkhorn_vanilla_class(a, b, C, reg, FALSE, FALSE, 1000, 1e-6)
sinkhorn_vanilla_class(a, b, C, reg, FALSE, TRUE, 1000, 1e-6)


sinkhorn_vanilla_class(a, b, C, reg, TRUE, FALSE, 1000, 1e-6)
sinkhorn_vanilla_class(a, b, C, reg, TRUE, TRUE, 1000, 1e-6)

# K <- exp(-C/reg)
# u <- 1:5
# diag(log(u)) * K
# exp(diag())

sinkhorn_vanilla_class(a, b, C, reg, FALSE, FALSE, 1000, 1e-6)$P
sinkhorn_vanilla_class(a, b, C, reg, TRUE, 1000, 1e-6)$grad_a

sinkhorn_vanilla_class(asub, bsub, Csub, reg, FALSE, FALSE, 1000, 1e-6)$P
sinkhorn_vanilla_class(asub, bsub, Csub, reg, FALSE, TRUE, 1000, 1e-6)$P

sinkhorn_vanilla_class(asub, bsub, Csub, reg, TRUE, FALSE, 1000, 1e-6)$grad_a


sinkhorn_vanilla_withoutgrad_cpp(asub, bsub, Csub, reg, 1000, 1e-6)$P
sol <- sinkhorn_vanilla_class(a, b, C, reg, FALSE, 1000, 1e-6)
sol$P[which(a != 0), which(b != 0)]

sinkhorn_log_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)
sol <- sinkhorn_log_class(a, b, C, reg, FALSE, 1000, 1e-6)

sol$err

sinkhorn_vanilla_class(a, b, C, reg, FALSE, 1000, 1e-6)$P
sinkhorn_log_class(a, b, C, reg, FALSE, 1000, 1e-6)$P

(sinkhorn_vanilla_class(a, b, C, reg, FALSE, 1000, 1e-6)$P -
  sinkhorn_log_class(a, b, C, reg, FALSE, 1000, 1e-6)$P) |>
  abs() |>
  sum()


sinkhorn_vanilla_withgrad_cpp(a, b, C, reg, 1000, 1e-6)$Ju |> sum()
sol <- sinkhorn_vanilla_class(a, b, C, reg, TRUE, 1000, 1e-6)

sol$u |> min()
sol$u |> max()
colSums(sol$P)
rowSums(sol$P)
sum(sol$P)

sol |> names()

bench::mark(
  sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6),
  sinkhorn_vanilla_class(a, b, C, reg, FALSE, 1000, 1e-6)
)

bench::mark(
  sinkhorn_vanilla_withgrad_cpp(a, b, C, reg, 1000, 1e-6),
  sinkhorn_vanilla_class(a, b, C, reg, TRUE, 1000, 1e-6)
)

sinkhorn_test(C)


docdist |> dim()
A <- docdist[, 1:5] |> unname()
C <- distmat |> unname()
reg <- 10
barycenter_parallel_withoutjac_cpp(A, C, rep(.2, 5), reg, 1000, 1e-6, FALSE)
barycenter_parallel_withjac_cpp(A, C, rep(.2, 5), reg, 1000, 1e-6, FALSE)

bench::mark(
  barycenter_parallel_withoutjac_cpp(A, C, rep(.2, 5), reg, 1000, 1e-6, FALSE),
  barycenter_parallel_withjac_cpp(A, C, rep(.2, 5), reg, 1000, 1e-6, FALSE),
  check = FALSE
)


# try BLAS
# Reference: https://medium.com/@shiansu/calling-blas-in-r-the-hard-way-da90e0206d99

# R CMD SHLIB blas.c
dyn.load(here::here("src", "blas.so"))
mat_mult_blas <- function(A, B) {
  .Call(
    "mat_mult_blas",
    A,
    nrow(A),
    B,
    ncol(B),
    ncol(A)
  )
}

cpp11::cpp_register()
devtools::document()

A <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)
B <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)

bench::mark(
  mat_mult_blas(A, B),
  A %*% B,
  matmul_arma(A, B)
)


# try Eigen
cpp11::cpp_register()
devtools::document()


A <- matrix(rnorm(4), nrow = 2, ncol = 2)
B <- matrix(rnorm(4), nrow = 2, ncol = 2)

A %*% B
kernel_matmul(A, B)
arma_matmul(A, B)

A <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)
B <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)

bench::mark(
  A %*% B,
  kernel_matmul(A, B),
  arma_matmul(A, B)
)


cpp11::cpp_register()
devtools::document()


cpp11_matvec(A, c(1, 2))
c(A %*% c(1, 2))

bench::mark(
  cpp11_matvec(A, c(1, 2)),
  c(A %*% c(1, 2))
)


v <- rep(1, 5)
K <- exp(-C / reg)
kernel_aKv(a, K, v)
arma_aKv(a, K, v)

a <- rnorm(1000)
K <- matrix(rnorm(1000000), 1000, 1000)
v <- rep(1, 1000)

bench::mark(
  kernel_aKv(a, K, v),
  cpp11_aKv(a, K, v),
  arma_aKv(a, K, v)
)


cpp11::cpp_function(
  "
int test() {
  double arr[] = {1.5, 2.7, 3.9, 4.1, 5.3};
  double* ptr = arr;
  int size = 5;

  // Loop using pointer arithmetic
  for (int i = 0; i < size; i++) {
      // std::cout << *(ptr + i) << \" \";
      cpp11::message(\"%f\", *(ptr + i));
  }
  return(0);
}
"
)
test()


cpp11::cpp_register()
devtools::document()


cpp11::cpp_function(
  "
writable::doubles_matrix<> test1() {
  using namespace cpp11;
  writable::r_vector<double> vec;  // Start empty

  // Dynamically add data
  vec.push_back(1.0);
  vec.push_back(2.0);
  vec.push_back(3.0);
  vec.push_back(4.0);
  //vec.push_back(5.0);
  //vec.push_back(6.0);

  // Now decide dimensions (2 rows by 3 columns)
  vec.attr(R_DimSymbol) = writable::r_vector<int>({2, 2});

  // Create matrix from the vector
  writable::doubles_matrix<> mat(vec.data());

  return(mat);
}
"
)

cpp11::cpp_function(
  "
writable::r_vector<double> test2() {
  using namespace cpp11;
  writable::r_vector<double> vec;  // Start empty

  vec.resize(6);

  // Dynamically add data
  vec[0] = 1.0;
  vec[1] = 2.0;
  vec[2] = 3.0;
  vec[3] = 4.0;
  vec[4] = 5.0;
  vec[5] = 6.0;

  // Now decide dimensions (2 rows by 3 columns)
  //int m = 2;
  //int n = 3;
  //vec.attr(R_DimSymbol) = writable::r_vector<int>({m, n});
  //int * dims = INTEGER(vec.attr(R_DimSymbol)); // array of ints
  //m = dims[0];
  //n = dims[1];
  //message(\"dim: %d by %d\", m, n);
  // message(\"%s\", vec.attr(R_DimSymbol));
  SEXP dims = Rf_getAttrib(vec.data(), R_DimSymbol);
  if (dims == R_NilValue) {
    message(\"1\");
  } else {
    message(\"0\");
  }

  // vec.attr(\"lsjdf\") = 12232;
  // message(\"lsjdf: %d\", *INTEGER(vec.attr(\"lsjdf\")));
  // message(\"lsjdf: %d\", INTEGER(vec.attr(\"lsjdf\"))[1]);
  // INTEGER() either de-reference, or just the first element?

  return(vec);
}
"
)
test2()


cpp11::cpp_function(
  "
writable::r_vector<double> test3(const r_vector<double>& x) {
  return(x);
}
"
)
test1() |> class()
test2()
test3(c(1, 2, 3))
bench::mark(
  test1(),
  test2()
)


cpp11::cpp_register()
devtools::document()

a <- rnorm(10000)
b <- rnorm(10000)

bench::mark(
  test_hadamard_prod(a, b),
  a * b
)

bench::mark(
  test_hadamard_div(a, b),
  a / b
)

test_hadamard_prod(a, b)
test_hadamard_div(a, b)

a * b

arma_hadamard_prod(a, b)
arma_hadamard_div(a, b)


N <- 10000
S <- 20
a <- matrix(rnorm(N * S), nrow = N, ncol = S)
K <- exp(-matrix(rnorm(N * N), N, N) / .1)
v <- matrix(rep(1, N * S), nrow = N, ncol = S)

kernel_aKv(a, K, v)
c((a / (K %*% v)))

kernel_aKv(a, K, v)[1:5, 1:5]
blas_aKv(a, K, v)[1:5, 1:5]
(a / (K %*% v))[1:5, 1:5]

blas_aKv(a, K, v) |> dim()
blas_aKv(a, K, v) |> length()

cpp11::cpp_register()
devtools::document()


bench::mark(
  kernel_aKv(a, K, v),
  blas_aKv(a, K, v),
  arma_aKv(a, K, v),
  a / (K %*% v)
)


bench::mark(
  purrr::walk(1:100, ~ kernel_aKv(a, K, v)),
  purrr::walk(1:100, ~ blas_aKv(a, K, v)),
  purrr::walk(1:100, ~ arma_aKv(a, K, v)),
  purrr::walk(1:100, ~ a / (K %*% v)),
  check = FALSE
)

cpp11::cpp_function(
  "
doubles test(const doubles& a) {
  auto issym = cpp11::package(\"base\")[\"isSymmetric\"];
  //message(\"issym: %d\", (int)issym(a));
  // issym(a) ? message(\"true\") : message(\"false\");
  //message(\"nrow: %i\", Rf_nrows(a));
  //message(\"ncol: %i\", Rf_ncols(a));
  Rf_nrows(a) = 3;

  int M = 2;
  int N = 3;
  auto d{cpp11::r_vector<int>({M,N})};
  return(a);
}
"
)
test(b)
test(matrix(rnorm(12), 3, 4))
test(distmat)
distmat |> class()
c(1, 2) |> class()

cpp11::cpp_function(
  "
#include <vector>
integers test(int M) {
  // integers a;
  integers a1{integers({1,2})};
  integers a2{integers({2,3})};
  //a = a1;
  //a = a2;
  //std::vector<double> s(10);
  //int arr[M];
  //integers a(arr);

  return a;
}
"
)
test(10)


cpp11::cpp_function(
  "
#include <vector>
int test() {
  std::vector<double> s;
  return 0;
}
"
)
test()


cpp11::cpp_function(
  "
doubles test(const doubles& a) {
  writable::doubles r(a.size());
  r.attr(R_DimSymbol) = integers({Rf_nrows(a), Rf_ncols(a)});
  r.attr(R_DimSymbol) = R_NilValue;
  return r;
}
"
)
test(b)
test(a[1:10, 1:3])

cpp11::cpp_function(
  "
doubles test() {
  int x[4] = {1,2,3,4};
  writable::doubles r(std::begin(x), std::end(x));
  return r;
}
"
)
test()

bench::mark(isSymmetric.matrix(distmat))

cpp11::cpp_function(
  "
double test(const doubles& a, int i, int j) {
  return a[i + Rf_nrows(a) * j];
}
"
)

test(a, 3, 2)
a[4, 3]

bench::mark(
  test(a, 3, 2),
  a[4, 3]
)

cpp11::cpp_function(
  "
bool isSymmetric(const doubles& A) {
  // loop the entire matrix to see if A_ij == A_ji
  for (int i = 0; i < Rf_nrows(A); ++i) {
    for (int j = 0; j < Rf_ncols(A); ++j) {
      if (i != j) {
        if (A[i + j * Rf_nrows(A)] != A[j + i * Rf_nrows(A)]) return false;
      }
    }
  }
  return true;
}
"
)
isSymmetric(a)
isSymmetric.matrix(a)
bench::mark(
  isSymmetric(a),
  isSymmetric.matrix(a)
)


cpp11::cpp_register()
devtools::document()

A <- matrix(rnorm(12), 3, 4)

test_init_matrix(3, 2)
t(A)
A
test_tran_matrix(A)

A[2, 3]
test_index_matrix(A, 1, 10)

N <- 1000
S <- 1000
A <- matrix(rnorm(N * S), N, S)
B <- matrix(rnorm(N * S), N, S)

dim(t(A) %*% B)

cpp11_trmatmul(A, B)[1:3, 1:3]
(t(A) %*% B)[1:3, 1:3]

bench::mark(
  cpp11_trmatmul(A, B),
  arma_trmatmul(A, B),
  t(A) %*% B
)

cpp11::cpp_function(
  "
doubles test_write(const doubles& x) {
  safe[REAL](x)[0] = 1;
  return x;
}
"
)
x <- c(2, 3)
test_write(x)

#############################################################
# test BLAS
#############################################################

cpp11::cpp_register()
devtools::document()
tinytest::test_all()


cpp11::cpp_function(
  "
doubles testfunc(const doubles& x, int m, int n) {
  writable::doubles r(m*n);
  r.attr(R_DimSymbol) = {m, n};
  for (int i{0}; i < x.size(); ++i) {
    r[i] = x[i];
  }
  return r;
}
"
)
x <- rnorm(6)
testfunc(x, 2, 3)
matrix(x, 2, 3)

cpp11::cpp_source(here::here("data-raw", "test_print.cpp"))
testprint(exp(-C / reg))


cpp11::cpp_function(
  "
doubles set(const doubles& x, const doubles& y,
                const integers& indx, const integers& indy) {
  int Mx = Rf_nrows(x);
  int My = Rf_nrows(y);
  writable::doubles x_(x);
  for (int i{0}; i < indx.size(); ++i) {
    for (int j{0}; j < indy.size(); ++j) {
      x_[indx[i] + indy[j] * Mx] = y[i + j * My];
    }
  }
  return x_;
}
"
)

A <- matrix(rnorm(12), 3, 4)
a <- matrix(c(1, 2, 3, 4), 2, 2)
set(A, a, c(0L, 1L), c(0L, 1L))


cpp11::cpp_function(
  "
int testfunc(const doubles& x) {
  double * x_ = REAL(x.data());
  for (int i{0}; i < x.size(); i++) {
    message(\"%.3f\", x_[i]);
  }
  return 0;
}
"
)
testfunc(c(1, 2, 3) / 10)

cpp11::cpp_function(
  "
doubles test1(int n, double v) {
  writable::doubles r(n);
  SEXP r_{r.data()};
  for (int i{0}; i < n; ++i) {
    SET_REAL_ELT(r_, i, v);
  }
  return r;
}
"
)
cpp11::cpp_function(
  "
doubles test2(int n, double v) {
  writable::doubles r(n);
  SEXP r_{r.data()};
  for (int i{0}; i < n; ++i) {
    REAL0(r_)[i] = v;
  }
  return r;
}
"
)
cpp11::cpp_function(
  "
doubles test3(int n, double v) {
  writable::doubles r(n);
  double * r_{REAL(r.data())};
  for (int i{0}; i < n; ++i) {
    r_[i] = v;
  }
  return r;
}
"
)
test1(5, 2.5)
test2(5, 2.5)
test3(5, 2.5)
bench::mark(test1(50, 2.5), test2(50, 2.5), test3(50, 2.5))

cpp11::cpp_function(
  "
SEXP testfunc(SEXP a) {
  message(\"%d*%d\", Rf_nrows(a), Rf_ncols(a));
  return a;
}
"
)
testfunc(c(1, 2, 3))

x <- matrix(rnorm(12), 3, 4) |>
  abs() |>
  madness::madness()
C <- matrix(rnorm(12), 3, 4)
reg <- .1

loss <- sum(x * (C + reg * log(x) - reg))

x@dvdx
loss@val
loss@dvdx

Deriv::Deriv(sum(x * (C + reg * log(x) - reg)), x = "x")


Rcpp::sourceCpp(here::here("data-raw", "test_arma_eachcol.cpp"))

m <- matrix(rnorm(1200), 30, 40)
a <- rnorm(30)
b <- rnorm(40)

test1(m, a)
test2(m, a)
diag(a) %*% m

bench::mark(
  test1(m, a),
  test2(m, a),
  diag(a) %*% m
)


library(tidyverse)

dir_df <- fs::dir_info("/home/fangzhou/workspace/", recurse = TRUE)

dir_df |>
  arrange(desc(size)) |>
  select(path, size)
