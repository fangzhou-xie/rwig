
# test arma usage in package


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

sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)


cpp11::cpp_register()
devtools::document()


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
  c(.2, .2, .5, .1), c(.3, .4, .2, .1), c(.5, .4, .05, .05)
)
C <- cbind(c(1,2,3,4), c(2,3,4,5), c(3,4,5,6), c(4,5,6,7))





cpp11::cpp_register()
devtools::document()


wdl_cpp(A, C, .1, 5, 2, 1, 0, .1,1000,1e-6,2,.001,.01,.9,.999,1e-8, 2)





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


m <- wdl(c("this is a sentence", "this is another sentence"),
         specs = wdl_spec(
           word2vec.control = list(min_count = 1),
           sinkhorn.control = list(sinkhorn_mode = "parallel")
           # sinkhorn.control = list(sinkhorn_mode = "log")
         ))
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


wig(headlines_df, Date, headlines,
    specs = wig_spec(
      wdl.control = list(verbose = TRUE)
    ))




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







tictoc::tic()
# sol <- sinkhorn_vanilla(docdist[,1], docdist[,2], distmat, .1, TRUE)
sol <- sinkhorn_vanilla(docdist[,1], docdist[,2], distmat, .1)
tictoc::toc()

tictoc::tic()
sol <- barycenter_parallel(docdist[,1:5], distmat, rep(.2,5), .1)
tictoc::toc()

sol$b |> sort(decreasing = TRUE) |> head(10)

sol$P |> is.na() |> sum()
sum(sol$P == 0)
sum(sol$P != 0)
sum(rowSums(sol$P) - docdist[,1])
sum(colSums(sol$P) - docdist[,2])


C <- distmat[1:4, 1:5] |> unname()

f1 <- function(C, v, eps = 1) {
  K <- exp(-C/eps)
  out <- rep(0, nrow(C))
  for (i in 1:nrow(C)) {
    for (j in 1:ncol(C)) {
      out[i] <- out[i] + C[i,j] * K[i,j] * v[j]
    }
  }
  out
}
f2 <- function(C, v, eps = 1) {
  K <- exp(-C/eps)
  # diagv <- diag(v)
  c((C * K) %*% v)
}
f1(C, rep(1, ncol(C)), .1)
f2(C, rep(1, ncol(C)), .1)

g1 <- function(C, u, eps = 1) {
  K <- exp(-C/eps)
  out <- rep(0, ncol(C))
  for (j in 1:ncol(C)) {
    for (i in 1:nrow(C)) {
      out[j] <- out[j] + u[i] * C[i,j] * K[i,j]
    }
  }
  out
}
g2 <- function(C, u, eps = 1) {
  K <- exp(-C/eps)
  c(u %*% (C * K))
}
g1(C, rep(1, nrow(C)), .1)
g2(C, rep(1, nrow(C)), .1)


C <- distmat[1:4, 1:5] |> unname()
u <- c(1,2,3,4)
v <- c(1,2,3,4,5)



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

a <- docdist[,1]
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



cpp11::cpp_register()
devtools::document()

softmax <- function(A) {
  # each col
  for (j in 1:ncol(A)) {
    expAj <- exp(A[,j] - max(A[,j]))
    A[,j] <- expAj / sum(expAj)
  }
  A
}
set.seed(1)
A <- matrix(rnorm(dim(docdist)[1]*4), ncol = 4) |>
  softmax()
w <- rnorm(4)
w <- exp(w - max(w)) / sum(exp(w - max(w)))



cpp11::cpp_register()
devtools::document()


sol1 <- barycenter_parallel_cpp(A, distmat, w, .1,TRUE,1000,1e-6) # 1.77s
sol2 <- barycenter_log_cpp(A, distmat, w, .1,TRUE,1000,1e-6)

sol1$JbA[1:10,1:5]
sol2$JbA[1:10,1:5]

sol <- barycenter_parallel_cpp(A, distmat, w, .1,FALSE,1000,1e-6)
sol <- barycenter_log_cpp(A, distmat, w, .1,FALSE,1000,1e-6)




# improve performance!!!
tictoc::tic()
sol <- barycenter_parallel_cpp(A, distmat, w, .1,TRUE,1000,1e-6)
tictoc::toc()

tictoc::tic()
sol <- barycenter_log_cpp(A, distmat, w, .1,TRUE,1000,1e-6)
tictoc::toc()


b1 <- barycenter_parallel_cpp(A, distmat, w, 10,TRUE,1000,1e-6)$b
b2 <- barycenter_parallel_cpp(A, distmat, w, .1,TRUE,1000,1e-6)$b


# plot the histogram of both distributions
library(tidyverse)

bdf <- tibble(b1 = b1, b2 = b2) |>
  pivot_longer(cols = everything(), names_to = "b", values_to = "val")

ggplot(bdf) +
  geom_histogram(aes(x = val, fill = b), alpha = .2)




tictoc::tic()
sol <- barycenter_parallel_cpp(A, distmat, w, .1,FALSE,1000,1e-6)
tictoc::toc()

tictoc::tic()
sol <- barycenter_parallel_withoutjac_cpp(A, distmat, w, .1, 1000, 1e-6,FALSE)
tictoc::toc()

tictoc::tic()
sol <- barycenter_parallel_cpp(A, distmat, w, .1,TRUE,1000,1e-6)
tictoc::toc()

tictoc::tic()
sol <- barycenter_parallel_withjac_cpp(A, distmat, w, .1, 1000, 1e-6,FALSE)
tictoc::toc()


tictoc::tic()
sol <- barycenter_log_cpp(A, distmat, w, .1,FALSE,1000,1e-6)
tictoc::toc()

tictoc::tic()
sol <- barycenter_log_withoutjac_cpp(A, distmat, w, .1, 1000, 1e-6,FALSE)
tictoc::toc()

tictoc::tic()
sol <- barycenter_log_cpp(A, distmat, w, .1,TRUE,1000,1e-6)
tictoc::toc()

tictoc::tic()
sol <- barycenter_log_withjac_cpp(A, distmat, w, .1, 1000, 1e-6,FALSE)
tictoc::toc()





sol <- wdl_cpp(docdist,distmat,1,4,32,2,1,1000,1e-6,2,.001,.01,.9,.999,1e-8,123,TRUE)












a <- docdist[,1] |> unname()
b <- docdist[,2] |> unname()
C <- distmat |> unname()
reg <- 10


asub <- a[a != 0]
bsub <- b[b != 0]
Csub <- C[which(a != 0), which(b != 0)]

sinkhorn_test(a, b, C, reg)
C[c(1,6,10),c(7,10)]

sinkhorn_vanilla_withoutgrad_cpp(
  a[c(1,6,10)], b[c(7,10)], C[c(1,6,10),c(7,10)], reg, 10000, 1e-6)$P

sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)$P[c(1,6,10),c(7,10)]


sinkhorn_vanilla_withoutgrad_cpp(asub, bsub, Csub, reg, 10000, 1e-6)$P
sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)$P[which(a != 0), which(b != 0)]

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


A <- docdist[,1:4] |> unname()
C <- distmat |> unname()
w <- rep(.25, 4)
reg <- 1

cpp11::cpp_register()
devtools::document()


wdl_legacy_cpp(docdist,distmat,1,4,32,2,1,.1,1000,1e-6,0,.001,.01,.9,.999,1e-8,123,FALSE)$W
wdl_legacy_cpp(docdist,distmat,1,4,32,2,1,.1,1000,1e-6,1,.001,.01,.9,.999,1e-8,123,FALSE)$W
wdl_legacy_cpp(docdist,distmat,1,4,32,2,1,.1,1000,1e-6,2,.001,.01,.9,.999,1e-8,123,FALSE)$W

wdl_cpp(docdist,distmat,1,4,32,2,1,1000,1e-6,0,.001,.01,.9,.999,1e-8,123,FALSE)$W
wdl_cpp(docdist,distmat,1,4,32,2,1,1000,1e-6,1,.001,.01,.9,.999,1e-8,123,FALSE)$W
wdl_cpp(docdist,distmat,1,4,32,2,1,1000,1e-6,2,.001,.01,.9,.999,1e-8,123,FALSE)$W







sol <- wdl_cpp(A, C, 1, 4, 32, 2, 1, 1000,1e-6,0,.001,.01,.9,.999,1e-8,123,TRUE)
sol$A |> colSums()
sol$W |> colSums()
sol$Yhat


bench::mark(
  wdl_legacy_cpp(docdist,distmat,1,4,32,2,1,.1,1000,1e-6,0,.001,.01,.9,.999,1e-8,123,FALSE),
  wdl_cpp(docdist,distmat,1,4,32,2,1,1000,1e-6,0,.001,.01,.9,.999,1e-8,123,FALSE),
  check = FALSE
)

wdl_legacy_cpp(docdist,distmat,1,4,32,2,1,.1,1000,1e-6,0,.001,.01,.9,.999,1e-8,123,TRUE)

sol <- wdl_cpp(docdist,distmat,1,4,32,2,1,1000,1e-6,0,.001,.01,.9,.999,1e-8,123,TRUE)
docdist
sol$Yhat


sol$A |> dim()
sol$W |> dim()
sol$Yhat |> dim()

sol$A[,1] |> sum()
sol$W[,1] |> sum()
sol$Yhat[,1]


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



barycenter_log_withjac_cpp(A, C, w, reg, 1000, 1e-6, FALSE)[c("b","JbA","Jbw","iter","err")]
barycenter_log_cpp(A, C, w, reg, TRUE, 1000, 1e-6)[c("b","JbA","Jbw","iter","err")]

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


(barycenter_parallel_cpp(A, C, w, reg, TRUE, 1000, 1e-6)$JbA - barycenter_parallel_withjac_cpp(A, C, w, reg, 1000, 1e-6, FALSE)$JbA) |> abs() |> sum()









barycenter_parallel_cpp(A, C, w, reg, TRUE, 1000, 1e-6)
barycenter_parallel_cpp(A, C, w, reg, FALSE, 1000, 1e-6)


A <- cbind(c(.3,.2,.1,.0,.4), c(.2,.1,.2,.0,.5))
C <- cbind()











sinkhorn_vanilla_withoutgrad_cpp(a, b, C, reg, 1000, 1e-6)
sinkhorn_vanilla_class(a, b, C, reg, FALSE, 1000, 1e-6)
sinkhorn_log_cpp(a, b, C, reg, FALSE, 1000, 1e-6)

sinkhorn_vanilla_cpp(asub, bsub, Csub, reg, FALSE, 1000, 1e-6)$P
sinkhorn_vanilla_cpp(a, b, C, reg, FALSE, 1000, 1e-6)$P[which(a != 0), which(b != 0)]
sinkhorn_log_cpp(a, b, C, reg, FALSE, 1000, 1e-6)$P[which(a != 0), which(b != 0)]
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

(sinkhorn_vanilla_class(a, b, C, reg, FALSE, 1000, 1e-6)$P - sinkhorn_log_class(a, b, C, reg, FALSE, 1000, 1e-6)$P) |> abs() |> sum()




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
A <- docdist[,1:5] |> unname()
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
    A, nrow(A),
    B, ncol(B), ncol(A)
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
cpp11_matmul(A, B)

A <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)
B <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)

bench::mark(
  A %*% B,
  cpp11_matmul(A, B),
  arma_matmul(A, B)
)


cpp11::cpp_register()
devtools::document()


cpp11_matvec(A, c(1,2))
c(A %*% c(1,2))

bench::mark(
  cpp11_matvec(A, c(1,2)),
  c(A %*% c(1,2))
)


v <- rep(1, 5)
K <- exp(-C/reg)
cpp11_aKv(a, K, v)
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
test3(c(1,2,3))
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


N <- 1000
S <- 20
a <- matrix(rnorm(N*S), nrow = N, ncol = S)
K <- exp(-matrix(rnorm(N*N),N,N)/.1)
v <- matrix(rep(1, N*S), nrow = N, ncol = S)

kernel_aKv(a, K, v)
c((a / (K %*% v)))

kernel_aKv(a, K, v)[1:5,1:5]
(a / (K %*% v))[1:5,1:5]

cpp11::cpp_register()
devtools::document()


bench::mark(
  kernel_aKv(a, K, v),
  arma_aKv(a, K, v),
  a / (K %*% v),
  check = FALSE
)


bench::mark(
  purrr::walk(1:100, ~ kernel_aKv(a, K, v)),
  purrr::walk(1:100, ~ arma_aKv(a, K, v)),
  purrr::walk(1:100, ~ a / (K %*% v)),
  check = FALSE
)
