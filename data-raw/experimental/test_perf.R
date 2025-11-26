
# use `perf` to test performance?

set.seed(1)
N <- 100
a <- rnorm(N) |> abs()
b <- rnorm(N) |> abs()
a <- a / sum(a)
b <- b / sum(b)
C <- matrix(rnorm(N*N)*10, N, N) |> abs()
reg <- .1


for (i in 1:1000) {
  sinkhorn_vanilla_cpp(a, b, C, reg, TRUE, 1000, 1e-6, 0)$err
}
