library(tinytest)

# tol <- 1e-12
tol <- 10 * sqrt(.Machine$double.eps) # the default precision

# test data
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

############################################################
# 1. Vanilla Sinkhorn
############################################################

# solution is calculated with Zygote AD library in Julia
sol_jl_vanilla <- list(
  loss = -0.08551825585323469,
  P = rbind(
    c(0.15387273448546623, 0.1377352555897001, 0.00839235688471941),
    c(0.20516364598062165, 0.18364700745293355, 0.011189809179625875),
    c(0.00944109726108573, 0.06244459928563187, 0.028114002008261082),
    c(0.009441097261085737, 0.062444599285631855, 0.028114002008261085),
    c(0.02208142501174065, 0.05372853838610261, 0.024189829919132565)
  ),
  u = c(
    0.49812211751311114,
    1.8053817338525748,
    0.6138751754880418,
    0.22583205650756197,
    3.9028258998838967
  ),
  v = c(
    0.8396926041657585,
    2.043140616670284,
    0.3384009425683915
  ),
  grad_a = c(
    -0.06627250204750616,
    0.0624958541223469,
    -0.04537729963641905,
    -0.14537699819139782,
    0.13958838748094693
  )
)

sol_sk_vanilla <- sinkhorn(
  a,
  b,
  C,
  list(
    reg = reg,
    method = "vanilla",
    with_grad = TRUE,
    verbose = 0L
  )
)

# test the solution to the vanilla Sinkhorn
expect_equal(sol_jl_vanilla$loss, sol_sk_vanilla$loss, tolerance = tol)
expect_equal(sol_jl_vanilla$P, sol_sk_vanilla$P, tolerance = tol)
expect_equal(sol_jl_vanilla$u, sol_sk_vanilla$u, tolerance = tol)
expect_equal(sol_jl_vanilla$v, sol_sk_vanilla$v, tolerance = tol)
expect_equal(sol_jl_vanilla$grad_a, sol_sk_vanilla$grad_a, tolerance = tol)


############################################################
# 2. Log Sinkhorn
############################################################

sol_jl_log <- list(
  loss = -0.08551828893965516,
  P = rbind(
    c(0.15387266486398005, 0.13773502478893654, 0.008392325227486723),
    c(0.2051635531519735, 0.18364669971858213, 0.011189766969982296),
    c(0.009441140523239513, 0.06244480904290916, 0.028114037505477415),
    c(0.009441140523239513, 0.06244480904290916, 0.028114037505477415),
    c(0.02208150093756758, 0.05372865740666314, 0.024189832791576143)
  ),
  f = c(
    -0.06969114121716402,
    0.0590770660280141,
    -0.04879600300231282,
    -0.1487960030023128,
    0.13617033753210625
  ),
  g = c(
    -0.017471845795974508,
    0.07144878623468662,
    -0.10835262420004768
  ),
  grad_a = c(
    -0.06627246652209813,
    0.062495747110156215,
    -0.04537730175603455,
    -0.14537728882766066,
    0.13958900170936495
  )
)

# test log serial
sol_sk_log <- sinkhorn(
  a,
  b,
  C,
  list(
    reg = reg,
    method = "log",
    with_grad = TRUE,
    verbose = 0
  )
)

# test the solution to the log Sinkhorn
expect_equal(sol_jl_log$loss, sol_sk_log$loss, tolerance = tol)
expect_equal(sol_jl_log$P, sol_sk_log$P, tolerance = tol)
expect_equal(sol_jl_log$f, sol_sk_log$f, tolerance = tol)
expect_equal(sol_jl_log$g, sol_sk_log$g, tolerance = tol)
expect_equal(sol_jl_log$grad_a, sol_sk_log$grad_a, tolerance = tol)

# test log threading
sol_sk_log <- sinkhorn(
  a,
  b,
  C,
  list(
    reg = reg,
    n_threads = 4,
    method = "log",
    with_grad = TRUE,
    verbose = 0L
  )
)

# test the solution to the log Sinkhorn
expect_equal(sol_jl_log$loss, sol_sk_log$loss, tolerance = tol)
expect_equal(sol_jl_log$P, sol_sk_log$P, tolerance = tol)
expect_equal(sol_jl_log$f, sol_sk_log$f, tolerance = tol)
expect_equal(sol_jl_log$g, sol_sk_log$g, tolerance = tol)
expect_equal(sol_jl_log$grad_a, sol_sk_log$grad_a, tolerance = tol)
