library(tinytest)

# tol <- 1e-12
tol <- 10 * sqrt(.Machine$double.eps) # the default precision

# test data
A <- rbind(c(.3, .2), c(.2, .1), c(.1, .2), c(.1, .1), c(.3, .4))
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


############################################################
# 1. Parallel Barycenter
############################################################

# solution from julia Forward diff
sol_jl_parallel <- list(
  grad_A = rbind(
    c(-0.006760869016071031, -0.02391482377435357),
    c(0.004821729585004115, -0.014440148989139237),
    c(0.100480994027538, 0.07311169324284003),
    c(0.018535015085824783, -0.041358625025306815),
    c(0.1655770413887294, 0.21626675329487113)
  ),
  grad_w = c(0.06395620382985355, 0.12898154801524858),
  U = rbind(
    c(0.8267955753556165, 0.24411366823845884),
    c(0.5916985228183498, 0.21586076548039212),
    c(0.11321503783659982, 0.3121854249073409),
    c(0.1464499108805673, 0.1343508875275325),
    c(0.17441203785320222, 0.4368383480936876)
  ),
  V = rbind(
    c(0.5238000347903181, 1.538944595154966),
    c(0.6429279465635749, 1.3424304586397788),
    c(0.957873738180596, 1.029108477221267),
    c(1.554487182609397, 0.7452041012172674),
    c(1.035756385385436, 0.9768508387714551)
  ),
  b = c(
    0.20678519000127618,
    0.10800238756066238,
    0.11915320248239648,
    0.43600814073022814,
    0.13005107922474135
  ),
  loss = 0.07563849817530696
)

# sol_bc_parallel <- barycenter_parallel(A, C, w, reg, b, TRUE, 1000, 1e-6, 0)
sol_bc_parallel <- barycenter(
  A,
  C,
  w,
  b,
  list(reg = reg, method = "parallel", with_grad = TRUE, verbose = 0L)
)

# test the solution to the vanilla Barycenter
expect_equal(sol_jl_parallel$loss, sol_bc_parallel$loss, tolerance = tol)
expect_equal(sol_jl_parallel$P, sol_bc_parallel$P, tolerance = tol)
expect_equal(sol_jl_parallel$U, sol_bc_parallel$U, tolerance = tol)
expect_equal(sol_jl_parallel$V, sol_bc_parallel$V, tolerance = tol)
expect_equal(sol_jl_parallel$grad_A, sol_bc_parallel$grad_A, tolerance = tol)
expect_equal(sol_jl_parallel$grad_w, sol_bc_parallel$grad_w, tolerance = tol)


############################################################
# 2. Log Barycenter
############################################################

sol_jl_log <- list(
  grad_A = rbind(
    c(-0.0067608311611870135, -0.023914840355582733),
    c(0.0048217455691098796, -0.014440192013372316),
    c(0.10048103241863328, 0.07311159632893578),
    c(0.018535128919150035, -0.04135868868746976),
    c(0.1655769481646337, 0.21626684349591252)
  ),
  grad_w = c(0.0639562003946849, 0.12898158562029838),
  F = rbind(
    c(-0.01901955039093642, -0.14101232110784803),
    c(-0.05247578673966771, -0.15331226777843576),
    c(-0.21784675720092742, -0.11641576376822282),
    c(-0.19210715221550018, -0.2007301125164142),
    c(-0.17463368162555232, -0.0828190828260999)
  ),
  G = rbind(
    c(-0.06466472495907682, 0.04310981663938457),
    c(-0.04417241315950124, 0.029448275439667465),
    c(-0.00430399407623952, 0.00286932938415968),
    c(0.0441147053448392, -0.02940980356322616),
    c(0.0035132414136644086, -0.002342160942442967)
  ),
  b = c(
    0.20678515414652007,
    0.10800239007173053,
    0.1191532090133692,
    0.43600815383048386,
    0.13005109293780684
  ),
  loss = 0.0756385004358107
)


# sol_bc_log <- barycenter_log(A, C, w, reg, b, TRUE, 0, 1000, 1e-6, 0)
sol_bc_log <- barycenter(
  A,
  C,
  w,
  b,
  list(reg = reg, with_grad = TRUE, n_threads = 0, method = "log", verbose = 0L)
)

# test the solution to the log Barycenter
expect_equal(sol_jl_log$loss, sol_bc_log$loss, tolerance = tol)
expect_equal(sol_jl_log$P, sol_bc_log$P, tolerance = tol)
expect_equal(sol_jl_log$U, sol_bc_log$U, tolerance = tol)
expect_equal(sol_jl_log$V, sol_bc_log$V, tolerance = tol)
expect_equal(sol_jl_log$grad_A, sol_bc_log$grad_A, tolerance = tol)
expect_equal(sol_jl_log$grad_w, sol_bc_log$grad_w, tolerance = tol)

# sol_bc_log <- barycenter_log(A, C, w, reg, b, TRUE, 4, 1000, 1e-6, 0)
sol_bc_log <- barycenter(
  A,
  C,
  w,
  b,
  list(reg = reg, with_grad = TRUE, n_threads = 4, method = "log", verbose = 0L)
)

# test the solution to the log Barycenter
expect_equal(sol_jl_log$loss, sol_bc_log$loss, tolerance = tol)
expect_equal(sol_jl_log$P, sol_bc_log$P, tolerance = tol)
expect_equal(sol_jl_log$U, sol_bc_log$U, tolerance = tol)
expect_equal(sol_jl_log$V, sol_bc_log$V, tolerance = tol)
expect_equal(sol_jl_log$grad_A, sol_bc_log$grad_A, tolerance = tol)
expect_equal(sol_jl_log$grad_w, sol_bc_log$grad_w, tolerance = tol)
