# Regression gate: rerun every golden case with the *installed* rwig and compare.
# Usage: Rscript data-raw/bench/compare.R [tolerance]
`%||%` <- function(a, b) if (is.null(a)) b else a
args <- commandArgs(trailingOnly = TRUE)
tol <- if (length(args)) as.numeric(args[1]) else 1e-8
suppressPackageStartupMessages(library(rwig))
source("data-raw/bench/problems.R")
cat("rwig", as.character(packageVersion("rwig")), "from", find.package("rwig"), "cuda:", check_cuda(), "\n")
cases <- make_cases(cuda = check_cuda(), skip_crash = nzchar(Sys.getenv("RWIG_SKIP_CRASH")))
files <- list.files("data-raw/bench/golden", pattern = "\\.rds$", full.names = TRUE)
# cases with no golden of their own (baseline crashed) are compared to the
# serial (nt0) golden: threading must not change the result
extra <- setdiff(names(cases), sub("\\.rds$", "", basename(files)))
alias_name <- function(nm) {
  if (nm == "cuda_barycenter_parallel_grad0") return("barycenter_parallel_M200_N200_S4_grad0_nt0") # baseline GPU output was garbage
  sub("_nt4$", "_nt0", nm)
}
alias <- setNames(file.path("data-raw/bench/golden", paste0(vapply(extra, alias_name, ""), ".rds")), extra)
files <- c(files, alias[file.exists(alias)])
nfail <- 0
for (k in seq_along(files)) {
  f <- files[[k]]
  nm <- if (nzchar(names(files)[k] %||% "")) names(files)[k] else sub("\\.rds$", "", basename(f))
  if (is.null(cases[[nm]])) { cat(sprintf("%-55s SKIP (no case)\n", nm)); next }
  gold <- readRDS(f)
  res <- tryCatch(cases[[nm]](), error = function(e) structure(list(error = conditionMessage(e)), class = "golden_error"))
  if (inherits(gold, "golden_error")) {
    cat(sprintf("%-55s golden was ERROR (%s); now %s\n", nm, gold$error,
                if (inherits(res, "golden_error")) paste("ERROR:", res$error) else "ok"))
    next
  }
  if (inherits(res, "golden_error")) { cat(sprintf("%-55s FAIL: %s\n", nm, res$error)); nfail <- nfail + 1; next }
  cmp <- all.equal(gold, res, tolerance = tol, check.attributes = TRUE)
  if (isTRUE(cmp)) cat(sprintf("%-55s PASS\n", nm)) else {
    cat(sprintf("%-55s FAIL\n  %s\n", nm, paste(cmp, collapse = "\n  "))); nfail <- nfail + 1
  }
}
cat(sprintf("\n%d/%d cases failed\n", nfail, length(files)))
quit(status = if (nfail) 1 else 0)
