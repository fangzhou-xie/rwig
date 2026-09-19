# Generate golden outputs with the *reference* build of rwig.
# The golden/ directory is gitignored (~11 MB). Regenerate it from a reference
# build, e.g. `git checkout v0.1.0` installed into a scratch library, before
# running compare.R.
# Usage: Rscript data-raw/bench/golden.R [libpath]
args <- commandArgs(trailingOnly = TRUE)
if (length(args)) .libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(rwig))
source("data-raw/bench/problems.R")
cat("rwig", as.character(packageVersion("rwig")), "from", find.package("rwig"), "cuda:", check_cuda(), "\n")
dir.create("data-raw/bench/golden", showWarnings = FALSE, recursive = TRUE)
cases <- make_cases(cuda = check_cuda())
for (nm in names(cases)) {
  res <- tryCatch(cases[[nm]](), error = function(e) structure(list(error = conditionMessage(e)), class = "golden_error"))
  if (inherits(res, "golden_error")) cat(sprintf("%-55s ERROR: %s\n", nm, res$error)) else {
    it <- if (is.list(res) && !is.null(res$iter)) sprintf("iter=%d err=%.2e", res$iter, res$err) else ""
    cat(sprintf("%-55s ok %s\n", nm, it))
  }
  saveRDS(res, file.path("data-raw/bench/golden", paste0(nm, ".rds")))
}
