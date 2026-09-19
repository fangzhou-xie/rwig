
# utility functions for WDL

euclidean <- function(embedding) {
  dist_mat <- euclidean_cpp(embedding)
  colnames(dist_mat) <- rownames(embedding)
  rownames(dist_mat) <- rownames(embedding)
  dist_mat
}

doc2dist <- function(doc_tokens, dict) {
  docmat <- doc2dist_cpp(doc_tokens, dict)
  rownames(docmat) <- dict
  docmat
}

# Resolve `method = "auto"` for sinkhorn()/barycenter(): when the Gibbs kernel
# exp(-C / reg) gets smaller than `threshold` anywhere, the plain algorithm
# underflows, so switch to the log-stabilized one; otherwise use `plain`.
resolve_method <- function(control, C, plain) {
  if (control$method != "auto") {
    return(control)
  }
  kmin <- min(exp(-min(C) / control$reg), exp(-max(C) / control$reg))
  control$method <- if (kmin < control$threshold) "log" else plain
  if (control$verbose) {
    message(sprintf("`method` is automatically switched to \"%s\"", control$method))
  }
  control
}
