# Argument checking and default filling for the control lists.

# Fill missing entries of `args` from `defaults`; entries the user supplied win.
fill_defaults <- function(args, defaults) {
  if (is.null(args)) {
    args <- list()
  }
  utils::modifyList(defaults, args)
}

# The formal default of argument `name` of the calling function, so that the
# documented default list in the signature is the single source of truth.
formal_default <- function(name, fn = sys.function(sys.parent())) {
  eval(formals(fn)[[name]])
}

# Stop if `args` has names outside `allowed`.
check_arg_names <- function(args, allowed, what) {
  bad <- setdiff(names(args), allowed)
  if (length(bad)) {
    stop(
      paste0(bad, collapse = ", "),
      " not matching one of the ", what, " arguments: ",
      paste0("\"", allowed, "\"", collapse = ", ")
    )
  }
  invisible(args)
}

check_wig_args <- function(wig_args) {
  wig_args <- fill_defaults(
    wig_args,
    list(group_unit = "month", svd_method = "topics", standardize = TRUE)
  )
  if (!wig_args$svd_method %in% c("topics", "docs")) {
    stop("`svd_method` must be from: \"topics\" or \"docs\"")
  }
  wig_args
}

check_wdl_args <- function(wdl_args) {
  fill_defaults(
    wdl_args,
    list(num_topics = 4, batch_size = 64, epochs = 2, shuffle = TRUE, seed = 42L)
  )
}

check_tok_args <- function(tok_args) {
  fill_defaults(tok_args, list(stopwords = stopwords::stopwords()))
}

check_w2v_args <- function(w2v_args) {
  fill_defaults(w2v_args, list(dim = 10, min_count = 3, type = "cbow"))
}

check_opt_args <- function(opt_args) {
  opt_args <- fill_defaults(
    opt_args,
    list(
      optimizer = "adamw", lr = .005, decay = .01,
      beta1 = .9, beta2 = .999, eps = 1e-8
    )
  )
  optimizers <- c("sgd", "adam", "adamw")
  if (!opt_args$optimizer %in% optimizers) {
    stop("optimizer must be from: \"sgd\", \"adam\", \"adamw\"")
  }
  # integer code passed to the C++ side
  opt_args$optimizer_int <- match(opt_args$optimizer, optimizers) - 1L
  opt_args
}

# Shared checker for the sinkhorn() / barycenter() control lists.
# `plain` is the non-stabilized method name ("vanilla" or "parallel").
check_ot_args <- function(args, what, plain, defaults) {
  check_arg_names(args, names(defaults), what)
  args <- fill_defaults(args, defaults)

  methods <- c("auto", plain, "log")
  if (!args$method %in% methods) {
    stop("method must be from: ", paste0("\"", methods, "\"", collapse = ", "))
  }
  if (args$verbose) {
    if (args$method == plain && args$n_threads > 0) {
      warning(sprintf("`n_threads` is not used in %s %s!", plain, what))
    }
    if (args$method == "log" && args$n_threads == 0) {
      warning(sprintf(paste0(
        "`n_threads = 0` for log %s might be slow! ",
        "Considering setting `n_threads` for faster computation."
      ), what))
    }
  }
  args
}

ot_defaults <- function(max_iter = 1000L) {
  list(
    reg = .1, with_grad = FALSE, use_cuda = TRUE, n_threads = 0L,
    method = "auto", threshold = .1, max_iter = max_iter, zero_tol = 1e-6,
    verbose = 0L
  )
}

#' @keywords internal
check_sinkhorn_args <- function(skh_args) {
  check_ot_args(skh_args, "sinkhorn", "vanilla", ot_defaults())
}

#' @keywords internal
check_barycenter_args <- function(brc_args, defaults = ot_defaults()) {
  check_ot_args(brc_args, "barycenter", "parallel", defaults)
}
