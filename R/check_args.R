# check args and set the default args in case of missing arguments

check_wig_args <- function(wig_args) {
  # check WIG arguments
  if (is.null(wig_args$group_time)) {
    wig_args$group_time <- "months"
  }
  if (is.null(wig_args$svd_method)) {
    wig_args$svd_method <- "topics"
  }
  if (is.null(wig_args$standardize)) {
    wig_args$standardize <- TRUE
  }

  wig_args
}


check_wdl_args <- function(wdl_args) {
  # check WDL arguments: hyper-parameters
  if (is.null(wdl_args$num_topics)) {
    wdl_args$num_topics <- 4
  }
  if (is.null(wdl_args$batch_size)) {
    wdl_args$batch_size <- 64
  }
  if (is.null(wdl_args$epochs)) {
    wdl_args$epochs <- 2
  }
  if (is.null(wdl_args$n_threads)) {
    wdl_args$n_threads <- 0
  }
  if (is.null(wdl_args$shuffle)) {
    wdl_args$shuffle <- TRUE
  }
  # if (is.null(wdl_args$rng_seed))
  #   wdl_args$rng_seed <- 123L
  if (is.null(wdl_args$verbose)) {
    wdl_args$verbose <- FALSE
  }

  # if `verbose` is NA, abort!
  if (is.na(wdl_args$verbose)) {
    stop("`verbose` argument cannot be NA!")
  }

  # # map verbose into integer
  # if (wdl_args$verbose) {
  #   wdl_args$verbose_int <- 1L
  # } else {
  #   wdl_args$verbose_int <- 0L
  # }

  wdl_args
}

check_tok_args <- function(tok_args) {
  # check tokenizer arguments: m
  # if (is.null(tok_args$stopwords))
  #   tok_args$stopwords <- stopwords::stopwords()

  tok_args
}

check_w2v_args <- function(w2v_args) {
  # check word2vec arguments: must-have: embedding depths
  if (is.null(w2v_args$dim)) {
    # embedding depth: hyper-parameters
    w2v_args$dim <- 10
  }
  if (is.null(w2v_args$min_count)) {
    w2v_args$min_count <- 5
  }
  if (is.null(w2v_args$type)) {
    w2v_args$type <- "cbow"
  }
  # if (is.null(w2v_args$stopwords))
  #   w2v_args$stopwords <- stopwords::stopwords()

  w2v_args
}

# check_skh_args <- function(skh_args) {
#   # check the parameters for Sinkhorn (Barycenter) algorithm
#   if (is.null(skh_args$regularizer)) {
#     skh_args$regularizer <- .1
#   }
#   if (is.null(skh_args$sinkhorn_mode)) {
#     skh_args$sinkhorn_mode <- "auto"
#   }
#   if (is.null(skh_args$sinkhorn_mode_threshold)) {
#     skh_args$sinkhorn_mode_threshold <- .1
#   }
#   if (is.null(skh_args$max_iter)) {
#     skh_args$max_iter <- 1000
#   }
#   if (is.null(skh_args$zero_tol)) {
#     skh_args$zero_tol <- 1e-6
#   }
#
#   # map sinkhorn algo type from character to integer
#   if (skh_args$sinkhorn_mode == "auto") {
#     skh_args$sinkhorn_mode_int <- 0L
#   } else if (skh_args$sinkhorn_mode == "parallel") {
#     skh_args$sinkhorn_mode_int <- 1L
#   } else if (skh_args$sinkhorn_mode == "log") {
#     skh_args$sinkhorn_mode_int <- 2L
#   } else {
#     stop("sinkhorn_mode must be from: \"auto\", \"parallel\", or \"log\"")
#   }
#
#   skh_args
# }

check_opt_args <- function(opt_args) {
  # check the parameters for the optimizer
  if (is.null(opt_args$lr)) {
    opt_args$lr <- .005
  }
  if (is.null(opt_args$decay)) {
    opt_args$decay <- .01
  }
  if (is.null(opt_args$beta1)) {
    opt_args$beta1 <- .9
  }
  if (is.null(opt_args$beta2)) {
    opt_args$beta2 <- .999
  }
  if (is.null(opt_args$eps)) {
    opt_args$eps <- 1e-8
  }
  if (is.null(opt_args$optimizer)) {
    opt_args$optimizer <- "adamw"
  }

  # map optimizer from character to int
  if (opt_args$optimizer == "sgd") {
    opt_args$optimizer_int <- 0L
  } else if (opt_args$optimizer == "adam") {
    opt_args$optimizer_int <- 1L
  } else if (opt_args$optimizer == "adamw") {
    opt_args$optimizer_int <- 2L
  } else {
    stop("optimizer must be from: \"sgd\", \"adam\", \"adamw\"")
  }

  opt_args
}

#' only used for \code{\link{sinkhorn()}}
check_sinkhorn_args <- function(skh_args) {
  # check the parameters for Sinkhorn algorithm
  args <- c(
    "reg",
    "with_grad",
    "n_threads",
    "method",
    "threshold",
    "max_iter",
    "zero_tol",
    "verbose"
  )
  args_match <- names(skh_args) %in% args
  if (!all(args_match)) {
    args_print <-
      paste0(paste0("\"", args, "\""), collapse = ", ")
    stop(
      paste0(names(skh_args)[!args_match], collapse = ", "),
      " not matching one of the sinkhorn arguments: ",
      args_print
    )
  }
  # browser()

  # warning messages for the `n_thread argument`

  # if (skh_args$n_threads) {
  #   # set the default is not set
  #   if (is.null(skh_args$reg)) {
  #     skh_args$reg <- .1
  #   }
  # }
  if (is.null(skh_args$with_grad)) {
    skh_args$with_grad <- FALSE
  }
  if (is.null(skh_args$n_threads)) {
    skh_args$n_threads <- 0L
  }
  if (is.null(skh_args$method)) {
    skh_args$method <- "auto"
  }
  if (is.null(skh_args$threshold)) {
    skh_args$threshold <- .1
  }
  if (is.null(skh_args$max_iter)) {
    skh_args$max_iter <- 1000
  }
  if (is.null(skh_args$zero_tol)) {
    skh_args$zero_tol <- 1e-6
  }
  if (is.null(skh_args$verbose)) {
    skh_args$verbose <- 10L
  }

  # map sinkhorn algo type from character to integer
  if (skh_args$method == "auto") {
    # skh_args$method_int <- 0L
  } else if (skh_args$method == "vanilla") {
    # skh_args$method_int <- 1L
    if ((skh_args$n_threads > 0) && (skh_args$verbose)) {
      warning("`n_threads` is not used in vanilla Sinkhorn!")
    }
  } else if (skh_args$method == "log") {
    # skh_args$method_int <- 2L
    if ((skh_args$n_threads == 0) && (skh_args$verbose)) {
      warning(
        paste0(
          "`n_threads = 0` for log Sinkhorn might be slow! ",
          "Considering setting `n_threads` for faster computation."
        )
      )
    }
  } else {
    stop("method must be from: \"auto\", \"vanilla\", or \"log\"")
  }

  skh_args
}

#' used for \code{\link{barycenter()}} and \code{\link{wdl()}}
check_barycenter_args <- function(brc_args) {
  # check the parameters for the Barycenter algorithm
  args <- c(
    "reg",
    "with_grad",
    "n_threads",
    "method",
    "threshold",
    "max_iter",
    "zero_tol",
    "verbose"
  )
  args_match <- names(brc_args) %in% args
  if (!all(args_match)) {
    args_print <-
      paste0(paste0("\"", args, "\""), collapse = ", ")
    stop(
      paste0(names(brc_args)[!args_match], collapse = ", "),
      " not matching one of the barycenter arguments: ",
      args_print
    )
  }

  # set the default is not set
  if (is.null(brc_args$reg)) {
    brc_args$reg <- .1
  }
  if (is.null(brc_args$with_grad)) {
    brc_args$with_grad <- FALSE
  }
  if (is.null(brc_args$n_threads)) {
    brc_args$n_threads <- 0L
  }
  if (is.null(brc_args$method)) {
    brc_args$method <- "auto"
  }
  if (is.null(brc_args$threshold)) {
    brc_args$threshold <- .1
  }
  if (is.null(brc_args$max_iter)) {
    brc_args$max_iter <- 1000
  }
  if (is.null(brc_args$zero_tol)) {
    brc_args$zero_tol <- 1e-6
  }
  if (is.null(brc_args$verbose)) {
    brc_args$verbose <- 10L
  }

  # map sinkhorn algo type from character to integer
  if (brc_args$method == "auto") {
    brc_args$method_int <- 0L
  } else if (brc_args$method == "parallel") {
    brc_args$method_int <- 1L
    if ((brc_args$n_threads > 0) && (brc_args$verbose)) {
      if (brc_args$verbose) {
        warning("`n_threads` is not used in parallel Barycenter!")
      }
    }
  } else if (brc_args$method == "log") {
    brc_args$method_int <- 2L
    if ((brc_args$n_threads == 0) && (brc_args$verbose)) {
      warning(
        paste0(
          "`n_threads = 0` for log Barycenter might be slow! ",
          "Considering setting `n_threads` for faster computation."
        )
      )
    }
  } else {
    stop("method must be from: \"auto\", \"parallel\", or \"log\"")
  }

  brc_args
}
