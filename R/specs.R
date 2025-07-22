
# check the specs of the model args

check_wig_args <- function(wig_args) {
  # check WIG arguments
  if (is.null(wig_args$group_time))
    wig_args$group_time <- "months"
  if (is.null(wig_args$svd_method))
    wig_args$svd_method <- "topics"
  if (is.null(wig_args$standardize))
    wig_args$standardize <- TRUE

  wig_args
}


check_wdl_args <- function(wdl_args) {
  # check WDL arguments: hyper-parameters
  if (is.null(wdl_args$num_topics))
    wdl_args$num_topics <- 4
  if (is.null(wdl_args$batch_size))
    wdl_args$batch_size <- 64
  if (is.null(wdl_args$epochs))
    wdl_args$epochs <- 2
  if (is.null(wdl_args$rng_seed))
    wdl_args$rng_seed <- 123L
  if (is.null(wdl_args$verbose))
    wdl_args$verbose <- FALSE

  # if `verbose` is NA, abort!
  if (is.na(wdl_args$verbose)) stop("`verbose` argument cannot be NA!")

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
  if (is.null(w2v_args$dim)) # embedding depth: hyper-parameters
    w2v_args$dim <- 10
  if (is.null(w2v_args$min_count))
    w2v_args$min_count <- 5
  if (is.null(w2v_args$type))
    w2v_args$type <- "cbow"
  # if (is.null(w2v_args$stopwords))
  #   w2v_args$stopwords <- stopwords::stopwords()

  w2v_args
}

check_skh_args <- function(skh_args) {
  # check the parameters for Sinkhorn (Barycenter) algorithm
  if (is.null(skh_args$regularizer))
    skh_args$regularizer <- .1
  if (is.null(skh_args$sinkhorn_mode))
    skh_args$sinkhorn_mode <- "auto"
  if (is.null(skh_args$sinkhorn_mode_threshold))
    skh_args$sinkhorn_mode_threshold <- .1
  if (is.null(skh_args$max_iter))
    skh_args$max_iter <- 1000
  if (is.null(skh_args$zero_tol))
    skh_args$zero_tol <- 1e-6

  # map sinkhorn algo type from character to integer
  if (skh_args$sinkhorn_mode == "auto") {
    skh_args$sinkhorn_mode_int <- 0L
  } else if (skh_args$sinkhorn_mode == "parallel") {
    skh_args$sinkhorn_mode_int <- 1L
  } else if (skh_args$sinkhorn_mode == "log") {
    skh_args$sinkhorn_mode_int <- 2L
  } else {
    stop("sinkhorn_mode must be from: \"auto\", \"parallel\", or \"log\"")
  }

  skh_args
}

check_opt_args <- function(opt_args) {

  # check the parameters for the optimizer
  if (is.null(opt_args$lr))
    opt_args$lr <- .005
  if (is.null(opt_args$decay))
    opt_args$decay <- .01
  if (is.null(opt_args$beta1))
    opt_args$beta1 <- .9
  if (is.null(opt_args$beta2))
    opt_args$beta2 <- .999
  if (is.null(opt_args$eps))
    opt_args$eps <- 1e-8
  if (is.null(opt_args$optimizer))
    opt_args$optimizer <- "adamw"

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
