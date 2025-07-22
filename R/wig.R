
#' @export
wig_spec <- function(
    wig.control = list(
      group_time = "months", svd_method = "docs", standardize = TRUE
    ),
    wdl.control = list(
      embed_dim = 10, num_topics = 4, batch_size = 64, epochs = 2,
      verbose = FALSE, rng_seed = 123
    ),
    tokenizer.control = list(),
    word2vec.control = list(type = "cbow", dim = 10, min_count = 1),
    sinkhorn.control = list(
      sinkhorn_mode = "auto", regularizer = .1, sinkhorn_mode_threshold = .1,
      max_iter = 1000, zero_tol = 1e-6
    ),
    optimizer.control = list(
      optimizer = "adamw", lr = .005, decay = .01,
      beta1 = .9, beta2 = .999, eps = 1e-8
    )
) {

  # check if the must-have default parameters are there
  # fill them if needed

  list(
    wig.control = check_wig_args(wig.control),
    wdl.control = check_wdl_args(wdl.control),
    tokenizer.control = check_tok_args(tokenizer.control),
    word2vec.control = check_w2v_args(word2vec.control),
    sinkhorn.control = check_skh_args(sinkhorn.control),
    optimizer.control = check_opt_args(optimizer.control)
  )
}

#' @export
wig <- function(.data, date_col, docs_col, ...) {
  UseMethod("wig")
}

#' @export
wig.data.frame <- function(.data, date_col, docs_col, specs = wig_spec()) {

  # load all the parameters needed for the model
  wig_args <- specs$wig.control
  wdl_args <- specs$wdl.control
  tok_args <- specs$tokenizer.control
  w2v_args <- specs$word2vec.control
  skh_args <- specs$sinkhorn.control
  opt_args <- specs$optimizer.control

  wdl_specs <- list(
    wdl.control = wdl_args,
    tokenizer.control = tok_args,
    word2vec.control = w2v_args,
    sinkhorn.control = skh_args,
    optimizer.control = opt_args
  )

  # TODO: defuse the arguments
  # defuse the column names into vectors, without using `rlang`
  date_col <- substitute(date_col)
  docs_col <- substitute(docs_col)
  date_vec <- eval(date_col, .data, parent.frame())
  docs_vec <- eval(docs_col, .data, parent.frame())

  # check the `date_vec` is actually date/datetime
  if (!xts::is.timeBased(date_vec)) stop("`date_col` is not of date/datetime!")

  # run the WDL model, obtain A and W
  wdl_fit <- wdl(docs_vec, specs = wdl_specs)
  A <- wdl_fit$topics
  W <- wdl_fit$weights
  Yhat <- wdl_fit$docs_pred

  # after WDL, now need to run Truncated-SVD to reduce dimension
  if (wig_args$svd_method == "topics") {
    # this is the original implementation of WIG
    topics_svd <- t(tsvd(t(A), k = 1))
    wig_doc_scores <- c(topics_svd %*% W)
  } else if (wig_args$svd_method == "docs") {
    # this is the new proposed/improved implementation of WIG
    wig_doc_scores <- c(tsvd(t(Yhat), k = 1))
  } else {
    stop("`svd_method` not implemented!")
  }
  # print(wig_doc_scores)

  # TODO: maybe use `lubridate` or string manipulation instead of loop?
  # after generating the document-wise score, regroup them by time for index
  dts_end_inds <- xts::endpoints(date_vec, on = wig_args$group_time)
  dts_vec <- numeric(length(dts_end_inds) - 1)
  wig_index <- numeric(length(dts_end_inds) - 1)
  for (i in 2:length(dts_end_inds)) {
    dts_vec[i-1] <- date_vec[dts_end_inds[i-1]+1]
    wig_index[i-1] <- sum(wig_doc_scores[(dts_end_inds[i-1]+1):dts_end_inds[i]])
  }
  # print(wig_index)

  if (wig_args$standardize) {
    wm <- mean(wig_index)
    wv <- sd(wig_index)
    wig_index <- (wig_index - wm) / wv + 100
  }

  # TODO: output: dataframe with the index generated
  # TODO: regropu `date_vec` into first day of months
  data.frame(ref_date = date_vec, WIG = wig_index)
}
