#' Wasserstein Index Generation model
#'
#' @description
#' Wasserstein Index Generation (WIG) model for time-series sentiment index
#' autogeneration
#'
#' @details
#' This is the re-implementation of WIG model from scratch in R.
#'
#' @references
#'
#' Xie, F. (2020). Wasserstein index generation model: Automatic generation of
#' time-series index with application to economic policy uncertainty.
#' *Economics Letters*, 186, 108874.
#' https://doi.org/10.1016/j.econlet.2019.108874
#'
#' @param date_col name of the column for dates
#' @param docs_col name of the column for the texts
#' @param spec list, model specification for WIG
#' see \code{\link{wig_spec()}} for reference
#' @param verbose bool, whether to print useful info
#'
#' @examples
#' # create a small dataset
#' wigdf <- data.frame(
#'   ref_date = as.Date(c("2012-01-01", "2012-02-01")),
#'   docs = c("this is a sentence", "this is another sentence")
#' )
#'
#' wigfit <- wig(wigdf, ref_date, docs)
#'
#'
#'
#' @export
wig <- function(.data, date_col, docs_col, ...) {
  UseMethod("wig")
}

#' @export
wig.data.frame <- function(
  .data,
  date_col,
  docs_col,
  specs = wig_specs(),
  verbose = TRUE
) {
  # load all the parameters needed for the model
  wig_args <- specs$wig_control
  wdl_args <- specs$wdl_control
  tok_args <- specs$tokenizer_control
  w2v_args <- specs$word2vec_control
  brc_args <- specs$barycenter_control
  opt_args <- specs$optimizer_control

  wdl_specs <- list(
    wdl_control = wdl_args,
    tokenizer_control = tok_args,
    word2vec_control = w2v_args,
    barycenter_control = brc_args,
    optimizer_control = opt_args
  )

  # TODO: defuse the arguments
  # defuse the column names into vectors, without using `rlang`
  date_col <- substitute(date_col)
  docs_col <- substitute(docs_col)
  date_vec <- eval(date_col, .data, parent.frame())
  docs_vec <- eval(docs_col, .data, parent.frame())

  # check the `date_vec` is actually date/datetime
  if (!xts::is.timeBased(date_vec)) {
    stop("`date_col` is not of date/datetime!")
  }

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
  # dts_end_inds <- xts::endpoints(date_vec, on = wig_args$group_time)
  # dts_vec <- numeric(length(dts_end_inds) - 1)
  # wig_index <- numeric(length(dts_end_inds) - 1)
  # for (i in 2:length(dts_end_inds)) {
  #   dts_vec[i - 1] <- date_vec[dts_end_inds[i - 1] + 1]
  #   wig_index[i - 1] <- sum(wig_doc_scores[
  #     (dts_end_inds[i - 1] + 1):dts_end_inds[i]
  #   ])
  # }

  # dts_vec_chr <- as.character(dts_vec)
  dts_vec <- gsub("[[:digit:]]{2}$", "01", as.character(date_vec))
  wig_raw_df <- data.frame(ref_date = dts_vec, WIG = wig_doc_scores)

  # base R solution to summarize
  wig_raw_by_df <- by(wig_raw_df, wig_raw_df$ref_date, function(df) {
    with(df, data.frame(ref_date = ref_date[[1]], WIG = sum(WIG)))
  })
  wig_df <- do.call(rbind, wig_raw_by_df)
  rownames(wig_df) <- NULL # remove the row names

  date_vec <- wig_df$ref_date
  wig_index <- wig_df$WIG
  if (wig_args$standardize) {
    wig_index <- (wig_index - mean(wig_index)) / sd(wig_index) + 100
  }

  wig_df <- data.frame(ref_date = date_vec, WIG = wig_index)

  # prep the output and set class
  out <- list(
    index = wig_df,
    wdl_model = wdl_fit
  )
  class(out) <- "wig"
  out
}

#' @rdname wig
#' @export
print.wig <- function(object, topic = 1, token_per_topic = 5, ...) {
  cat(sprintf(
    "WIG model from %s to %s\n",
    as.character(min(object$index$ref_date)),
    as.character(max(object$index$ref_date))
  ))

  print.wdl(object$wdl_model, topic, token_per_topic, ...)
}

#
# #' @rdname wig
# #' @export
# summary.wig <- function(object, topic = 1, token_per_topic = 5, ...) {
#   cat("Summary of WDL topics:\n")
#
#   summary.wdl(object$wdl_model, topic, token_per_topic, ...)
# }
