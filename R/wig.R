# Start of the period (day, week, month, quarter, year, ...) containing each
# date or datetime, using base R's cut() for dates. Weeks start on Monday.
floor_period <- function(x, unit = "month") {
  starts <- as.character(cut(x, breaks = unit))
  if (inherits(x, "Date")) {
    as.Date(starts)
  } else {
    as.POSIXct(starts, tz = attr(x, "tzone") %||% "")
  }
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# WIG score aggregation: sum the document scores within each period
aggregate_by_period <- function(datetimes, values, unit = "month") {
  out <- stats::aggregate(
    values,
    by = list(ref_date = floor_period(datetimes, unit)),
    FUN = sum
  )
  names(out) <- c("ref_date", "WIG")
  out
}

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
#' @param .data a dataframe containing the dates/datetimes and documents
#' @param date_col name of the column for dates (`Date`) or datetimes
#' (`POSIXct`); documents are grouped by the `group_unit` of `wig_control`
#' (see [wig_specs()]), any `breaks` accepted by [cut.Date()] such as
#' "day", "week", "month", "quarter", "year" or "2 months"
#' @param docs_col name of the column for the texts/documents
#' @param specs list, model specification for WIG
#' see \code{\link{wig_specs}} for reference
#' @param verbose bool, whether to print useful info
#' @param ... only for compatibility
#'
#' @return "wig" class, i.e. list of the index and the WDL model
#'
#' @examples
#' # create a small dataset
#' wigdf <- data.frame(
#'   ref_date = as.Date(c("2012-01-01", "2012-02-01")),
#'   docs = c("this is a sentence", "this is another sentence"))
#'
#' wigfit <- wig(wigdf, ref_date, docs,
#'   specs = wig_specs(wdl_control = list(num_topics = 2),word2vec_control = list(min_count = 1)),
#'   verbose = FALSE)
#'
#' @seealso
#' \code{vignette("wdl-model")}
#'
#' @export
wig <- function(.data, date_col, docs_col, ...) {
  UseMethod("wig")
}

#' @rdname wig
#'
#' @export
wig.data.frame <- function(
  .data,
  date_col,
  docs_col,
  specs = wig_specs(),
  verbose = TRUE,
  ...
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

  # defuse the column names into vectors, without using `rlang`
  date_vec <- eval(substitute(date_col), .data, parent.frame())
  docs_vec <- eval(substitute(docs_col), .data, parent.frame())

  # check the `date_vec` is actually date/datetime
  if (!inherits(date_vec, c("Date", "POSIXt"))) {
    stop("`date_col` is not of date/datetime!")
  }

  # run the WDL model, obtain A and W
  wdl_fit <- wdl(docs_vec, specs = wdl_specs, verbose = verbose)
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

  # after generating the document-wise scores, regroup them by period
  wig_df <- aggregate_by_period(date_vec, wig_doc_scores, unit = wig_args$group_unit)
  if (wig_args$standardize) {
    wig_df$WIG <- (wig_df$WIG - mean(wig_df$WIG)) / stats::sd(wig_df$WIG) + 100
  }

  # prep the output and set class
  out <- list(
    index = wig_df,
    wdl_model = wdl_fit
  )
  class(out) <- "wig"
  out
}

#' @rdname wig
#'
#' @param x WIG model
#' @param topic int, number of topic to be printed
#' @param token_per_topic int, number of tokens to be printed
#'
#' @export
print.wig <- function(x, topic = 1, token_per_topic = 10, ...) {
  cat(sprintf(
    "WIG model from %s to %s\n",
    as.character(min(x$index$ref_date)),
    as.character(max(x$index$ref_date))
  ))

  print.wdl(x$wdl_model, topic, token_per_topic, ...)
}


#' @rdname wig
#'
#' @param object WIG model
#' @param topic int, number of topic to be printed
#' @param token_per_topic int, number of tokens to be printed
#'
#' @export
summary.wig <- function(object, topic = 1, token_per_topic = 10, ...) {
  cat("Summary of WDL topics:\n")

  summary.wdl(object$wdl_model, topic, token_per_topic, ...)
}
