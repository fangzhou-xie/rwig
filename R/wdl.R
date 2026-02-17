#' Wasserstein Dictionary Learning model
#'
#' @description
#' Wasserstein Dictionary Learning (WDL) model for topic modeling
#'
#' @details
#' This is the re-implementation of WDL model from ground up,
#' and it calls the \code{\link{barycenter}} under the hood
#' (to be precise directly calling the underlying C++ routine
#' for \code{\link{barycenter}})
#'
#' @references
#'
#' Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport:
#' With Applications to Data Science.
#' *Foundations and Trends® in Machine Learning*, 11(5–6), 355–607.
#' https://doi.org/10.1561/2200000073
#'
#' Schmitz, M. A., Heitz, M., Bonneel, N., Ngolè, F., Coeurjolly, D.,
#' Cuturi, M., Peyré, G., & Starck, J.-L. (2018).
#' Wasserstein dictionary learning:
#' Optimal transport-based unsupervised nonlinear dictionary learning.
#' *SIAM Journal on Imaging Sciences*, 11(1), 643–678.
#' https://doi.org/10.1137/17M1140431
#'
#' Xie, F. (2025). Deriving the Gradients of Some Popular Optimal
#' Transport Algorithms (No. arXiv:2504.08722). *arXiv*.
#' https://doi.org/10.48550/arXiv.2504.08722
#'
#' @param docs character vector, sentences to be analyzed
#' @param specs list, model specification for the WDL
#' see \code{\link{wdl_specs}} for reference
#' @param verbose bool, whether to print useful info
#' @param ... only for compatibility
#'
#' @returns topics and weights computed from the WDL given the input data
#'
#' @examples
#' # simple WDL example
#' sentences <- c("this is a sentence", "this is another one")
#' wdl_fit <- wdl(sentences)
#'
#' @seealso
#' \code{vignette("wdl-model")}
#'
#' @importFrom Rcpp evalCpp
#' @export
wdl <- function(docs, ...) {
  if (!is.character(docs)) {
    stop("docs should be Character Vector")
  }
  UseMethod("wdl")
}

#' @rdname wdl
#' @importFrom Rcpp evalCpp
#' @export
wdl.character <- function(docs, specs = wdl_specs(), verbose = TRUE, ...) {
  # unpack the arguments for the model
  wdl_args <- specs$wdl_control
  tok_args <- specs$tokenizer_control
  w2v_args <- specs$word2vec_control
  brc_args <- specs$barycenter_control
  opt_args <- specs$optimizer_control

  if (wdl_args$verbose) {
    message("Preprocessing the data...")
    message("Running tokenizer on the sentences...")
  }

  # docs: character vector of input docs
  # first tokenize and embed
  tok_args <- append(list(x = docs), tok_args)
  toks <- do.call(tokenizers::tokenize_word_stems, args = tok_args)

  if (wdl_args$verbose) {
    message("Running Word2Vec for the embeddings and distance matrix...")
  }

  w2v_args <- append(list(x = toks), w2v_args)
  model <- do.call(word2vec::word2vec, args = w2v_args)
  emb <- as.matrix(model)

  # stop if there is no valid vocab in the model, set the `min_count`
  if (length(emb) == 1) {
    stop(paste0(
      "There is no valid token for the model. ",
      "Try to decrease `min_count` argument (default to 5)."
    ))
  }

  # get distance matrix C and docs in dist Y
  distmat <- euclidean(emb)
  docdist <- doc2dist(toks, rownames(emb))

  # shuffle the input docs
  if (wdl_args$shuffle) {
    shuffled_ids <- sample.int(ncol(docdist), ncol(docdist))
    docdist <- docdist[, shuffled_ids]
  }

  # browser()
  k1 <- exp(-min(distmat) / brc_args$reg)
  k2 <- exp(-max(distmat) / brc_args$reg)

  # dispatch the barycenter method if "auto"
  if (brc_args$method_int == 0) {
    if (min(k1, k2) < brc_args$threshold) {
      brc_args$method_int <- 2 # log

      if (verbose) message("`method` is automatically switched to \"log\"")
    } else {
      brc_args$method_int <- 1 # parallel

      if (verbose) message("`method` is automatically switched to \"parallel\"")
    }
  }

  # running the WDL model here
  res <- wdl_cpp(
    docdist,
    distmat,
    brc_args$reg, # reg
    wdl_args$num_topics, # S
    wdl_args$n_threads, # num of threads
    wdl_args$batch_size, # batch_size
    wdl_args$epochs, # epochs
    # brc_args$method, # sinkhorn_mode
    brc_args$method_int, # sinkhorn_mode_threshold
    brc_args$max_iter, # maxIter of Sinkhorn
    brc_args$zero_tol, # zeroTol of Sinkhorn
    opt_args$optimizer_int, # optimizer
    opt_args$lr, # eta: learning rate
    opt_args$decay, # gamma: decay
    opt_args$beta1, # beta1: used in Adam/AdamW
    opt_args$beta2, # beta2: used in Adam/AdamW
    opt_args$eps, # eps: used in Adam/AdamW
    # wdl_args$rng_seed, # rng_seed: seed for reproducibility
    wdl_args$verbose # verbose: print information in wdl
  )
  topics <- res$A
  weights <- res$W
  Yhat <- res$Yhat

  # browser()

  # from the topics/weights matrices, build the topics matrix
  rownames(topics) <- rownames(emb)
  rownames(Yhat) <- rownames(emb)
  colnames(topics) <- paste0("topic", 1:(wdl_args$num_topics))
  rownames(weights) <- paste0("topic", 1:(wdl_args$num_topics))

  # return a whole list of everything
  out <- list(
    docs = docs,
    docs_dist = docdist,
    docs_pred = res$Yhat,
    topics = topics,
    weights = weights,
    wdl_control = wdl_args,
    barycenter_control = brc_args,
    optimizer_control = opt_args
  )
  class(out) <- "wdl"
  out
}

#' @rdname wdl
#'
#' @param x WDL model
#' @param topic int, number of topic to be printed
#' @param token_per_topic int, number of tokens to be printed
#'
#' @export
print.wdl <- function(x, topic = 0, token_per_topic = 5, ...) {
  # `topic`: which topic to list (default all)
  # `token_per_topic`: number of tokens shown per topic

  cat("WDL model topics:\n\n")

  print_topic <- function(topic) {
    cat(sprintf("Topic %s:\n", topic))
    print.default(
      summary.wdl(x, topic, token_per_topic = token_per_topic),
      digits = 2,
      ...
    )
    cat("\n")
  }

  if (topic == 0) {
    for (topic in 1:ncol(x$topics)) {
      print_topic(topic)
    }
  } else {
    print_topic(topic)
  }
}

#' @rdname wdl
#'
#' @param object WDL model
#' @param topic int, number of topic to be printed
#' @param token_per_topic int, number of tokens to be printed
#'
#' @export
summary.wdl <- function(object, topic = 1, token_per_topic = 5, ...) {
  # `topic`: which topic to list (default all)
  # `token_per_topic`: number of tokens shown per topic

  topics_mat <- object$topics
  n <- min(nrow(topics_mat), token_per_topic)
  if (topic > ncol(topics_mat)) {
    stop("topic index greater than number of topics")
  }
  sort(topics_mat[, topic], decreasing = TRUE)[1:n]
}

# TODO: maybe add `plot` too? for topic word cloud?
