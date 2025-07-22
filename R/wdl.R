# Wasserstein Dictionary Learning
# TODO: shuffle the input docs

#' @export
wdl_spec <- function(
    wdl.control = list(
      embed_dim = 10, num_topics = 4, batch_size = 64, epochs = 2,
      verbose = FALSE, rng_seed = 123
    ),
    tokenizer.control = list(),
    word2vec.control = list(type = "cbow", dim = 10, min_count = 5),
    sinkhorn.control = list(
      regularizer = .1, sinkhorn_mode = "auto", sinkhorn_mode_threshold = .1,
      max_iter = 1000, zero_tol = 1e-6
    ),
    optimizer.control = list(
      optimizer = "adamw", lr = .005, decay = .01,
      beta1 = .9, beta2 = .999, eps = 1e-8
    )
) {
  list(
    wdl.control = check_wdl_args(wdl.control),
    tokenizer.control = check_tok_args(tokenizer.control),
    word2vec.control = check_w2v_args(word2vec.control),
    sinkhorn.control = check_skh_args(sinkhorn.control),
    optimizer.control = check_opt_args(optimizer.control)
  )
}

#' @export
wdl <- function(docs, ...) {
  if (!is.character(docs)) stop("docs should be Character Vector")
  UseMethod("wdl")
}

#' @export
wdl.character <- function(docs, specs = wdl_spec()) {

  # unpack the arguments for the model
  wdl_args <- specs$wdl.control
  tok_args <- specs$tokenizer.control
  w2v_args <- specs$word2vec.control
  skh_args <- specs$sinkhorn.control
  opt_args <- specs$optimizer.control

  # docs: character vector of input docs
  # first tokenize and embed
  tok_args <- append(list(x = docs), tok_args)
  toks <- do.call(tokenizers::tokenize_words, args = tok_args)

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

  if (wdl_args$verbose) cat("Preprocessing the data\n")

  # get distance matrix C and docs in dist Y
  distmat <- euclidean(emb)
  docdist <- doc2dist(toks, rownames(emb))

  browser()
  # test the performance of different algos?
  sinkhorn_log(docdist[,1], docdist[,2], distmat, .1) |> bench::mark()
  sinkhorn_log(docdist[,1], docdist[,2], distmat, .1, withgrad = TRUE) |> length()
  sinkhorn_parallel(docdist[,1:2], docdist[,3:4], distmat, .1) |> bench::mark()
  sinkhorn_parallel(docdist[,1:2], docdist[,3:4], distmat, .1, withgrad = TRUE) |> bench::mark()
  sol <- sinkhorn_parallel(docdist[,1:2], docdist[,3:4], distmat, .1)
  sol$iter
  sol$err
  A <- docdist[,1:4]
  C <- distmat
  sol1 <- barycenter_parallel(A, C, rep(.25, 4), .1)
  sol2 <- barycenter_log(A, C, rep(.25, 4), .1)
  sum(sol1$b - sol$b)
  tictoc::tic()
  sol1 <- barycenter_parallel(A, C, rep(.25, 4), .1)
  tictoc::toc()

  tictoc::tic()
  sol2 <- barycenter_log(A, C, rep(.25, 4), .1, withjac = TRUE)
  tictoc::toc()



  # TODO: maybe add checks in the algo?
  res <- wdl_cpp(
    docdist, distmat,
    skh_args$regularizer,              # reg
    wdl_args$num_topics,               # S
    wdl_args$batch_size,               # batch_size
    wdl_args$epochs,                   # epochs
    skh_args$sinkhorn_mode_int,        # sinkhorn_mode
    skh_args$sinkhorn_mode_threshold,  # sinkhorn_mode_threshold
    skh_args$max_iter,                 # maxIter of Sinkhorn
    skh_args$zero_tol,                 # zeroTol of Sinkhorn
    opt_args$optimizer_int,            # optimizer
    opt_args$lr,                       # eta: learning rate
    opt_args$decay,                    # gamma: decay
    opt_args$beta1,                    # beta1: used in Adam/AdamW
    opt_args$beta2,                    # beta2: used in Adam/AdamW
    opt_args$eps,                      # eps: used in Adam/AdamW
    wdl_args$rng_seed,                 # rng_seed: seed for reproducibility
    wdl_args$verbose                   # verbose: print information in wdl
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
    wdl.control = wdl_args,
    sinkhorn.control = skh_args,
    optimizer.control = opt_args
  )
  class(out) <- "wdl"
  out
}

# #' @rdname wdl
#' @export
print.wdl <- function(object, topic = 0, token_per_topic = 5, ...) {
  # `topic`: which topic to list (default all)
  # `token_per_topic`: number of tokens shown per topic

  print_topic <- function(topic) {
    cat(sprintf("Topic %s:\n", topic))
    print.default(
      summary.wdl(object, topic, token_per_topic = token_per_topic),
      digits = 2
    )
  }

  if (topic == 0) {
    for (topic in 1:ncol(object$topics)) {
      print_topic(topic)
    }
  } else {
    print_topic(topic)
  }

}

# #' @rdname wdl
#' @export
summary.wdl <- function(object, topic = 1, token_per_topic = 5, ...) {
  # `topic`: which topic to list (default all)
  # `token_per_topic`: number of tokens shown per topic

  topics_mat <- object$topics
  n <- min(nrow(topics_mat), token_per_topic)
  if (topic > ncol(topics_mat))
    stop("topic index greater than number of topics")
  sort(topics_mat[,topic], decreasing = TRUE)[1:n]
}


