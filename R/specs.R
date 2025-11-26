# check the specs of the model args

# TODO: check all the argument names are valid!

#' @export
wdl_specs <- function(
  wdl_control = list(
    embed_dim = 10,
    num_topics = 4,
    batch_size = 64,
    epochs = 2,
    shuffle = TRUE,
    verbose = FALSE
  ),
  tokenizer_control = list(),
  word2vec_control = list(type = "cbow", dim = 10, min_count = 1),
  barycenter_control = list(
    reg = .1,
    with_grad = TRUE,
    method = "auto",
    threshold = .1,
    max_iter = 20,
    zero_tol = 1e-6
  ),
  optimizer_control = list(
    optimizer = "adamw",
    lr = .005,
    decay = .01,
    beta1 = .9,
    beta2 = .999,
    eps = 1e-8
  )
) {
  # barycenter `with_grad` defaults to FALSE, but we need TRUE for WDL/WIG
  barycenter_control <- check_barycenter_args(barycenter_control)
  barycenter_control$with_grad <- TRUE

  # return the list of arguments
  list(
    wdl_control = check_wdl_args(wdl_control),
    tokenizer_control = check_tok_args(tokenizer_control),
    word2vec_control = check_w2v_args(word2vec_control),
    barycenter_control = barycenter_control,
    optimizer_control = check_opt_args(optimizer_control)
  )
}

#' @export
wig_specs <- function(
  wig_control = list(
    group_time = "months",
    svd_method = "docs",
    standardize = TRUE
  ),
  wdl_control = list(
    embed_dim = 10,
    num_topics = 4,
    batch_size = 64,
    epochs = 2,
    rng_seed = 123,
    verbose = 0
  ),
  tokenizer_control = list(),
  word2vec_control = list(type = "cbow", dim = 10, min_count = 1),
  barycenter_control = list(
    reg = .1,
    with_grad = TRUE,
    method = "auto",
    threshold = .1,
    max_iter = 20,
    zero_tol = 1e-6
  ),
  optimizer_control = list(
    optimizer = "adamw",
    lr = .005,
    decay = .01,
    beta1 = .9,
    beta2 = .999,
    eps = 1e-8
  )
) {
  # check if the must-have default parameters are there
  # fill them if needed

  # barycenter `with_grad` defaults to FALSE, but we need TRUE for WDL/WIG
  barycenter_control <- check_barycenter_args(barycenter_control)
  barycenter_control$with_grad <- TRUE

  list(
    wig_control = check_wig_args(wig_control),
    wdl_control = check_wdl_args(wdl_control),
    tokenizer_control = check_tok_args(tokenizer_control),
    word2vec_control = check_w2v_args(word2vec_control),
    barycenter_control = barycenter_control,
    optimizer_control = check_opt_args(optimizer_control)
  )
}
