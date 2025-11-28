# check the specs of the model args

# TODO: check all the argument names are valid!

#' Model Specs for WDL and WIG models
#'
#' @description
#' Control the parameters of WDL and WIG models
#'
#' @details
#' Control parameters for the WDL and WIG model
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
#' Kingma, D. P., & Ba, J. (2015).
#' Adam: A method for stochastic optimization.
#' International Conference on Learning Representations (ICLR).
#'
#' Loshchilov, I., & Hutter, F. (2019).
#' Decoupled Weight Decay Regularization (No. arXiv:1711.05101). *arXiv*.
#' https://doi.org/10.48550/arXiv.1711.05101
#'
#' Xie, F. (2020). Wasserstein index generation model: Automatic generation of
#' time-series index with application to economic policy uncertainty.
#' *Economics Letters*, 186, 108874.
#' https://doi.org/10.1016/j.econlet.2019.108874
#'
#' Xie, F. (2025). Deriving the Gradients of Some Popular Optimal
#' Transport Algorithms (No. arXiv:2504.08722). *arXiv*.
#' https://doi.org/10.48550/arXiv.2504.08722
#'
#' @param wdl_control, list, parameters for WDL
#' @param tokenizer_control, list, parameters for
#' [tokenizers::tokenize_words()]
#' @param word2vec_control, list, parameters for
#' [word2vec::word2vec()]
#' @param barycenter_control, list, parameters for
#' [barycenter()]
#' @param optimizer_control, list, parameters for the optimizer
#' (SGD, Adam, AdamW)
#'
#' @seealso [wig_specs()], [barycenter()],
#' [word2vec::word2vec()], [tokenizers::tokenize_words()]
#'
#' @return list of the control lists
#'
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
    n_threads = 0,
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

  # barycenter `verbose` is ignored in WDL

  # return the list of arguments
  list(
    wdl_control = check_wdl_args(wdl_control),
    tokenizer_control = check_tok_args(tokenizer_control),
    word2vec_control = check_w2v_args(word2vec_control),
    barycenter_control = barycenter_control,
    optimizer_control = check_opt_args(optimizer_control)
  )
}


#' @rdname wdl_specs
#'
#' @param wig_control, list, parameters for WIG model
#'
#' @export
wig_specs <- function(
  wig_control = list(
    group_unit = "months",
    svd_method = "docs",
    standardize = TRUE
  ),
  wdl_control = list(
    embed_dim = 10,
    num_topics = 4,
    batch_size = 64,
    epochs = 2,
    rng_seed = 123,
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
  # check if the must-have default parameters are there
  # fill them if needed

  # barycenter `with_grad` defaults to FALSE, but we need TRUE for WDL/WIG
  barycenter_control <- check_barycenter_args(barycenter_control)
  barycenter_control$with_grad <- TRUE

  # barycenter `verbose` is ignored in WIG

  list(
    wig_control = check_wig_args(wig_control),
    wdl_control = check_wdl_args(wdl_control),
    tokenizer_control = check_tok_args(tokenizer_control),
    word2vec_control = check_w2v_args(word2vec_control),
    barycenter_control = barycenter_control,
    optimizer_control = check_opt_args(optimizer_control)
  )
}
