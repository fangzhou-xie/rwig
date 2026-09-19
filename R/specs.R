# model specs: the control lists for wdl() and wig()

#' Model Specs for WDL and WIG models
#'
#' @description
#' Control the parameters of WDL and WIG models
#'
#' @details
#' See \code{vignette("specs")} for details on the parameters.
#' Entries left out of a control list take the defaults shown in the usage
#' section. In `barycenter_control`, `with_grad` is always `TRUE` (the
#' gradients are what WDL trains on) and `max_iter` is the fixed number of
#' barycenter iterations per training step, so keep it small: the batched
#' training keeps a history of `max_iter` iterations for every document
#' in a batch.
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
#' *International Conference on Learning Representations (ICLR)*.
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
#' [word2vec::word2vec()], [tokenizers::tokenize_words()],
#' \code{vignette("specs")}
#'
#' @return list of the control lists
#'
#' @export
wdl_specs <- function(
  wdl_control = list(
    num_topics = 4,
    batch_size = 64,
    epochs = 2,
    shuffle = TRUE,
    seed = 42L
  ),
  tokenizer_control = list(stopwords = stopwords::stopwords()),
  word2vec_control = list(type = "cbow", dim = 10, min_count = 3),
  barycenter_control = list(
    reg = .1,
    with_grad = TRUE,
    use_cuda = TRUE,
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
  list(
    wdl_control = check_wdl_args(wdl_control),
    tokenizer_control = check_tok_args(tokenizer_control),
    word2vec_control = check_w2v_args(word2vec_control),
    barycenter_control = check_wdl_barycenter_args(barycenter_control),
    optimizer_control = check_opt_args(optimizer_control)
  )
}

# barycenter control for WDL/WIG: defaults come from the wdl_specs() signature
# (max_iter = 20, not the 1000 of barycenter()); `with_grad` must be TRUE
check_wdl_barycenter_args <- function(barycenter_control) {
  # signature defaults (max_iter = 20, ...) layered over the full set from
  # barycenter(), so entries the signature omits (verbose) still exist
  defaults <- fill_defaults(formal_default("barycenter_control", wdl_specs), ot_defaults())
  barycenter_control <- check_barycenter_args(barycenter_control, defaults)
  barycenter_control$with_grad <- TRUE
  barycenter_control
}


#' @rdname wdl_specs
#'
#' @param wig_control, list, parameters for WIG model
#'
#' @export
wig_specs <- function(
  wig_control = list(
    group_unit = "month",
    svd_method = "topics",
    standardize = TRUE
  ),
  wdl_control = list(
    num_topics = 4,
    batch_size = 64,
    epochs = 2,
    shuffle = TRUE,
    seed = 42L
  ),
  tokenizer_control = list(stopwords = stopwords::stopwords()),
  word2vec_control = list(type = "cbow", dim = 10, min_count = 1),
  barycenter_control = list(
    reg = .1,
    with_grad = TRUE,
    use_cuda = TRUE,
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
  list(
    wig_control = check_wig_args(wig_control),
    wdl_control = check_wdl_args(wdl_control),
    tokenizer_control = check_tok_args(tokenizer_control),
    word2vec_control = check_w2v_args(word2vec_control),
    barycenter_control = check_wdl_barycenter_args(barycenter_control),
    optimizer_control = check_opt_args(optimizer_control)
  )
}
