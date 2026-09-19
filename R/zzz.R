# onAttach and onUnload
# start up warning message: following my `rethnicity` package
# https://github.com/fangzhou-xie/rethnicity/blob/main/R/zzz.R

inform_startup <- function(msg, ...) {
  if (is.null(msg)) {
    return()
  }

  # replace `rlang::inform()`, therefore no need to depend directly on rlang
  # though cli eventually will depend on rlang
  # cli::cli_inform(msg, ..., class = "packageStartupMessage")
  rlang::inform(msg, ..., class = "packageStartupMessage")
}

# Advice on the BLAS backend and its thread count. rwig does not change the
# BLAS thread count itself (that is a session-wide setting owned by the
# user); it only recommends a single BLAS thread and points to RhpcBLASctl.
startup_warning_message <- function(
  blas_ctl = requireNamespace("RhpcBLASctl", quietly = TRUE)
) {
  warn_rule <- cli::rule(
    left = cli::style_bold("NOTE:"),
    right = cli::format_inline("{.pkg rwig}"),
    col = "cyan",
    line = 2
  )

  blas <- utils::sessionInfo()$BLAS
  optimized <- grepl("openblas|mkl|atlas|blis|accelerate|flexiblas", blas,
                     ignore.case = TRUE)
  warn_blas <- if (optimized) {
    paste0(" Your current BLAS backend is: ", blas, " (optimized).")
  } else {
    paste(
      " Your current BLAS backend is:",
      paste0(blas, "."),
      "For better performance,",
      "it's recommended to use an optimized BLAS library,",
      "such as Intel MKL or OpenBLAS.",
      "For example, you can consider",
      "{.href [ropenblas](https://github.com/prdm0/ropenblas)}."
    )
  }

  # rwig runs its own worker threads (`n_threads`) and its BLAS calls are
  # mostly small, so one BLAS thread is usually fastest and avoids
  # oversubscribing the cores
  why_one <- paste(
    "{.pkg rwig} does its own multi-threading through the `n_threads` argument of",
    "{.href [sinkhorn()](https://fangzhou-xie.github.io/rwig/reference/sinkhorn.html)}",
    "and {.href [barycenter()](https://fangzhou-xie.github.io/rwig/reference/barycenter.html)},",
    "so a single BLAS thread is recommended to avoid oversubscribing the cores."
  )
  warn_thread <- if (blas_ctl) {
    n <- RhpcBLASctl::blas_get_num_procs()
    paste(
      sprintf(" Your BLAS is currently set to use %d thread%s.", n, if (n == 1) "" else "s"),
      why_one,
      if (n != 1) "You can set it for this session with {.code RhpcBLASctl::blas_set_num_threads(1)}." else "",
      "Please read the",
      "{.href [vignette](https://fangzhou-xie.github.io/rwig/articles/threading.html)}",
      "for advanced usage on threading."
    )
  } else {
    paste(
      "", why_one,
      "To control the BLAS thread count, install {.pkg RhpcBLASctl}",
      "({.code install.packages(\"RhpcBLASctl\")}) and call",
      "{.code RhpcBLASctl::blas_set_num_threads(1)}.",
      "Please read the",
      "{.href [vignette](https://fangzhou-xie.github.io/rwig/articles/threading.html)}",
      "for advanced usage on threading."
    )
  }
  warns <- paste0(
    cli::col_yellow("!"),
    c(
      cli::format_inline(warn_blas),
      cli::format_inline(warn_thread)
    ),
    collapse = "\n"
  )
  paste0(warn_rule, "\n", warns, collapse = "\n")
}

startup_info_message <- function() {
  cite_rule <- cli::rule(
    left = cli::style_bold("CITATION:"),
    right = cli::format_inline("{.pkg rwig}"),
    col = "cyan",
    line = 2
  )

  # NOTE: double curly brackets for escaping!
  cite_meth <- paste0(
    c(
      " @article{{xie2020,",
      "   title = {{Wasserstein Index Generation Model: Automatic Generation of Time-Series Index with Application to Economic Policy Uncertainty}},",
      "   author = {{Xie, Fangzhou}},",
      "   year = 2020,",
      "   journal = {{Economics Letters}},",
      "   volume = {{186}},",
      "   pages = {{108874}},",
      "   issn = {{0165-1765}},",
      "   doi = {{10.1016/j.econlet.2019.108874}},",
      "   urldate = {{2019-12-10}},",
      " }}"
    ),
    collapse = "\n"
  )

  # TODO: add the following citation after the software paper becomes public
  cite_soft <- paste0(
    c(
      " The software paper is:"
    )
  )

  cite_msg <- cli::format_inline(
    cli::col_blue(cli::symbol$info),
    c(
      " Please use the following to cite my works, thanks!\n\n",
      cite_meth,
      "\n"
      # cite_soft,
      # "\n"
    )
  )

  # paste0(info_rule, "\n", info_msg, "\n\n", cite_rule, "\n", cite_msg, "\n")
  paste0("\n", cite_rule, "\n", cite_msg, "\n")
}

.onAttach <- function(...) {
  # advice only: rwig never changes the session's BLAS thread count
  warn_msg <- startup_warning_message()
  inform_startup(warn_msg)

  info_msg <- startup_info_message()
  inform_startup(info_msg)
}

.onUnload <- function(libpath) {
  library.dynam.unload("rwig", libpath)
}
