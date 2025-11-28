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


wig_startup_warning_message <- function() {
  warn_rule <- cli::rule(
    left = cli::style_bold("WARNING:"),
    right = cli::format_inline("{.pkg wig}"),
    col = "cyan",
    line = 2
  )

  warn_blas <- paste0(
    c(
      " Your current BLAS backend is:",
      paste0("", utils::sessionInfo()$BLAS, ".", collapse = ""),
      "For better performance,",
      "it's recommended to use an optimized BLAS library,",
      "such as Inter MKL or OpenBLAS.",
      "For example, you can consider",
      "{.href [ropenblas](https://github.com/prdm0/ropenblas)}."
    ),
    collapse = " "
  )

  # TODO: add vignette link for user to click directly
  # TODO: refer to the documentation page instead of help
  warn_thread <- paste0(
    c(
      " Automatically setting BLAS thread to be 1,",
      # "This should not affect most users with the default reference BLAS,",
      # "but only advanced users who have set up an optimized BLAS backend.",
      "and this is recommended for most users.",
      "If you want to set up threading for faster processing,",
      "you can set `n_threads` argument in functions:",
      # "{.run [sinkhorn()](wig::help(\"sinkhorn\"))}",
      # "and {.run [barycenter()](wig::help(\"barycenter\"))}.",
      "{.href [sinkhorn()](https://fangzhou-xie.github.io/wig/reference/sinkhorn.html)}",
      "and {.href [barycenter()](https://fangzhou-xie.github.io/wig/reference/barycenter.html)}.",
      "Please read the",
      "{.href [vignette](https://fangzhou-xie.github.io/wig/articles/threading.html)}",
      # "{.run vignette(\"threading\", package = \"wig\")}",
      "for advanced usage on threading."
    ),
    collapse = " "
  )
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

wig_startup_info_message <- function() {
  cite_rule <- cli::rule(
    left = cli::style_bold("CITATION:"),
    right = cli::format_inline("{.pkg wig}"),
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
  # warn user about the change in the threads
  RhpcBLASctl::blas_set_num_threads(1)

  # TODO: add vignette link about the threading (advanced)
  warn_msg <- wig_startup_warning_message()
  inform_startup(warn_msg)

  info_msg <- wig_startup_info_message()
  inform_startup(info_msg)
}

.onUnload <- function(libpath) {
  library.dynam.unload("wig", libpath)
}
