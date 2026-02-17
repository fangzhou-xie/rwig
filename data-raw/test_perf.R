# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so:/usr/lib/x86_64-linux-gnu/libopenblas.so CPUPROFILE=/tmp/profile.out bash -c 'trap "" ALRM; Rscript test_perf.R'
# ls -la /tmp/profile.out*
# pprof --text /usr/bin/Rscript /tmp/profile.out_*

# fixed the "perf" issue
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libopenblas.so NVBLAS_CONFIG_FILE=nvblas.conf perf record Rscript test_perf.R
# perf report
#

library(rwig) |> suppressPackageStartupMessages()

headlines_df <- headlines |>
  tibble::as_tibble() |>
  dplyr::mutate(headline = stringr::str_to_lower(headline)) |>
  head(400)

wig_fit <- headlines_df |>
  wig(
    ref_date,
    headline,
    wig_specs(
      wdl_control = list(verbose = FALSE),
      word2vec_control = list(min_count = 3),
      barycenter_control = list(method = "parallel", max_iter = 1000)
    )
  )
