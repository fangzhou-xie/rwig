# script to process the downloaded NYT headlines dataset
# Date: Nov 18, 2025

library(tidyverse) |> suppressPackageStartupMessages()
options(readr.show_progress = FALSE)
options(readr.show_col_types = FALSE)

headlines_raw_df <- here::here(
  "/mnt/Research/Rprojects/collect_nyt/data-raw/"
) |>
  fs::path("nyt_headlines_1980-2024.tsv") |>
  read_tsv(col_types = cols(.default = "c")) |>
  filter(!is.na(headline), !is.na(dt_firstpublished)) |>
  mutate(
    nn = nchar(headline),
    ref_date = as_date(dt_firstpublished) |> suppressWarnings(),
    ref_date = if_else(
      is.na(ref_date),
      as_date(dt_lastmajormodification) |> suppressWarnings(),
      ref_date
    ),
    ref_month = map(ref_date, function(x) {
      lubridate::mday(x) <- 1
      x
    }) |>
      list_c()
  )


#####################################################################
# 1. subset
#####################################################################

# filter titles with too many or too few characters
headlines_raw_df |>
  pull(nn) |>
  quantile(seq(0, 1, length.out = 200))

headlines_raw_df |> nrow()

# NOTE:
# on the left side
# from 0-10, remove
# from 11-20, there are some short titles (sort by frequency)
# from 21+, keep
headlines_raw_df |>
  filter(nn >= 11, nn <= 20) |>
  summarize(n = n(), .by = headline) |>
  filter(n == 1) |>
  # arrange(desc(n))
  pull(headline)

# on the right side
headlines_raw_df |>
  filter(nn >= 150, nn <= 180) |>
  pull(headline)

headlines_raw_df |>
  filter(nn >= 200) |>
  summarize(n = n(), .by = ref_month) |>
  ggplot(aes(x = ref_month, y = n)) +
  geom_line()

headlines_raw_df |>
  filter(nn >= 21, nn <= 30) |>
  pull(headline)

# original paper starting from 1989

headlines_1 <- headlines_raw_df |>
  filter(nn >= 11, nn <= 20) |>
  summarize(n = n(), .by = headline) |>
  filter(n == 1)

headlines <- bind_rows(
  # merge the first part
  headlines_raw_df |>
    semi_join(headlines_1, by = join_by(headline)),
  headlines_raw_df |>
    filter(nn >= 21, nn <= 200)
) |>
  select(headline, ref_date) |>
  as.data.frame()

# save the headlines as R data

usethis::use_data(headlines, overwrite = TRUE)

# headlines_df |>
#   write_tsv()

# headlines_df |>
#   summarize(n = n(), .by = ref_month) |>
#   ggplot(aes(x = ref_month, y = n)) +
#   geom_line()
# problems(headlines_raw_df)
#
# headlines_raw_df[288, ] |>
#   as.character()
#
#
# headlines_raw_df[290, ] |>
#   as.character()
#
# headlines_raw_df |> names()
#
# headlines_raw_df |>
#   filter(!is.na(headline)) |>
#   mutate(nn = nchar(headline)) |>
#   pull(nn) |>
#   quantile(seq(0.98, 1, length.out = 100))
#
# headlines_raw_df |>
#   filter(!is.na(headline)) |>
#   # filter(nchar(headline) >= 20, nchar(headline) <= 20) |>
#   filter(nchar(headline) >= 300) |>
#   mutate(
#     ref_date = as_date(dt_firstpublished),
#     ref_month = map(ref_date, function(x) {
#       lubridate::mday(x) <- 1
#       x
#     }) |>
#       list_c()
#   ) |>
#   select(ref_date, ref_month) |>
#   summarize(n = n(), .by = ref_month) |>
#   # pull(headline)
#   # pull(dt_firstpublished)
#   ggplot() +
#   geom_line(aes(x = ref_month, y = n))
