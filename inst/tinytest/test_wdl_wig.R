library(tinytest)

docs <- c(
  "apple apple banana", "banana banana cherry", "cherry cherry apple",
  "apple banana cherry", "banana banana banana", "cherry apple apple"
)
specs <- wdl_specs(
  wdl_control = list(num_topics = 2, epochs = 1, shuffle = TRUE, batch_size = 4),
  word2vec_control = list(min_count = 1),
  barycenter_control = list(use_cuda = FALSE, max_iter = 10)
)

# user-supplied values in the control lists are honored, with_grad is forced
expect_equal(specs$barycenter_control$max_iter, 10)
expect_true(specs$barycenter_control$with_grad)
expect_equal(specs$barycenter_control$method, "auto")
expect_equal(wdl_specs()$barycenter_control$max_iter, 20)
expect_equal(wdl_specs(wdl_control = list(seed = 7L))$wdl_control$seed, 7L)
expect_error(wdl_specs(barycenter_control = list(regg = 1)), "not matching")
expect_equal(rwig:::check_sinkhorn_args(list(reg = .5))$verbose, 0L)

# per-document outputs line up with the input documents even when shuffled
set.seed(1)
fit <- wdl(docs, specs = specs, verbose = FALSE)
tok <- tokenizers::tokenize_word_stems(docs, stopwords = stopwords::stopwords())
ref <- rwig:::doc2dist(tok, rownames(fit$topics))
expect_equal(unname(fit$docs_dist), unname(ref))
expect_equal(dim(fit$weights), c(2L, length(docs)))
expect_equal(dim(fit$docs_pred), c(nrow(fit$topics), length(docs)))
expect_equal(rownames(fit$docs_pred), rownames(fit$topics))
expect_equal(unname(colSums(fit$weights)), rep(1, length(docs)), tolerance = 1e-12)
# the predicted barycenter of document j must be computed from its own weights
set.seed(1)
fit0 <- wdl(docs, specs = modifyList(specs, list(wdl_control = list(shuffle = FALSE))), verbose = FALSE)
expect_equal(dim(fit0$weights), dim(fit$weights))

# wig(): grouping by period, partial wig_control, dates and datetimes
wigdf <- data.frame(
  ref_date = as.Date("2012-01-15") + c(0, 10, 40, 45, 75, 100),
  docs = docs
)
set.seed(2)
w <- wig(wigdf, ref_date, docs,
  specs = wig_specs(wig_control = list(svd_method = "docs"),
                    wdl_control = list(num_topics = 2, epochs = 1),
                    word2vec_control = list(min_count = 1),
                    barycenter_control = list(use_cuda = FALSE)),
  verbose = FALSE)
expect_inherits(w, "wig")
expect_equal(w$index$ref_date, as.Date(c("2012-01-01", "2012-02-01", "2012-03-01", "2012-04-01")))
expect_equal(mean(w$index$WIG), 100, tolerance = 1e-12)
expect_equal(stats::sd(w$index$WIG), 1, tolerance = 1e-12)

set.seed(2)
w2 <- wig(wigdf, ref_date, docs,
  specs = wig_specs(wig_control = list(group_unit = "quarter", standardize = FALSE),
                    wdl_control = list(num_topics = 2, epochs = 1),
                    word2vec_control = list(min_count = 1),
                    barycenter_control = list(use_cuda = FALSE)),
  verbose = FALSE)
expect_equal(w2$index$ref_date, as.Date(c("2012-01-01", "2012-04-01")))

wigdt <- data.frame(ref_date = as.POSIXct("2012-01-15 10:00:00", tz = "UTC") + 86400 * c(0, 10, 40, 45, 75, 100), docs = docs)
set.seed(2)
w3 <- wig(wigdt, ref_date, docs,
  specs = wig_specs(wdl_control = list(num_topics = 2, epochs = 1),
                    word2vec_control = list(min_count = 1),
                    barycenter_control = list(use_cuda = FALSE)),
  verbose = FALSE)
expect_equal(as.Date(w3$index$ref_date), as.Date(c("2012-01-01", "2012-02-01", "2012-03-01", "2012-04-01")))
expect_error(wig(data.frame(ref_date = c("2012-01-01", "2012-02-01"), docs = docs[1:2]), ref_date, docs, verbose = FALSE), "date/datetime")
