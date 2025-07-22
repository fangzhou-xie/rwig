
# utility functions for WDL

euclidean <- function(embedding) {
  dist_mat <- euclidean_cpp(embedding)
  colnames(dist_mat) <- rownames(embedding)
  rownames(dist_mat) <- rownames(embedding)
  dist_mat
}

doc2dist <- function(doc_tokens, dict) {
  docmat <- doc2dist_cpp(doc_tokens, dict)
  rownames(docmat) <- dict
  docmat
}
