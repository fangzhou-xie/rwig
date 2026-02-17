# check if CUDA is available

#' Check if CUDA is available
#'
#' @description
#' Check if CUDA is available for GPU computations.
#'
#' @return logical, TRUE if CUDA is available, FALSE otherwise
#'
#' @examples
#' if (check_cuda()) {
#'   cat("CUDA is available for GPU computations.\n")
#' } else {
#'   cat("CUDA is not available.\n")
#' }
#' @export
check_cuda <- function() {
  cuda_available_cpp()
}
