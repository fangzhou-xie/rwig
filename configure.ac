AC_INIT([rwig],[0.1.0],[fangzhou.xie@rutgers.edu])

# find R home and set correct compiler + flags
: ${R_HOME="`R RHOME`"}
if test -z "${R_HOME}"; then
  AC_MSG_ERROR([cannot determine R_HOME. Make sure you use R CMD INSTALL!])
fi
RBIN="${R_HOME}/bin/R"

# AC_PATH_PROG([PKG_CONFIG], [pkg-config])
AC_LANG(C++)

CUDA_FLAGS=""
CUDA_LIBS=""
CUDA_DEFS=""
CUDA_RULES=""

# Simple OS detection without AC_CANONICAL_HOST
case "$host_os" in
	"") # If host_os is empty, use uname
		host_os=`uname -s | tr '[[A-Z]]' '[[a-z]]'`
		;;
esac

case "${host_os}" in
	linux*)
		AC_MSG_NOTICE([Linux detected, checking for CUDA...])

		if test -z "$CUDA_FLAGS"; then
			AC_MSG_RESULT([no])

			# Fallback: manual search
			AC_MSG_NOTICE([manual search])
			AC_MSG_CHECKING([for CUDA in standard locations])
			for cudapath in "$CUDA_HOME" /usr/local/cuda /opt/cuda /usr/lib/cuda; do
				if test -n "$cudapath" && test -d "$cudapath/include" && test -f "$cudapath/bin/nvcc"; then
					CUDA_FLAGS="-I${cudapath}/include"
					CUDA_LIBS="-L${cudapath}/lib64 -lcudart -lcublas"
					CUDA_DEFS="-DHAVE_CUBLAS -DHAVE_CUDA_RUNTIME"
					# CUDA compile rules
				CUDA_RULES="NVCC = $cudapath/bin/nvcc
NVCC_FLAGS = -x cu -Xcompiler -fPIC -O2 -std=c++17 -I../inst/include -DHAVE_CUBLAS -DHAVE_CUDA_RUNTIME

OBJECTS = cuda_kernels.o cuda_sinkhorn.o cuda_barycenter.o cuda_wdl.o \$(patsubst %.cpp,%.o,\$(wildcard *.cpp))

.PHONY: all

all: \$(SHLIB)
\$(SHLIB): cuda_kernels.o

cuda_kernels.o: cuda_kernels.cu
	\$(NVCC) \$(NVCC_FLAGS) -c \$< -o \$@

cuda_sinkhorn.o: cuda_sinkhorn.cu
	\$(NVCC) \$(NVCC_FLAGS) -c \$< -o \$@

cuda_barycenter.o: cuda_barycenter.cu
	\$(NVCC) \$(NVCC_FLAGS) -c \$< -o \$@

cuda_wdl.o: cuda_wdl.cu
	\$(NVCC) \$(NVCC_FLAGS) -c \$< -o \$@
"
					AC_MSG_RESULT([yes ($cudapath)])
					break
				fi
			done

			if test -z "$CUDA_FLAGS"; then
				AC_MSG_RESULT([no])
			fi
		fi

		if test -n "$CUDA_FLAGS"; then
			AC_MSG_NOTICE([CUDA support enabled])
			AC_MSG_NOTICE([  CUDA_FLAGS: $CUDA_FLAGS])
			AC_MSG_NOTICE([  CUDA_LIBS: $CUDA_LIBS])
		else
			AC_MSG_NOTICE([CUDA not found, building without GPU support])
		fi
		;;
	*)
		AC_MSG_NOTICE([Non-Linux system ($host_os), skipping CUDA])
		;;
esac

AC_MSG_NOTICE([R path: $RBIN])
AC_MSG_NOTICE([CUDA_FLAG: $CUDA_FLAGS])
AC_MSG_NOTICE([CUDA_LIBS: $CUDA_LIBS])
AC_MSG_NOTICE([CUDA_DEFS: $CUDA_DEFS])
AC_MSG_NOTICE([CUDA_RULES: $CUDA_RULES])

AC_SUBST(CUDA_FLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_DEFS)
AC_SUBST(CUDA_RULES)

# `AC_CHECK_lib` check BLAS functions?

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
