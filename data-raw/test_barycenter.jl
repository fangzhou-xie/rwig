
# calculate the gradients using Julia ForwardDiff

using LinearAlgebra
using ForwardDiff: gradient, jacobian


A = [0.3 0.2; 0.2 0.1; 0.1 0.2; 0.1 0.1; 0.3 0.4]
b = [0.2, 0.2, 0.2, 0.2, 0.2]
C = [
	0.1  0.2  0.3  0.4  0.5;
	0.2  0.3  0.4  0.3  0.2;
	0.4  0.3  0.2  0.1  0.2;
	0.3  0.2  0.1  0.2  0.5;
	0.5  0.5  0.4  0.0  0.2
]
w = [0.4, 0.6]
reg = 0.1

# utility functions
softmin(x, eps) = minimum(x) - eps * log(sum(exp.(-(x .- minimum(x)) ./ eps)))
minrow(M, reg) = map(x -> softmin(x, reg), eachslice(M, dims = 1))
mincol(M, reg) = map(x -> softmin(x, reg), eachslice(M, dims = 2))
R(C, f, g) = C - f * ones(size(g, 1))' - ones(size(f, 1)) * g'

# Algo: WDL parallel
function wdl(A, b2, C, w, reg, maxiter = 1000, zerotol = 1e-6, verbose = false)
	M = size(C, 1)
	N = size(C, 2)
	S = size(A, 2)
	U = ones(M, S)
	V = ones(N, S)
	b = zeros(N)
	K = exp.(-C / reg)
	onesN = ones(N)
	onesS = ones(S)

	iter = 0
	err = 1000.0

	while (iter < maxiter) & (err >= zerotol)
		iter += 1

		U = A ./ (K * V)
		KTU = K' * U
		b = vec(prod((KTU) .^ (onesN * w'), dims = 2))
		V = (b * onesS') ./ KTU

		err = sqrt(sum((U .* (K * V) .- A) .^ 2))
	end
	 loss = sum((b .- b2) .^ 2)

	if (verbose)
  	println("U:")
  	Base.print_matrix(stdout, U)
  	println()
  	println("V:")
  	Base.print_matrix(stdout, V)
  	println()
  	println("b:")
  	Base.print_matrix(stdout, b)
  	println()
  	#println("err: ", err)
  	#println("iter: ", iter)
  	println("loss:\n", loss)
	end

	loss
end


wdl(A, b, C, w, reg, 1000, 1e-6, true)

gradA = gradient(x -> wdl(x, b, C, w, reg, 1000, 1e-6), A)
gradw = gradient(x -> wdl(A, b, C, x, reg, 1000, 1e-6), w)

println("\ngrad_A:")
Base.print_matrix(stdout, gradA)
println("\ngrad_w:")
Base.print_matrix(stdout, gradw)
println()


# Algo: WDL log
function wdllog(A, b2, C, w, reg, maxiter = 1000, zerotol = 1e-6, verbose = false)
	M = size(C, 1)
	N = size(C, 2)
	S = size(A, 2)

	F = zeros(M, S)
	G = zeros(N, S)
	logb = zeros(N)

	iter = 0
	err = 1000.0

	while (iter < maxiter) && (err >= zerotol)
		iter += 1

		Rrowmin = reduce(hcat, map(s -> minrow(R(C, F[:, s], G[:, s]), reg), 1:S))
		F = F + reg * log.(A) + Rrowmin

		Rcolmin = reduce(hcat, map(s -> mincol(R(C, F[:, s], G[:, s]), reg), 1:S))
		logb = -G * w ./ reg - Rcolmin * w ./ reg
		G = G + reg * logb * ones(S)' + Rcolmin

		Rrowmin = reduce(hcat, map(s -> minrow(R(C, F[:, s], G[:, s]), reg), 1:S))
		err = sqrt(sum((-Rrowmin ./ reg .- log.(A)) .^ 2))
	end
	loss = sum((exp.(logb) .- b2) .^ 2)

  if (verbose)
  	println("F:")
  	Base.print_matrix(stdout, F)
  	println()
  	println("G:")
  	Base.print_matrix(stdout, G)
  	println()
  	println("b:")
  	Base.print_matrix(stdout, exp.(logb))
  	println()
  	println("loss:\n", loss)
	end

  loss
end


wdllog(A, b, C, w, reg, 1000, 1e-6, true)

gradA = gradient(x -> wdllog(x, b, C, w, reg, 1000, 1e-6), A)
gradw = gradient(x -> wdllog(A, b, C, x, reg, 1000, 1e-6), w)

println("\ngrad_A:")
Base.print_matrix(stdout, gradA)
println("\ngrad_w:")
Base.print_matrix(stdout, gradw)
println()
