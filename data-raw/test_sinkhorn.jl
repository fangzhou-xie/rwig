
# calculate the gradients using Julia ForwardDiff

using LinearAlgebra
using ForwardDiff: gradient, jacobian

a = [0.3, 0.4, 0.1, 0.1, 0.1]
b = [0.4, 0.5, 0.1]
C = [1 2 3; 2 3 4; 4 3 2; 3 2 1; 5 5 4] ./ 10
reg = 0.1

A = [0.3 0.4; 0.7 0.6]
B = [0.4 0.7; 0.5 0.1; 0.1 0.2]
w = [0.5, 0.5]

# utility functions
softmax(x) = exp.(x .- maximum(x)) ./ sum(exp.(x .- maximum(x)))
softmin(x, eps) = minimum(x) - eps * log(sum(exp.(-(x .- minimum(x)) ./ eps)))
minrow(M, reg) = map(x -> softmin(x, reg), eachslice(M, dims = 1))
mincol(M, reg) = map(x -> softmin(x, reg), eachslice(M, dims = 2))
Rf(C, f, g) = C - f * ones(size(g, 1))' - ones(size(f, 1)) * g'

# Algo 3.1
function sinkhorn_vanilla(a, b, C, reg, maxiter = 1000, zerotol = 1e-6,
                          verbose = false)
	M = size(a, 1)
	N = size(b, 1)
	u = ones(M)
	v = ones(N)

	K = exp.(-C ./ reg)

	iter = 0
	err = 1000.0
	while ((iter < maxiter) & (err >= zerotol))
		iter = iter + 1

		u = a ./ (K * v)
		v = b ./ (K' * u)

		err = sqrt(sum((u .* (K * v) - a) .^ 2)) + sqrt(sum((v .* (K' * u) - b) .^ 2))
	end
	# println(iter)
	P = diagm(u) * K * diagm(v)
	# loss
	loss = sum(P .* C) + reg * sum(P .* (log.(P) .- 1))

  if (verbose)
    println("\nP:")
  	Base.print_matrix(stdout, P)
  	println("\nu:")
  	Base.print_matrix(stdout, u)
  	println("\nv:")
  	Base.print_matrix(stdout, v)
  	println("\nloss:\n", loss)
	end

  loss
end

println("testing the sinkhorn vanilla with gradient")

sinkhorn_vanilla(a, b, C, reg, 1000, 1e-6, true)

grad_a = gradient(x -> sinkhorn_vanilla(x, b, C, reg, 1000, 1e-6, false), a)

println("\ngrad_a:")
Base.print_matrix(stdout, grad_a)
println()


# Algo 3.3
function sinkhorn_log(a, b, C, reg, maxiter = 1000, zerotol = 1e-6,
                      verbose = false)
	M = size(a, 1)
	N = size(b, 1)
	f = zeros(M)
	g = zeros(N)
	onesM = ones(M)
	onesN = ones(N)
	P = zeros(M, N)

	iter = 0
	err = 1000.0

	R = C - f * onesN' - onesM * g'

	while ((iter < maxiter) && (err >= zerotol))
		iter += 1

		f = f + reg * log.(a) + minrow(R, reg)

		R = C - f * onesN' - onesM * g'
		g = g + reg * log.(b) + mincol(R, reg)

		R = C - f * onesN' - onesM * g'
		err = sqrt(sum((-minrow(R, reg) / reg - log.(a)) .^ 2)) + sqrt(sum((-mincol(R, reg) / reg - log.(b)) .^ 2))
	end

	R = C - f * onesN' - onesM * g'
	P = exp.(-R / reg)
	# loss
	loss = sum(P .* C) + reg * sum(P .* (log.(P) .- 1))

  if (verbose)
    println("\nP:")
  	Base.print_matrix(stdout, P)
  	println("\nf:")
  	Base.print_matrix(stdout, f)
  	println("\ng:")
  	Base.print_matrix(stdout, g)
  	println("\nloss:\n", loss)
	end

  loss
end


println("testing the sinkhorn log with gradient")

sinkhorn_log(a, b, C, reg, 1000, 1e-6, true)

grad_a = gradient(x -> sinkhorn_log(x, b, C, reg, 1000, 1e-6, false), a)

println("\ngrad_a:")
Base.print_matrix(stdout, grad_a)
println()

