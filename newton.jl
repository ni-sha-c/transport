using Test
using LinearAlgebra
function target_score(x, m, v, w, k)
	prob = mixture_prob(x, m, v, w, k)
	score = 0.0
	for n = 1:k
			score -= w[n]*gaussian(x,m[n],v[n])*(x-m[n])/v[n]/v[n]
	end
	score = 1/prob*score
end
function gaussian(x, m, v)
	prob = (x-m)^2/2/v/v
	prob = exp(-prob)
	return prob/sqrt(2*π)/v
end
function mixture_prob(x, m, v, w, k)
	prob = 0
	for n = 1:k
		prob += w[n]*gaussian(x, m[n], v[n])
	end
	return prob
end
function source_score(x, m, v)
	return -(x-m)/v/v
end
function dtarget_score(x, m, v, w, k)
	prob = mixture_prob(x,m,v,w,k)
	score = target_score(x,m,v,w,k)
	dscore = 0.0
	for n = 1:k
			dscore += gaussian(x, m[n], v[n])*w[n]*((x-m[n])^2/v[n]^4 - 1/v[n]^2)
	end
	dscore /= prob
	dscore -= score*score
	return dscore
end
function test_target_score(x,m,v,w,k)
	eps = 1.e-5
	score = target_score(x,m,v,w,k)
	score_fd = (log(mixture_prob(x+eps,m,v,w,k)) - log(mixture_prob(x-eps,m,v,w,k)))/(2*eps)
	@show score, score_fd
	@test score≈score_fd atol=1.e-6
end
function test_dtarget_score(x,m,v,w,k)
	eps = 1.e-5
	score1 = target_score(x+eps,m,v,w,k)
	score2 = target_score(x-eps,m,v,w,k)
	dscore_fd = (score1 - score2)/(2*eps)
	dscore= dtarget_score(x,m,v,w,k)
	@show "d scores from fd, analytical", dscore_fd, dscore
	@test dscore_fd ≈ dscore atol=1.e-8
end
function solve_newton_step(N,r,mref,vref,m,v,w,k)
	x_gr = LinRange(-r,r,N+1)
	dx = x_gr[2] - x_gr[1]
	x_gr = x_gr[1:N]
	x_gr .+= dx/2

	dxinv = 1/dx
	dx2inv = dxinv*dxinv

	A = (-2.0*dx2inv)*I(N-2)
	b = zeros(N-2)
	x = x_gr[2:N-1]
	for n = 1:N-2
		p = source_score(x[n], mref, vref)
		q = target_score(x[n],m,v,w,k)
		dq = dtarget_score(x[n],m,v,w,k)

		b[n] = p - q
		if n < N-1
			A[n, n+1] += dx2inv
			A[n, n+1] += p*dxinv/2
		end
		if n > 1
			A[n, n-1] += dx2inv
			A[n, n-1] += -p*dxinv/2
		end
		A[n, n] += dq
	end
	vint = A\b
	v = zeros(N)
	v[2:N-1] .= vint
	return v
end

w1, w2 = 0.5, 0.4
w3 = 1 - (w1 + w2)
m1, m2, m3 = 0.0, 1.0, 2.0
v1, v2, v3 = 1.0, 0.5, 2.0
w = [w1, w2, w3]
m = [m1, m2, m3]
v = [v1, v2, v3]
x = rand()

