using Test
using LinearAlgebra
using Interpolations
using PyPlot
using Printf
function target_score(x, m, s, w, k)
	return -6.0
	prob = mixture_prob(x, m, s, w, k)
	score = 0.0
	for n = 1:k
			score -= w[n]*gaussian(x,m[n],s[n])*(x-m[n])/s[n]/s[n]
	end
	score = 1/prob*score
end
function gaussian(x, m, s)
	prob = (x-m)^2/2/s/s
	prob = exp(-prob)
	return prob/sqrt(2*π)/s
end
function mixture_prob(x, m, s, w, k)
	prob = 0
	for n = 1:k
		prob += w[n]*gaussian(x, m[n], s[n])
	end
	return prob
end
function source_score(x, m, s)
	#return -(x-m)/s/s
	#return 0.0
	return -6.0
end
function dtarget_score(x, m, s, w, k)
	return 8.0
	prob = mixture_prob(x,m,s,w,k)
	score = target_score(x,m,s,w,k)
	dscore = 0.0
	for n = 1:k
			dscore += gaussian(x, m[n], s[n])*w[n]*((x-m[n])^2/s[n]^4 - 1/s[n]^2)
	end
	dscore /= prob
	dscore -= score*score
	return dscore
end
function test_target_score(x,m,s,w,k)
	eps = 1.e-5
	score = target_score(x,m,s,w,k)
	score_fd = (log(mixture_prob(x+eps,m,s,w,k)) - log(mixture_prob(x-eps,m,s,w,k)))/(2*eps)
	@show score, score_fd
	@test score≈score_fd atol=1.e-6
end
function test_dtarget_score(x,m,s,w,k)
	eps = 1.e-5
	score1 = target_score(x+eps,m,s,w,k)
	score2 = target_score(x-eps,m,s,w,k)
	dscore_fd = (score1 - score2)/(2*eps)
	dscore= dtarget_score(x,m,s,w,k)
	@show "d scores from fd, analytical", dscore_fd, dscore
	@test dscore_fd ≈ dscore atol=1.e-8
end
function transport_by_kam(N,K,r,mref,sref,m,s,w,nm,v0,vn)
	x_gr = Array(LinRange(-r,r,N+1))
	dx = x_gr[2] - x_gr[1]
		
	x_gr = x_gr .+ dx/2
	x_gr = x_gr[1:N]	
	x = x_gr[2:N-1]
	Tx = zeros(N)

	dxinv = 1/dx
	dx2inv = dxinv*dxinv

	A = zeros(N-2,N-2)	
	b = zeros(N-2)
	parr = zeros(N-2)
	qarr = zeros(N-2)
	dqarr = zeros(N-2)
	v = zeros(N)
	v[1] = v0
	v[N] = vn
	vp = zeros(N)
	vpp = zeros(N)
	Tx = zeros(N)
	x_temp = zeros(N-2)
	for n = 1:N-2
		parr[n] = source_score(x[n], mref, sref)
		qarr[n] = target_score(x[n],m,s,w,nm)
		dqarr[n] = dtarget_score(x[n],m,s,w,nm)
	end
	for k = 1:K
		for n = 1:N-2
			vp[n+1] = (v[n+2]-v[n])*dxinv/2.0
			vpp[n+1] = (-2*v[n+1] + v[n+2] + v[n])*dx2inv
			parr[n] = parr[n]/(1 + vp[n+1]) - vpp[n+1]/(1 + vp[n+1])^2
			x_temp[n] = x_gr[n+1] + v[n+1]
		end
		order = sortperm(x_temp)
		x_temp = x_temp[order]
		parr = parr[order]
		parr_int = linear_interpolation(x_temp, parr,extrapolation_bc=Line())
		v_int = linear_interpolation(x_gr, v, extrapolation_bc=Line())
		x = x .+ v_int.(x)
		for n = 1:N-2
			p = parr_int(x_gr[n+1])			
			q = qarr[n] 			
			dq = dqarr[n]
			b[n] = p - q
			if n < N-2
				A[n, n+1] += dx2inv
				A[n, n+1] += p*dxinv/2
			end
			if n > 1
				A[n, n-1] += dx2inv
				A[n, n-1] += -p*dxinv/2
			end
			if n == 1
				b[n] -= (v0*dx2inv + p/2*dxinv*v0)
			end
			if n == N-2
				b[n] -= (vn*dx2inv - p/2*dxinv*vn)
			end
			A[n, n] = dq - 2.0*dx2inv

		end
		vint = A\b
		v[2:N-1] .= vint
		@printf "At k= %d, ||v|| = %f \n" k norm(v)
	end
	return v
end
function sample_target(N, m, s, w, K)
	x = zeros(N)
	order = sortperm(w)
	w = w[order]
	m = m[order]
	s = s[order]
	c = K
	for n = 1:N
		u = rand()
		c = K
		for k = 1:K
			if u < w[k]
				c = k
				break
			end
		end
		x[n] = m[c] + s[c]*randn()
	end
	return x
end
w1, w2 = 0.5, 0.4
w3 = 1 - (w1 + w2)
m1, m2, m3 = 0.0, 1.0, 2.0
s1, s2, s3 = 1.0, 0.5, 2.0
w = [w1, w2, w3]
m = [m1, m2, m3]
s = [s1, s2, s3]
#fig, ax = subplots()
#ax.xaxis.set_tick_params(labelsize=30)
#ax.yaxis.set_tick_params(labelsize=30)
k = 3
#x = sample_target(2000,m,s,w,k)
#ax.hist(x,bins=100,density=true)
#Tx = transport_by_kam(2000,3,10,1,1,m,s,w,k,0,0)
#ax.hist(Tx,bins=100,density=true)
r = 1
M = 5000
x_gr = Array(LinRange(-r,r,M))
v0, vn = exp(-1)*exp(-1), exp(1)*exp(1)
fig, ax = subplots()
ax.xaxis.set_tick_params(labelsize=30)
ax.yaxis.set_tick_params(labelsize=30)
ax.plot(x_gr, exp.(2*x_gr), ".", label="Analytical", ms=10)
Tx = transport_by_kam(M,1,r,1,1,m,s,w,k,v0,vn)
ax.plot(x_gr, Tx, "^", label="Transport", ms=10)
ax.legend(fontsize=30)
@show norm(Tx - exp.(2*x_gr))
