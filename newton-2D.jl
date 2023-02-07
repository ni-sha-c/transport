using Test
using LinearAlgebra
using Interpolations
using PyPlot
using Printf
using Polynomials
using SpecialFunctions
using Distributions
function target_score(x, m, s, w, k)
	#return -6.0
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
	return -(x-m)/s/s
	#return 0.0
end
function dtarget_score(x, m, s, w, k)
	#return 8.0
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
function transport_by_kam_fd(N,K,r,mref,sref,m,s,w,nm,v0,vn)
	x_gr = Array(LinRange(-r,r,N+1))
	dx = x_gr[2] - x_gr[1]
		
	x_gr = x_gr .+ dx/2
	x_gr = x_gr[1:N]	
	x = mref .+ sref*randn(N)
	Tinvx = copy(x)

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
		parr[n] = source_score(x_gr[n+1], mref, sref)
		qarr[n] = target_score(x_gr[n+1],m,s,w,nm)
		dqarr[n] = dtarget_score(x_gr[n+1],m,s,w,nm)
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
		
		#v_int = Polynomials.polyfitA(x_gr,v,2)
		v_int = linear_interpolation(x_gr, v, extrapolation_bc=Line())
		x = x .+ v_int.(x)
		for n = 1:N-2
			p = parr_int(x_gr[n+1])			
			q = qarr[n] 			
			dq = dqarr[n]
			b[n] = p - q
			@show p
			if n < N-2
					A[n, n+1] += (dx2inv + p*dxinv/2)
			end
			if n > 1
					A[n, n-1] += (dx2inv -p*dxinv/2)
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
	return Tinvx, x
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
function cheb_pts(N)
	x = zeros(N+1)
	for j = 1:N+1
		x[j] = cos((j-1)*π/N)	
	end
	return x
end
function cheb_diff(N)
	x = cheb_pts(N)
	D = zeros(N+1,N+1)
	for j = 2:N 
		xj = x[j]
		D[j,j] = -xj/2/(1-xj^2)
		for i = 2:N
			if !(i==j)
				xij = (-1)^(i+j)/(x[i] - xj)
				D[i,j] = xij
			end
		end
	end
	i = 1
	for j = 2:N
		xj = x[j]
		ij =  2.0*(-1)^(i+j)
		D[i, j] = ij/(1-xj)
		D[i+N, j] = -(-1)^N*ij/(1+xj)
	end
	j = 1
	for i = 2:N
		xi = x[i]
		ij =  (-1)^(i+j)/2.0
		D[i, j] = ij/(xi - 1)
		D[i, j+N] = (-1)^N*ij/(xi+1)
	end
	D[1,N+1] = (-1)^N/2
	D[N+1,1] = -(-1)^N/2


	nsqby6 = (2*N*N + 1)/6
	D[1,1] = nsqby6
	D[N+1, N+1] = -nsqby6

	return D
end
function transport_by_kam(N,K,r,mref,sref,m,s,w,nm,v0,vn)
	x_gr = cheb_pts(N-1)
	x = mref .+ sref.*randn(50*N)
	#x = -r .+ 2*r*rand(50*N)
	Tinvx = copy(x)

	A = zeros(N, N)	
	b = zeros(N)
	parr = zeros(N)
	qarr = zeros(N)
	dqarr = zeros(N)
	v = zeros(N)
	v[1] = v0
	v[N] = vn
	vp = zeros(N)
	vpp = zeros(N)
	x_temp = zeros(N)
	for n = 1:N
		parr[n] = source_score(x_gr[n], mref, sref)
		qarr[n] = target_score(x_gr[n],m,s,w,nm)
		dqarr[n] = dtarget_score(x_gr[n],m,s,w,nm)
	end
	D = cheb_diff(N-1)
	D2 = D*D
	#D = D_big[2:(N+1),2:(N+1)]
	#D2 = D2_big[2:(N+1),2:(N+1)]
	for k = 1:K
		vp .= D*v
		vpp .= D2*v
		for n = 1:N
			parr[n] = parr[n]/(1 + vp[n]) - vpp[n]/(1 + vp[n])^2
			x_temp[n] = x_gr[n] + v[n]
		end
		order = sortperm(x_temp)
		x_temp = x_temp[order]
		parr .= parr[order]
		p_int = linear_interpolation(x_temp,parr,extrapolation_bc=Line())
		v_int = Polynomials.polyfitA(x_gr,v,1)
		#v_coeff = Polynomials.coeffs(v_int)
		x .= x .+ v_int.(x)
		@printf "x_max = %f and x_min = %f" maximum(x) minimum(x)
		parr .= p_int.(x_gr) 
		A .= D2  
		for n = 2:(N-1)
			p = parr[n]
			q = qarr[n] 			
			dq = dqarr[n]
			b[n] = p - q
			A[n,:] .+= p.*D[n,:]
			A[n,n] += dq	
		end
		#@show "max b", maximum(b), "min b", minimum(b)
		vint = A\b
		v .= vint
		@printf "At k= %d, ||v|| = %f \n" k norm(v)
	end
	return Tinvx, x
end
function to_mixture(x,mref,sref,m,s,w,k)
	u = rand()
	wc = cumsum(w)
	a, b = 1.0, 1.0
	for n = 1:k
		if u < wc[n] 
				a = s[n]/sref
			b = m[n] - mref*a
			break
		end
	end
	return a*x + b
end
function kr_mixture(x,m_s,s_s,m,s,w,k)
	st2 = sqrt(2)
	cdfx = (1/2)*(1 + erf((x-m_s)/s_s/st2))
	inv_cdf(x, mu, si) = mu + st2*si*erfinv(x-1)
	wc = cumsum(w)
	Tx = 0.0
	for n = 1:k
		if cdfx < wc[n]
			Tx = inv_cdf(cdfx,m[n],s[n])
			break
		end
	end
	return Tx
end
function kr_two_mixture_general(x,m_s,s_s,m,s,w)
	w1, w2 = w[1], w[2]
	a = m[1] + 3*s[1]
	st2 = sqrt(2)
	cdf(x,mu,si) = (1/2)*(1 + erf((x-mu)/si/st2))
	inv_cdf(x, mu, si) = mu + st2*si*erfinv(2*x-1)
	w1cdf1a, w2cdf2a = w1*cdf(a,m[1],s[1]), w2*cdf(a,m[2],s[2])
	Tx = 0.0
	p = cdf(x,m_s,s_s)
	if p <= w1cdf1a
		b = p/w1
		if b > 1
			b = 1
		end
		Tx = inv_cdf(b, m[1], s[1])
	else
		b = (p-w1cdf1a+w2cdf2a)/w2
		if b > 1
			b = p - 1.e-6
		end
		if b < 0
			b = 0
		end
		Tx = inv_cdf(b, m[2], s[2])
	end
	return Tx
end

function kr_two_mixture(x,m_s,s_s,m,s,w)
	w1, w2 = w[1], w[2]
	a = m[1] + 3*s[1]
	st2 = sqrt(2)
	cdf(x,mu,si) = (1/2)*(1 + erf((x-mu)/si/st2))
	inv_cdf(x, mu, si) = mu + st2*si*erfinv(2*x-1)
	w1cdf1a, w2cdf2a = w1*cdf(a,m[1],s[1]), w2*cdf(a,m[2],s[2])
	Tx = 0.0
	p = cdf(x,m_s,s_s)
	if p <= w1cdf1a
		b = p/w1
		if b > 1
			b = 1
		end
		Tx = inv_cdf(b, m[1], s[1])
	else
		b = (p-w1cdf1a+w2cdf2a)/w2
		if b > 1
			b = p - 1.e-6
		end
		if b < 0
			b = 0
		end
		Tx = inv_cdf(b, m[2], s[2])
	end
	return Tx
end
#=
w1, w2 = 0.5, 0.5
w3 = 1 - (w1 + w2)
m1, m2, m3 = -0.5, 0.5, 2.0
s1, s2, s3 = 0.1, 0.1, 0.1
w = [w1, w2, w3]
m = [m1, m2, m3]
s = [s1, s2, s3]

fig, ax = subplots()
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
k = 3
r = -1
M = 64
K = 2
v0 = 0.0
vn = 0.0
x_t = sample_target(10000,m,s,w,k)
#x_t = -1.0 .+ 2.0*rand(10000)
m_s = 0
s_s = 0.5

ax.hist(x_t,bins=200,density=true,label="Target")
x, Tx = transport_by_kam(M,K,r,m_s,s_s,m,s,w,k,v0,vn)
N = size(x)[1]
Tx_ana = zeros(N)
Tx_ana_fit = zeros(N)
a = 0.2 
b = -0.5
c = 1/100.0
st2 = sqrt(2)
cdf(x,mu,si) = (1/2)*(1 + erf((x-mu)/si/st2))
for n = 1:N
	Tx_ana[n] = kr_two_mixture(x[n], m_s, s_s, m, s, w)
	Tx_ana_fit[n] = cdf(x[n],m_s,s_s/5) + (b + a*x[n])
end
ax.hist(Tx,bins=200,density=true,label="KAM")
ax.hist(Tx_ana,bins=200,density=true,label="KR Target")
ax.legend(fontsize=20)
ax.set_xlabel("x",fontsize=20)
ax.set_title("Density",fontsize=20)
tight_layout()
=#
μ1 = [-0.5, 0]
Σ1 = [1  0.01;
     0.01 0.1]
μ2 = [0.5, 0]
Σ2 = [0.1  0;
     0 2]

p1 = MvNormal(μ1, Σ1)
p2 = MvNormal(μ2, Σ2)
Nplot = 200
X = range(-2, 2, length=Nplot)
Y = range(-2, 2, length=Nplot)
f1(x,y) = pdf(p1, [x,y])
f2(x,y) = pdf(p2, [x,y])
Z = zeros(Nplot, Nplot)
for i = 1:Nplot
  for j = 1:Nplot
    Z[i,j] = 0.5*(f1(X[j], Y[i]) + f2(X[j], Y[i]))
  end
end
fig, ax = subplots()
ax.contourf(X, Y, Z)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.grid(true)
#=
fig, ax = subplots()
ax.plot(x, Tx, ".", ms=5, label="KAM-Cheb")
ax.plot(x, Tx_ana, "P", ms=5, label="Analytical")
ax.plot(x, Tx_ana_fit, "o", ms=5, label="Analytical Fit")
ax.set_xlabel("x", fontsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.legend(fontsize=20)
ax.grid(true)
tight_layout()

=#
