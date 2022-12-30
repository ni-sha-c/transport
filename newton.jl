using Test
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

w1, w2 = 0.5, 0.4
w3 = 1 - (w1 + w2)
m1, m2, m3 = 0.0, 1.0, 2.0
v1, v2, v3 = 1.0, 0.5, 2.0
w = [w1, w2, w3]
m = [m1, m2, m3]
v = [v1, v2, v3]
x = rand()
@show "Score of a gaussian at x = ", x, " is ", source_score(x, m1, v1)
@show "Score of a gaussian at x = ", x, " is ", target_score(x, m, v, w, 3)


