using DiffBase
using ForwardDiff
using PDMats
using Optim

const default_tol_kepler_eqn = 1.e-8

function calc_rv_pal_one_planet{T,A<:AbstractArray{T,1}}( theta::A, time::Float64; tol::Real = default_tol_kepler_eqn  )
  P = theta[1]
  K = theta[2]
  h = theta[3]
  k = theta[4]
  M0 = theta[5]
  ecc = sqrt(h*h+k*k)
  w = atan2(k,h)
  n = 2pi/P
  M = time*n-M0
  #M = mod2pi(M)
  Mdiv2pi = floor(M/(2pi))
  M -= 2pi*Mdiv2pi
  lambda = w+M
  #E = ecc_anom_itterative_laguerre(M,ecc,tol=tol)   # Kepler equation solve integrated into this function due to limitations of ForwardDiff
  if(M<zero(M)) M += 2pi end
  E = (M<pi) ? M + 0.85*ecc : M - 0.85*ecc;
  const max_its_laguerre = 200
  local E_old
  for i in 1:max_its_laguerre
       E_old = E
       es = ecc*sin(E)
       ec = ecc*cos(E)
	   F = (E-es)-M
       Fp = 1.0-ec
       Fpp = es
       const n = 5
       root = sqrt(abs((n-1)*((n-1)*Fp*Fp-n*F*Fpp)))
       denom = Fp>zero(Fp) ? Fp+root : Fp-root
       E = E-n*F/denom
       
       if abs(E-E_old)<tol break end
  end
  @assert abs(E-E_old)<tol
  
  c = cos(lambda+ecc*sin(E))
  s = sin(lambda+ecc*sin(E))
  if ecc >= 1.0
    println("# ERROR in calc_rv_pal_one_planet: ecc>=1.0:  ",theta)
  end
  @assert(0.0<=ecc<1.0)
  j = sqrt((1.0-ecc)*(1.0+ecc))
  p, q = (ecc == 0.0) ? (zero(T), zero(T)) : (ecc*sin(E), ecc*cos(E))
  a = K/(n/sqrt((1.0-ecc)*(1.0+ecc)))
  zdot = a*n/(1.0-q)*( cos(lambda+p)-k*q/(1.0+j) )
end

function calc_model_rv{T,A<:AbstractArray{T,1}}( theta::A, time::Float64; tol::Real = default_tol_kepler_eqn )
  const nparam_per_pl = 5
  const nparam_non_pl = 1
  num_pl = convert(Int64,floor((length(theta)-nparam_non_pl)//nparam_per_pl))
  v = zero(T)
  for p in 1:num_pl
     P = theta[1+(p-1)*nparam_per_pl]
     K = theta[2+(p-1)*nparam_per_pl]
     h = theta[3+(p-1)*nparam_per_pl]
	 k = theta[4+(p-1)*nparam_per_pl]
	 ecc = h*h+k*k
	 if P>zero(P) && K>=zero(K) && ecc<one(ecc)
	    #v += calc_rv_pal_one_planet(theta[1+(p-1)*nparam_per_pl:p*nparam_per_pl],time)
		v += calc_rv_pal_one_planet(view(theta,1+(p-1)*nparam_per_pl:p*nparam_per_pl),time)
     end
  end
  rv_offset = theta[num_pl*nparam_per_pl+1]
  v += rv_offset
  return v
end

function calc_model_rv{T,A<:AbstractArray{T,1}}( theta::A, times::Array{Float64}; tol::Real = default_tol_kepler_eqn )
  v = map(t->calc_model_rv(theta,t),times)
end

#=
time = 0.0
f(x) = calc_rv_pal_one_planet(x,time)
param = [12.0,5.0,0.1,0.1,pi/4]
f(param)
ForwardDiff.gradient(f,param)
cfg = ForwardDiff.GradientConfig(f, param)
ForwardDiff.gradient(f,param,cfg)
result = DiffBase.GradientResult(param)
ForwardDiff.gradient!(result,f,param,cfg)
@time ForwardDiff.gradient(f,param)
@time ForwardDiff.gradient(f,param,cfg)
@time ForwardDiff.gradient!(result,f,param,cfg)
f(x) = calc_model_rv(x,time)
param = [12.0,5.0,0.1,0.1,pi/4,  30.0,2.0,0.1,0.1,pi/4,  0.0, 0.0]
f(param)

times = collect(linspace(0.0,100.0,10));
f(x) = calc_model_rv(x,times)
ForwardDiff.gradient(f,param)
ForwardDiff.jacobian(f,param)
=#

function loglikelihood_fixed_jitter{T,PDMatT<:AbstractPDMat}(theta::Array{T,1}, times::Array{Float64}, vels::Array{Float64}, Sigma::PDMatT)
  const nparam_per_pl = 5
  const nparam_non_pl = 1
  num_pl = convert(Int64,floor((length(theta)-nparam_non_pl)//nparam_per_pl))
  logp = 0
  for p in 1:num_pl
     P = theta[1+(p-1)*nparam_per_pl]
	 if P<0.0
	   logp -= 1000*length(times)*P^2
	 end
     K = theta[2+(p-1)*nparam_per_pl]
	 if K<0.0
	   logp -= 1000*length(times)*K^2
	 end
	 h = theta[3+(p-1)*nparam_per_pl]
	 k = theta[4+(p-1)*nparam_per_pl]
	 ecc = h*h+k*k
	 if ecc>=1.0
	   logp -= 1000*length(times)*(ecc-1)^2
	 end
  end
  delta = calc_model_rv(view(theta,1:(length(theta)-1)),times).-vels
  chisq = dot(delta, Sigma \ delta)
  logp -= 0.5*(chisq + logdet(Sigma) +length(times)*log(2pi) )
  return logp
end

function loglikelihood_add_whitenoise{T,PDMatT<:AbstractPDMat}(theta::Array{T,1}, times::Array{Float64}, vels::Array{Float64}, Sigma::PDMatT)
  const nparam_per_pl = 5
  const nparam_non_pl = 2
  num_pl = convert(Int64,floor((length(theta)-nparam_non_pl)//nparam_per_pl))
  logp = 0
  for p in 1:num_pl
     P = theta[1+(p-1)*nparam_per_pl]
	 if P<0.0
	   logp -= 1000*length(times)*P^2
	 end
     K = theta[2+(p-1)*nparam_per_pl]
	 if K<0.0
	   logp -= 1000*length(times)*K^2
	 end
	 h = theta[3+(p-1)*nparam_per_pl]
	 k = theta[4+(p-1)*nparam_per_pl]
	 ecc = h*h+k*k
	 if ecc>=1.0
	   logp -= 1000*length(times)*(ecc-1)^2
	 end
  end
  delta = calc_model_rv(view(theta,1:(length(theta)-1)),times).-vels
  sigmaj = theta[end]
  SigmaNew = Sigma + PDiagMat(sigmaj^2*ones(length(times)))
  chisq = dot(delta, SigmaNew \ delta)
  logp -= 0.5*(chisq + logdet(SigmaNew) + length(times)*log(2pi) )
  return logp
end

function laplace_approximation_kepler_model{T,PDMatT<:AbstractPDMat}(theta::Array{T,1}, times::Array{Float64}, vels::Array{Float64}, Sigma::PDMatT, loglikelihood::Function )
    f(x) = -loglikelihood(x, times, vels, Sigma)
	td = TwiceDifferentiable(f, theta; autodiff = :forward) 
	optim_opts =  Optim.Options(show_trace=true,f_tol=1e-4)
    res_opt = optimize(td,theta,optim_opts)
    bestfit = res_opt.minimizer
	logp_max = -res_opt.minimum
	println("# best fit: ",logp_max, " : ",bestfit)
	flush(STDOUT)
    hess_cfg = ForwardDiff.HessianConfig(f, theta) 
	hess_result = DiffBase.HessianResult(theta)
    ForwardDiff.hessian!(hess_result,f,bestfit)
    println("# hessian = ",DiffBase.hessian(hess_result))
	flush(STDOUT)
	logdet_hess = logdet(DiffBase.hessian(hess_result))
    println("# logdet(hessian) = ",logdet_hess)
	flush(STDOUT)
	0.5*(length(theta)*log(2pi)-logdet_hess+logp_max)
end

#=
param = [12.0,15.0,0.1,0.1,pi/4,  30.0,10.0,-0.1,-0.1,pi*3/4,  0.0, 1.0]
nobs = 100
times = collect(linspace(0.0,100.0,nobs));
vels = calc_model_rv(param,times) + 1.0*randn(size(times))
Sigma = PDiagMat(ones(length(vels)))
loglikelihood_add_whitenoise(param, times, vels, Sigma)

laplace_approximation_kepler_model(param,times,vels,Sigma,loglikelihood_add_whitenoise)/log(10)
=#


#=
f(x) = -loglikelihood_add_whitenoise(param, times, vels, Sigma)
f(param)

grad_result = DiffBase.GradientResult(param);
hess_result = DiffBase.HessianResult(param);
grad_cfg = ForwardDiff.GradientConfig(f, param);
hess_cfg = ForwardDiff.HessianConfig(f, param);
ForwardDiff.gradient!(grad_result,f,param)
ForwardDiff.hessian!(hess_result,f,param)

@time f(param)
@time ForwardDiff.gradient!(grad_result,f,param)
@time ForwardDiff.hessian!(hess_result,f,param)


optim_opts = Optim.Options(show_trace=true,f_tol=1e-4);
optimize(f,param,optim_opts)

#od = OnceDifferentiable(f, param; autodiff = :forward)
#@time res_opt = optimize(od,param,optim_opts)

td = TwiceDifferentiable(f, param; autodiff = :forward)
@time res_opt = optimize(td,param,optim_opts)
@time ForwardDiff.hessian!(hess_result,f,res_opt.minimizer)
DiffBase.hessian(hess_result)
DiffBase.gradient(hess_result)
DiffBase.value(hess_result)

=#