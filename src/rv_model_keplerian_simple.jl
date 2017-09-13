if !isdefined(:PDMats)      using PDMats      end # Positive Definite Matrices
if !isdefined(:Optim)       using Optim       end # Optimizing over non-linear parameters
if !isdefined(:DiffBase)    using DiffBase    end # For accessing results of autodiff
if !isdefined(:ForwardDiff) using ForwardDiff end # For computing derivatives analytically
if !isdefined(:QuadGK)      using QuadGK      end # For 1-d integration

const default_tol_kepler_eqn = 1.e-8

function calc_rv_pal_one_planet{T,A<:AbstractArray{T,1}}( theta::A, time::Float64; tol::Real = default_tol_kepler_eqn, tref::Real = 300.0  )
  P = theta[1]
  K = theta[2]
  h = theta[3]
  k = theta[4]
  wpM0 = theta[5]
  ecc = sqrt(h*h+k*k)
  w = atan2(h,k)
  M0 = wpM0-w-2pi*tref/P
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

function calc_model_rv_deltas{T}(theta::Array{T,1}, times::Array{Float64}, vels::Array{Float64})
  const nparam_per_pl = 5
  const nparam_non_pl = 1
  num_pl = convert(Int64,floor((length(theta)-nparam_non_pl)//nparam_per_pl))
  logp = zero(T)
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
  delta = calc_model_rv(view(theta,1:num_pl*nparam_per_pl+1),times).-vels
  (delta,logp)
end
function loglikelihood_fixed_jitter_from_delta_Sigma{T,PDMatT<:AbstractPDMat}(delta::Array{T,1}, Sigma::PDMatT)
  chisq = dot(delta, Sigma \ delta)
  -0.5*(chisq + logdet(Sigma) +length(delta)*log(2pi) )
end
function loglikelihood_fixed_jitter{T,PDMatT<:AbstractPDMat}(theta::Array{T,1}, times::Array{Float64}, vels::Array{Float64}, Sigma::PDMatT)
   (delta,logp) = calc_model_rv_deltas(theta,times,vels)
   logp += loglikelihood_fixed_jitter_from_delta_Sigma(delta,Sigma)
end

function loglikelihood_marinalize_jitter{T}(theta::Array{T,1}, times::Array{Float64}, vels::Array{Float64}; sigma_obs::Array{Float64}=ones(length(vels)), num_Sigmas::Integer=20, Sigma_cache=make_Sigma_cache(num_Sigmas,PDMat(make_Sigma(times,sigma_obs,0.))) )
	(sigmaj_list, weights, SigmaList) = Sigma_cache
	(delta,logp) = calc_model_rv_deltas(theta,times,vels)
	marginalized_likelihood_fix_periods_sigmaj = map(k-> loglikelihood_fixed_jitter_from_delta_Sigma(delta,SigmaList[k]),1:length(sigmaj_list))
    int_norm = maximum(marginalized_likelihood_fix_periods_sigmaj);
    logp += log(dot( weights, exp.(marginalized_likelihood_fix_periods_sigmaj-int_norm) ))+int_norm
end

#=
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
  logp += logprior_kepler(theta)
  return logp
end
=#

#=
# Set priors
const Cmax = 1000.0
function logprior_offset(param)
  -log(2*Cmax)
end

const sigma_j_min = 0.0
const sigma_j_max = 99.0
const sigma_j_0 = 1.0
function logprior_jitter(sigma_j::Real)
  -log1p(sigma_j/sigma_j_0)-log(sigma_j_0)-log(log1p(sigma_j_max/sigma_j_0))
end

const Pmin = 1.25
const Pmax = 1.0e4
function logprior_period(P::Real)
   -log(P)-log(log(Pmax/Pmin))
end
function logprior_periods(P::Vector{T}) where T<:Real
   if length(P) == 0  return 0 end
   -sum(log.(P))-length(P)*log(log(Pmax/Pmin))
end

const Kmin = 0.0
const Kmax = 999.0
const K0 = 1.0
function logprior_amplitude(K::Real)
  logp = -log1p(K/K0)-log(log1p(Kmax/K0))
end

=#


arg_2d(h::Real,k::Real,sigma::Real) = exp(-0.5*(h^2+k^2)/sigma^2)
arg_1d(h::Real,sigma::Real) = QuadGK.quadgk(k->arg_2d(h,k,sigma),-sqrt(1-h^2),sqrt(1-h^2))[1]
compute_prior_ecc_norm(sigma::Real) = 1/QuadGK.quadgk(h->arg_1d(h,sigma),-1.0,1.0)[1]
sigma_ecc = 0.2
logprior_eccentricity_normalization = log(compute_prior_ecc_norm(sigma_ecc))

function logprior_eccentricty{T,A<:AbstractArray{T,1}}( theta::A )
  # logp = 0 # for e~Uniform[0,1)
  # For e~ Rayleigh(sigma_ecc) Truncated to unit disk
  h = theta[3]
  k = theta[4]
  logp = -0.5*(h^2+k^2)/sigma_ecc^2+ logprior_eccentricity_normalization
end


function logprior_eccentricty{T}( ecc::T )
  # logp = 0 # for e~Uniform[0,1)
  # For e~ Rayleigh(sigma_ecc) Truncated to unit disk
  logp = -0.5*(ecc^2)/sigma_ecc^2+ logprior_eccentricity_normalization +log(2pi)  # remove omega prior
end

function logprior_kepler{T,A<:AbstractArray{T,1}}( theta::A )
  const nparam_per_pl = 5
  const nparam_non_pl = 1
  num_pl = convert(Int64,floor((length(theta)-nparam_non_pl)//nparam_per_pl))
  @assert(num_pl>=0)
  logp = -2*num_pl*log(2pi)
  for p in 0:(num_pl-1)
    logp += logprior_period(theta[1+p*nparam_per_pl])
    logp += logprior_amplitude(theta[2+p*nparam_per_pl])
    logp += logprior_eccentricity(view(theta,1+p*nparam_per_pl:(p+1)*nparam_per_pl))
  end
  logp  += logprior_offset(theta[num_pl*nparam_per_pl+1])
  if length(theta) == num_pl*nparam_per_pl+2
    logp  += logprior_jitter(theta[num_pl*nparam_per_pl+2])
  end
  return logp
end

#=
function kernel_quaesiperiodic(dt::Float64)
  alpha_sq     = 3.0
  tau_evolve   = 50.0 
  lambda_p     = 0.5  
  tau_rotate   = 20.0 
  alpha_sq*exp(-0.5*((dt/tau_evolve)^2 + (sin(pi*dt/tau_rotate)/lambda_p)^2) )
end

function make_Sigma(t::Vector{Float64}, sigma_obs::Vector{Float64}, sigma_j::Float64)
  PDMat( [ kernel_quaesiperiodic(t[i]-t[j]) + (i==j ? sigma_obs[i]^2 + sigma_j^2 : 0.0) for i in 1:length(t), j in 1:length(t)] )
end

function make_dSigmadsigmaj(t::Vector{Float64}, sigma_obs::Vector{Float64}, sigma_j::Float64)
  ScalMat(length(t),2*sigma_j)
end

=#

function make_matrix_posdef_dumb(X::Array{T,2}) where T<:Real
   eps = 1e-9
   while !isposdef(X)
      X = X + ScalMat(size(X,1),eps) 
	  eps *= 10
   end
   PDMat(X)
end


function make_matrix_pd_eigval(A::Array{Float64,2}; epsabs::Float64 = 0.0, epsfac::Float64 = 1.0e-6)
  @assert(size(A,1)==size(A,2))
  B = copy(A)
  itt = 1
  while !isposdef(B)
	eigvalB,eigvecB = eig(B)
	neweigval = (epsabs == 0.0) ? epsfac*minimum(eigvalB[eigvalB.>0]) : epsabs
	eigvalB[eigvalB.<0] = neweigval
	B = eigvecB *diagm(eigvalB)*eigvecB'
	#println("# make_matrix_pd_eigval: ",itt,": ",eigvals(B))
	itt +=1
	if itt>size(A,1) 
	  #println("# There's a problem in make_matrix_pd:",eigvals(A),"\n")
	  error("*** make_matrix_pd_eigval ***\n")
  	  break 
	end
  end
  return B
end

function make_matrix_posdef_softabs(hessian::Array{T,2}; alpha::Float64 = 1000.0) where T<:Real
  lam, Q = eig(-hessian)
  lam_twig = lam./tanh.(alpha*lam)
  H_twig_try = full(Hermitian(Q*diagm(lam_twig)*Q'))
  if isposdef(H_twig_try) 
     H_twig = PDMat(H_twig_try)
  else
     println("# Problem in softabs: ",hessian)
	 println("# eigvals_orig =", lam)
	 println("# eigvals_new = ",eigvals(H_twig_try))
	 println("# diff = ",H_twig_try.-hessian)
     H_twig = PDMat(H_twig_try)
  end
  return H_twig
end
make_matrix_posdef = make_matrix_posdef_softabs
#make_matrix_posdef = make_matrix_pd_eigval

function laplace_approximation_kepler_model{T,PDMatT<:AbstractPDMat}(theta::Array{T,1}, times::Array{Float64}, vels::Array{Float64}, Sigma::PDMatT, loglikelihood::Function; report_log_MAP::Bool = false, verbose::Integer = 3 )
    f(x) = - ( loglikelihood(x, times, vels, Sigma)+logprior_kepler(theta) )
	td = TwiceDifferentiable(f, theta; autodiff = :forward) 
	if verbose >= 2
	   optim_opts =  Optim.Options(show_trace=true,f_tol=1e-4,time_limit=60)
	else
	   optim_opts =  Optim.Options(show_trace=false,f_tol=1e-4,time_limit=60)
	end
    res_opt = optimize(td,theta,optim_opts)
    bestfit = res_opt.minimizer
	logp_max = -res_opt.minimum
	if verbose >= 1
	   println("# best fit: ",logp_max/log(10), " : ",bestfit)
	end
    const nparam_per_pl = 5
    const nparam_non_pl = 1
    num_pl = convert(Int64,floor((length(theta)-nparam_non_pl)//nparam_per_pl))
    const nparam_no_jitter =  num_pl*nparam_per_pl+nparam_non_pl
	if report_log_MAP 
	   if length(theta) > nparam_no_jitter
	      sigmaj = theta[end]
	      logp_max += logprior_jitter(sigmaj)
	   end
	   return (logp_max,res_opt)
	end
    hess_cfg = ForwardDiff.HessianConfig(f, view(theta,1:nparam_no_jitter)) 
	hess_result = DiffBase.HessianResult(view(theta,1:nparam_no_jitter))
    #ForwardDiff.hessian!(hess_result,f,view(bestfit,1:nparam_no_jitter))
	ForwardDiff.hessian!(hess_result,f,bestfit[1:nparam_no_jitter])
	 
	logdet_hess = logdet(make_matrix_posdef(DiffBase.hessian(hess_result)))
    if verbose >= 3
 	  println("# hessian = ",DiffBase.hessian(hess_result))
	  println("# logdet(hessian) = ",logdet_hess)
	end
	logmarginal = 0.5*(length(view(theta,1:nparam_no_jitter))*log(2pi)-logdet_hess)+logp_max
	return (logmarginal,res_opt)
end

function laplace_approximation_kepler_model_jitter_separate{T,PDMatT<:AbstractPDMat}(theta::Array{T,1}, times::Array{Float64}, vels::Array{Float64}, Sigma::PDMatT; sigmaj_min::Real = 0.0, sigmaj_max::Real = 10.0, opt_jitter::Bool=true, report_log_MAP::Bool = false) 
                                                 # delta_sigmaj_factor::Real = 0.1 )
    rms = std(calc_model_rv(theta,times).-vels)
	if opt_jitter 
    SigmaTmp = Sigma + PDiagMat(rms^2*ones(length(times)))
	res_opt = laplace_approximation_kepler_model(theta,times,vels,SigmaTmp,loglikelihood_fixed_jitter,report_log_MAP=report_log_MAP)[2]
	theta = res_opt.minimizer
	function helper(sigmaj)
     SigmaTmp = Sigma + PDiagMat(sigmaj^2*ones(length(times)))
	 -(logprior_jitter(sigmaj)+laplace_approximation_kepler_model(theta,times,vels,SigmaTmp,loglikelihood_fixed_jitter,report_log_MAP=report_log_MAP)[1])
	 #-(laplace_approximation_kepler_model(theta,times,vels,SigmaTmp,loglikelihood_fixed_jitter)[1])
	end   
	res_opt = optimize(helper,sigmaj_min,sigmaj_max,show_trace=false)
	sigmaj = res_opt.minimizer
	println("# Best sigma_j = ",sigmaj)
	SigmaTmp = Sigma + PDiagMat(sigmaj^2*ones(length(times)))
	else
	   sigmaj = theta[end]
	   SigmaTmp = Sigma
    end
	(logp, res_opt) = laplace_approximation_kepler_model(theta,times,vels,SigmaTmp,loglikelihood_fixed_jitter,report_log_MAP=report_log_MAP)
    #logp += logprior_jitter(sigmaj)
	if report_log_MAP
	  return (logp, res_opt)
	end
	theta = res_opt.minimizer
	dSigmadJ = diagm(2*sigmaj*ones(length(times))) 
	d2logpdJ2 = 0.5*trace((SigmaTmp \ dSigmadJ)^2)
	# println("# logdetjitter (ana) = ",logdetjitter)
    #=
    delta_sigmaj = delta_sigmaj_factor*sigmaj
	SigmaTmp = Sigma + PDiagMat((sigmaj+delta_sigmaj)^2*ones(length(times)))
	logp_hi = laplace_approximation_kepler_model(theta,times,vels,SigmaTmp,loglikelihood_fixed_jitter)[1]
	logp_hi += logprior_jitter(sigmaj+delta_sigmaj)
	SigmaTmp = Sigma + PDiagMat((sigmaj-delta_sigmaj)^2*ones(length(times)))
    logp_lo = laplace_approximation_kepler_model(theta,times,vels,SigmaTmp,loglikelihood_fixed_jitter)[1]
    logp_lo += logprior_jitter(sigmaj-delta_sigmaj)
	d2logpdJ2 = (logp_hi-2*logp+logp_lo)/delta_sigmaj^2
	println("# logp = ",logp)
	println("# logp_hi = ",logp_hi)
	println("# logp_lo = ",logp_lo)
    println("# d^2 logp/dsigmaj^2 = ",d2logpdJ2/log(10))
	println("# logdetjitter (num) = ",log(abs(d2logpdJ2)))
	=#
    log_int_over_sigma = logp + 0.5*(log(2pi)-log(abs(d2logpdJ2)))
	(log_int_over_sigma,res_opt)
end

function laplace_approximation_kepler_model_jitter_separate_gauss{T,PDMatT<:AbstractPDMat}(theta::Array{T,1}, times::Array{Float64}, vels::Array{Float64}, Sigma0::PDMatT, num_Sigmas::Integer = 30; Sigma_cache = make_Sigma_cache(num_Sigmas,Sigma0) ) 
    
	(sigmaj_list, weights, SigmaList) = Sigma_cache
	 X = ones(length(times),1)
     const nparam = size(X,2)
    
	   marginalized_likelihood_fix_periods_sigmaj = map(k->laplace_approximation_kepler_model(theta,times, vels, SigmaList[k],loglikelihood_fixed_jitter)[1],1:length(sigmaj_list))
	   
       int_norm = maximum(marginalized_likelihood_fix_periods_sigmaj);
       logp = log(dot( weights, exp.(marginalized_likelihood_fix_periods_sigmaj-int_norm) ))+int_norm

	#(log_int_over_sigma,res_opt)
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