using PDMats  # Positive Definite Matrices
using QuadGK  # 1-d integration
using Cuba    # multi-d integration
using Optim   # Optimizing over non-linear parameters
using FastGaussQuadrature
using JLD

include("../src/rv_model_keplerian_simple.jl")

# Set priors
const Cmax = 1000.0
function logprior_offset(param)
  -log(2*Cmax)
end

const sigma_j_min = 0.0
const sigma_j_max = 99.0
const sigma_j_0 = 1.0
function logprior_jitter(sigma_j::Real)
  -log1p(sigma_j/sigma_j_0)-log(log1p(sigma_j_max/sigma_j_0)) # -log(sigma_j_0)
end

const Pmin = 1.0
const Pmax = 1.0e4
function logprior_period(P::Real)
   -log(P)-log(log(Pmax/Pmin))
end
function logprior_periods(P::Vector{T}) where T<:Real
   if length(P) == 0  return 0 end
   if any(P.<=zero(P)) return 0 end
   -sum(log.(P))-length(P)*log(log(Pmax/Pmin))
end

const Kmin = 0.0
const Kmax = 999.0
const K0 = 1.0
function logprior_amplitude(K::Real)
  logp = -log1p(K/K0)-log(log1p(Kmax/K0)) # -log(K0)
end


function logprior_sinusoids(param)
  const nparam = length(param)
  const namp_per_pl = 2
  npl = convert(Int64,floor((nparam-1)//namp_per_pl))
  logp = 0
  logj = 0
  for i in 0:(npl-1)
    K = sqrt(param[1+namp_per_pl*i]^2+param[2+namp_per_pl*i]^2)
    logp += logprior_amplitude(K)
	logj -= log(K)
  end
  logp -= npl*log(2pi)
  C = param[nparam]
  if !(-Cmax<=C<=Cmax) logp -=Inf
  else  logp -= log(2*Cmax) end
  return (logp,logj)
end

function logprior_epicycles(param)
  nparam = length(param)
  namp_per_pl = 4
  npl = convert(Int64,floor((nparam-1)//namp_per_pl))
  logp = 0
  logj = 0
  for i in 0:(npl-1)
    K  = sqrt(param[1+namp_per_pl*i]^2+param[2+namp_per_pl*i]^2)
	eK = sqrt(param[3+namp_per_pl*i]^2+param[4+namp_per_pl*i]^2)
	if !(0.0<=eK<K) logp -= Inf end
    logp += logprior_amplitude(K)
	logj -= 2*log(K)+log(eK)
  end
  logp -= 2*npl*log(2pi)
  C = param[nparam]
  if !(-Cmax<=C<=Cmax) logp -=Inf
  else  logp -= log(2*Cmax) end
  return (logp,logj)
end

#sincos from LombScargle.jl
# `sincos` is not yet in Julia, but it's available through the math library.  The following
# definitions are from Yichao Yu's code at https://github.com/nacs-lab/yyc-data and are
# likely to be later added to Julia (so they can be removed here).  See also
# https://discourse.julialang.org/t/poor-performance-of-the-cis-function/3402.
if !isdefined(Base, :sincos)
    @inline function sincos(v::Float64)
        Base.llvmcall("""
        %f = bitcast i8 *%1 to void (double, double *, double *)*
        %pres = alloca [2 x double]
        %p1 = getelementptr inbounds [2 x double], [2 x double]* %pres, i64 0, i64 0
        %p2 = getelementptr inbounds [2 x double], [2 x double]* %pres, i64 0, i64 1
        call void %f(double %0, double *nocapture noalias %p1, double *nocapture noalias %p2)
        %res = load [2 x double], [2 x double]* %pres
        ret [2 x double] %res
        """, Tuple{Float64,Float64}, Tuple{Float64,Ptr{Void}}, v,
                      cglobal((:sincos, Base.libm_name)))
    end
    @inline function sincos(v::Float32)
        Base.llvmcall("""
        %f = bitcast i8 *%1 to void (float, float *, float *)*
        %pres = alloca [2 x float]
        %p1 = getelementptr inbounds [2 x float], [2 x float]* %pres, i64 0, i64 0
        %p2 = getelementptr inbounds [2 x float], [2 x float]* %pres, i64 0, i64 1
        call void %f(float %0, float *nocapture noalias %p1, float *nocapture noalias %p2)
        %res = load [2 x float], [2 x float]* %pres
        ret [2 x float] %res
            """, Tuple{Float32,Float32}, Tuple{Float32,Ptr{Void}}, v,
                      cglobal((:sincosf, Base.libm_name)))
    end
    @inline function sincos(v)
        @fastmath (sin(v), cos(v))
    end
end

# Fit planet models
function make_design_matrix_circ(times::Vector{T},periods::Vector{T}) where T<:Real
  const namp_per_pl = 2
  nparam = 1+namp_per_pl*length(periods)
  freqs = 2pi./periods
  X = Array{T}(length(times),nparam)
  for i in 1:length(periods)
     for j in 1:length(times)
       (X[j,1+(i-1)*namp_per_pl],X[j,2+(i-1)*namp_per_pl]) = sincos(freqs[i]*times[j])
	 end
  end
  X[:,nparam] = one(T)
  return X
end

function make_design_matrix_circ_old(times::Vector{T},periods::Vector{T}) where T<:Real
  const namp_per_pl = 2
  nparam = 1+namp_per_pl*length(periods)
  freqs = 2pi./periods
  X = Array{T}(length(times),nparam)
  for i in 1:length(periods)
     X[:,1+(i-1)*namp_per_pl] = sin.(freqs[i]*times)
	 X[:,2+(i-1)*namp_per_pl] = cos.(freqs[i]*times)
  end
  X[:,nparam] = one(T)
  return X
end

function update_design_matrix_circ(X::Array{T,2}, times::Vector{T},periods::Vector{T}, pl_to_update::Integer) where T<:Real
  const namp_per_pl = 2
  nparam = 1+namp_per_pl*length(periods)
  freq = 2pi./periods[pl_to_update]
  @assert size(X) == (length(times),nparam)
  i = pl_to_update
  idx_s = 1+(i-1)*namp_per_pl
  idx_c = 2+(i-1)*namp_per_pl
  for j in 1:length(times)
     (X[j,idx_s],X[j,idx_c]) = sincos(freq*times[j])
  end
  return X
end

function make_design_matrix_epicycle(times::Vector{T},periods::Vector{T}) where T<:Real
  const namp_per_pl = 4
  nparam = 1+namp_per_pl*length(periods)
  freqs = 2pi./periods
  X = Array{T}(length(times),nparam)
  for i in 1:length(periods)
     s = sin.(freqs[i]*times)
	 c = cos.(freqs[i]*times)
	 X[:,1+(i-1)*namp_per_pl] = s
	 X[:,2+(i-1)*namp_per_pl] = c
	 X[:,3+(i-1)*namp_per_pl] = 2*s.*c
	 X[:,4+(i-1)*namp_per_pl] = c.^2.-s.^2
  end
  X[:,nparam] = one(T)
  return X
end

function update_design_matrix_epicycle(X::Array{T,2}, times::Vector{T},periods::Vector{T}, pl_to_update::Integer) where T<:Real
  const namp_per_pl = 4
  nparam = 1+namp_per_pl*length(periods)
  freq = 2pi./periods[pl_to_update]
  @assert size(X) == (length(times),nparam)
  i = pl_to_update
  idx_s = 1+(i-1)*namp_per_pl
  for j in 1:length(times)
	 (s,c) = sincos(freq*times[j])
     X[j,idx_s  ] = s
	 X[j,idx_s+1] = c
	 X[j,idx_s+2] = 2*s.*c
	 X[j,idx_s+3] = c.^2.-s.^2
  end
  return X
end

function calc_best_fit(data::Vector{T}, model::Array{T,2}, Sigma::PDMat{T}) where T<:Real
  inv_chol_Sigma = inv(Sigma.chol[:L])
  param_bestfit = (inv_chol_Sigma*model) \ (inv_chol_Sigma*data)
end

function calc_best_fit(data::Vector{T}, model::Array{T,2}, Sigma::PDiagMat{T}) where T<:Real
  inv_chol_Sigma = sqrt.(Sigma.inv_diag)
  param_bestfit = (inv_chol_Sigma.*model) \ (inv_chol_Sigma.*data)
end

function make_fischer_information_matrix(model::Array{T,2}, Sigma::SigmaT) where {T<:Real, SigmaT<:AbstractPDMat{T}}
   FIM = Xt_invA_X(Sigma,model)
   make_matrix_posdef(FIM)
end

function calc_chisq(data::Vector{T}, model::Array{T,2}, Sigma::SigmaT; FIM::PDMat{T} = make_fischer_information_matrix(model,Sigma), calclogprior::Function = x->(0.0,0.0) ) where {T<:Real, SigmaT<:AbstractPDMat{T} }
  @assert length(data) == size(model,1) == size(Sigma,1)
  nobs = length(data)
  nparam = size(model,2)

  param_bf_linalg = calc_best_fit(data,model,Sigma)
  predict = model*param_bf_linalg
  delta = data.-predict
  chisq = invquad(Sigma,delta) # sum(delta'*Sigma^-1*delta)
end

function compute_Laplace_Approx_param_and_integral(data::Vector{T}, model::Array{T,2}, Sigma::SigmaT; FIM::PDMat{T} = make_fischer_information_matrix(model,Sigma), calclogprior::Function = x->(0.0,0.0) ) where {T<:Real, SigmaT<:AbstractPDMat{T} }
  @assert length(data) == size(model,1) == size(Sigma,1)
  nobs = length(data)
  nparam = size(model,2)

  param_bf_linalg = calc_best_fit(data,model,Sigma)
  predict = model*param_bf_linalg
  delta = data.-predict
  chisq = invquad(Sigma,delta) # sum(delta'*Sigma^-1*delta)

  loglikelihood_mode = 0.5*(-chisq-logdet(Sigma)-nobs*log(2pi))
  (logprior_mode, log_jacobian) = calclogprior(param_bf_linalg) 
  log_mode = logprior_mode + loglikelihood_mode

  sigma_param = inv(FIM).chol[:U]
  LaplaceApprox = 0.5*nparam*log(2pi)-0.5*logdet(FIM)+log_mode + log_jacobian
  return (param_bf_linalg,sigma_param,LaplaceApprox)
end

function compute_Laplace_Approx(data::Vector{T}, model::Array{T,2}, Sigma::SigmaT; FIM::PDMat{T} = make_fischer_information_matrix(model,Sigma), calclogprior::Function = x->(0.0,0.0) ) where {T<:Real, SigmaT<:AbstractPDMat{T} }
  @assert length(data) == size(model,1) == size(Sigma,1)
  nobs = length(data)
  nparam = size(model,2)

  param_bf_linalg = calc_best_fit(data,model,Sigma)
  predict = model*param_bf_linalg
  delta = data.-predict
  chisq = invquad(Sigma,delta) # sum(delta'*Sigma^-1*delta)

  loglikelihood_mode = 0.5*(-chisq-logdet(Sigma)-nobs*log(2pi))
  (logprior_mode, log_jacobian) = calclogprior(param_bf_linalg) 
  log_mode = logprior_mode + loglikelihood_mode
  LaplaceApprox = 0.5*nparam*log(2pi)-0.5*logdet(FIM)+log_mode + log_jacobian
  return LaplaceApprox
end

function generate_K_samples(bf::Vector, sigma, nsamples::Integer = 1)
  nparam = length(bf)
  const namp_per_pl = 2
  theta = bf .+ sigma*randn(nparam,nsamples)
  #theta = param_bf'.+randn(nsamples,nparam)*sigma_param
  npl = convert(Int64,floor((nparam-1)//namp_per_pl))

  Ks = zeros(nsamples,npl)
  for i in 1:npl
    Ks[:,i] = sqrt.(theta[1+namp_per_pl*(i-1),:].^2+theta[2+namp_per_pl*(i-1),:].^2)
  end
  return Ks
end

function generate_Ke_samples(bf::Vector, sigma, nsamples::Integer = 1)
  nparam = length(bf)
  const namp_per_pl = 4
  theta = bf .+ sigma*randn(nparam,nsamples)
  npl = convert(Int64,floor((nparam-1)//namp_per_pl))

  Ks = zeros(nsamples,npl)
  es = zeros(nsamples,npl)
  hs = zeros(nsamples,npl)
  ks = zeros(nsamples,npl)
  for i in 1:npl
    Ks[:,i] = sqrt.(theta[1+namp_per_pl*(i-1),:].^2+theta[2+namp_per_pl*(i-1),:].^2)
	hs[:,i] = theta[3+namp_per_pl*(i-1),:] ./ Ks[:,i]
	ks[:,i] = theta[4+namp_per_pl*(i-1),:] ./ Ks[:,i]
	es[:,i] = sqrt.(hs[:,i].^2.+ks[:,i].^2)
  end
  return (Ks,es,hs,ks)
end

function generate_frac_valid_e(bf::Vector, sigma, nsamples::Integer = 100)
  nparam = length(bf)
  const namp_per_pl = 4
  npl = convert(Int64,floor((nparam-1)//namp_per_pl))
  es = generate_Ke_samples(bf,sigma,nsamples)[2]
  num_pass = 0
  for i in 1:size(es,1)
     if all(es[i,:].<1.0) num_pass += 1 end
  end
  num_pass/nsamples
end

# Now with covariances
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
  #PDMat( [ 0.0 + (i==j ? 2*sigma_j : 0.0) for i in 1:length(t), j in 1:length(t)] )
  PDiagMat( 2*sigma_j*ones(length(t)) )
  ScalMat(length(t),2*sigma_j)
end

function calc_opt_sigmaj(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, model::Array{T,2}; calclogprior_linear::Function = logprior_sinusoids) where T<:Real
  function calc_laplace_approx_arg_jitter(sigmaj::Real) 
    Sigma = Sigma0 + ScalMat(length(times),sigmaj) 
    FIM = PDMat(Xt_invA_X(Sigma,model))
    evid = compute_Laplace_Approx(vels,model,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
	evid += logprior_jitter(sigmaj)
    -evid
  end
  Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.))
  result = optimize(calc_laplace_approx_arg_jitter,sigma_j_min,sigma_j_max,rel_tol=1e-6)
end

# Integrate over period
function compute_marginalized_likelihood_fix_periods(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods::Vector{T}; calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, X = make_design_matrix(times,periods), Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.0)) ) where T<:Real
   const nparam = size(X,2)

   function optimize_jitter_helper(sigmaj::Real) 
     Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))
	 FIM = make_fischer_information_matrix(X,Sigma)
     evid = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
 	 evid += logprior_jitter(sigmaj)
	 -evid
   end
   result = optimize(optimize_jitter_helper,sigma_j_min,sigma_j_max,rel_tol=1e-6)
   #println(result)
   sigmaj = result.minimizer
   #Sigma = PDMat(make_Sigma(times,sigma_obs,sigmaj))
   Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))
   dSdj = make_dSigmadsigmaj(times,sigma_obs,sigmaj)
   FIM = make_fischer_information_matrix(X,Sigma)
   logp = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
   logp += logprior_jitter(sigmaj)+ logprior_periods(periods)
   logp += 0.5*(log(2pi)-log(0.5*trace((Sigma \ full(dSdj))^2)))
   #println("# logp = ",logp)
   return logp 
end

function make_Sigma_cache(num_Sigmas::Integer, Sigma0::SigmaT ) where {T<:Real, SigmaT<:AbstractPDMat{T}}
   @assert num_Sigmas >= 1
   nodes, weights = gausslegendre( num_Sigmas );
   weights *= 0.5;
   nodes = 0.5+0.5*nodes;
   sigmaj_list = sigma_j_0*(exp.(nodes*log1p(sigma_j_max/sigma_j_0)).-1);
   SigmaList = map(s->Sigma0+ScalMat(size(Sigma0,1),s),sigmaj_list);
   return (sigmaj_list, weights, SigmaList)
end

function make_Sigma_cache(times::Vector{T}, sigma_obs::Vector{T}, num_Sigmas::Integer ) where T<:Real
   @assert num_Sigmas >= 1
   Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.0))
   nodes, weights = gausslegendre( num_Sigmas );
   weights *= 0.5;
   nodes = 0.5+0.5*nodes;
   sigmaj_list = sigma_j_0*(exp.(nodes*log1p(sigma_j_max/sigma_j_0)).-1);
   SigmaList = map(s->Sigma0+ScalMat(length(times),s),sigmaj_list);
   return (sigmaj_list, weights, SigmaList)
end
 

function compute_marginalized_likelihood_const_sigmalist(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}; Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.0)), num_Sigmas::Integer = 20, Sigma_cache = make_Sigma_cache(num_Sigmas,Sigma0) ) where T<:Real
   X = ones(length(times),1)
   const nparam = size(X,2)
   sigmaj_list = Sigma_cache[1]
   weights = Sigma_cache[2]
   SigmaList = Sigma_cache[3]

	   marginalized_likelihood_fix_periods_sigmaj = map(k->compute_marginalized_likelihood_fix_periods_sigmaj(times, vels, sigma_obs, Float64[], sigmaj_list[k], Sigma=SigmaList[k],X=X),1:length(sigmaj_list))
       int_norm = maximum(marginalized_likelihood_fix_periods_sigmaj);
       logp = log(dot( weights, exp.(marginalized_likelihood_fix_periods_sigmaj-int_norm) ))+int_norm
end

function compute_marginalized_likelihood_fix_periods_sigmalist(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods::Vector{T}; calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, X = make_design_matrix(times,periods), Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.0)), num_Sigmas::Integer = 20, Sigma_cache = make_Sigma_cache(num_Sigmas,Sigma0) ) where T<:Real
   const nparam = size(X,2)
   sigmaj_list = Sigma_cache[1]
   weights = Sigma_cache[2]
   SigmaList = Sigma_cache[3]

	   marginalized_likelihood_fix_periods_sigmaj = map(k->compute_marginalized_likelihood_fix_periods_sigmaj(times, vels, sigma_obs, periods, sigmaj_list[k], Sigma=SigmaList[k],calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix, X=X),1:length(sigmaj_list))
       int_norm = maximum(marginalized_likelihood_fix_periods_sigmaj);
       logp = log(dot( weights, exp.(marginalized_likelihood_fix_periods_sigmaj-int_norm) ))+int_norm

end

  
function compute_marginalized_likelihood_fix_periods_sigmaj(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods::Vector{T}, sigmaj::T; calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, Sigma=PDMat(make_Sigma(times,sigma_obs,sigmaj)), X= make_design_matrix(times,periods), FIM = make_fischer_information_matrix(X,Sigma) ) where T<:Real
   #X = make_design_matrix(times,periods)
   const nparam = size(X,2)
   #Sigma = PDMat(make_Sigma(times,sigma_obs,sigmaj))
   #dSdj = make_dSigmadsigmaj(times,sigma_obs,sigmaj)
   #FIM = Xt_invA_X(Sigma,X)

   logp = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
   #logp += logprior_jitter(sigmaj)+ logprior_periods(periods)
   logp += logprior_periods(periods)
   #logp += 0.5*(log(2pi)-log(0.5*trace((Sigma \ full(dSdj))^2)))
   #println("# logp = ",logp)
   return logp 
end



function compute_marginalized_likelihood(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, minperiods::Vector{T}, maxperiods::Vector{T}; calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ  ) where T<:Real
  @assert length(minperiods)==length(maxperiods)
  function helper(periods::Vector{T})
     logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,periods,calclogprior_linear=calclogprior_linear,make_design_matrix=make_design_matrix)
	 (logp == -Inf) ? 0. : exp(logp)
  end
  function helper_1d(period::T)
    println("# Evaluating P= ",period) 
	flush(STDOUT)
    logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,[period],calclogprior_linear=calclogprior_linear,make_design_matrix=make_design_matrix)
	println("#      ", logp) 
	flush(STDOUT)
	return (logp == -Inf) ? 0. : exp(logp)
  end
  
  if length(minperiods)!=1
     error("# Sorry, we haven't implemented multiple periods yet")
  end
  result = QuadGK.quadgk(helper_1d,minperiods[1],maxperiods[1],reltol=1e-3)
end


function logsumlogs(x::T,y::T) where T<:Real
  (x<=y) ? y + log1p(exp(x-y)) : x + log1p(exp(y-x)) 
end


function brute_force_over_periods_1d(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, Plo::Real, Phi::Real; samples_per_peak::Real = 1.0, calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, sigmaj::Real = -1.0, num_Sigmas::Integer = 20, verbose::Integer = 0 ) where T<:Real
  min_freq = 1/Phi
  max_freq = 1/Plo
  duration = maximum(times)-minimum(times)
  mean_vel = sum(vels./sigma_obs.^2)/sum(sigma_obs.^(-2))
  rms_vel = std((vels.-mean_vel)./sigma_obs)
  num_periods = convert(Int64,ceil(samples_per_peak*2pi*duration*rms_vel*(max_freq-min_freq)))
  println("# num_periods = ",num_periods)
  periods = zeros(1)
  sigma_eps = 1e-2
  Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.+sigma_eps));
  if sigmaj >= 0.0
     Sigma = Sigma0 + ScalMat(length(times),sigmaj)
  else
     Sigma = Sigma0 + ScalMat(length(times),rms_vel)
  end
  
  if num_Sigmas > 1
#=     nodes, weights = gausslegendre( num_Sigmas );
     weights *= 0.5;
     nodes = 0.5+0.5*nodes;
     sigmaj_list = sigma_j_0*(exp.(nodes*log1p(sigma_j_max/sigma_j_0)).-1);
     SigmaList = map(s->Sigma0+ScalMat(length(times),s),sigmaj_list);
	 =#
	 (sigmaj_list,weights,SigmaList) = make_Sigma_cache(num_Sigmas,Sigma0)
  end

  deltaP_const = 1/(samples_per_peak*2pi*duration*rms_vel)
  P = Plo
  period_grid = Array{Float64}(2*num_periods)
  power_grid  = Array{Float64}(2*num_periods)
  logintegral = -Inf
  i = 0
  while P<=Phi
    deltaP = deltaP_const*P^2
	if i >0 P += deltaP end
	periods[1] = P
    i += 1
	X = make_design_matrix(times,periods)
    if num_Sigmas >1 
	   logp = compute_marginalized_likelihood_fix_periods_sigmalist(times, vels, sigma_obs, periods, calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix, X=X, Sigma0=Sigma0,  num_Sigmas=num_Sigmas, Sigma_cache=(sigmaj_list,weights,SigmaList) )
	#=
	   marginalized_likelihood_fix_periods_sigmaj = map(k->compute_marginalized_likelihood_fix_periods_sigmaj(times, vels, sigma_obs, periods, sigmaj_list[k], Sigma=SigmaList[k],calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix, X=X),1:length(sigmaj_list))
       int_norm = maximum(marginalized_likelihood_fix_periods_sigmaj);
       logp = log(dot( weights, exp.(marginalized_likelihood_fix_periods_sigmaj-int_norm) ))+int_norm
=#
    else
	   if sigmaj < 0
	      logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,periods,Sigma0=Sigma0, calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix,X=X)  
	   else
	      logp = compute_marginalized_likelihood_fix_periods_sigmaj(times,vels,sigma_obs,periods,sigmaj, Sigma=Sigma, calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix,X=X)  
	      logp += logprior_jitter(sigmaj)
	   end		  
    end
	logp += log(deltaP/P)
    if verbose >=2 && i%10 == 0
	  println("#i=",i," P=",P," dP=",deltaP," logp=",logp," logI=",logintegral)
	end
	if logintegral == -Inf
	  logintegral = logp
	else
	  logintegral = logsumlogs(logintegral,logp)
	end
	@assert i<=length(period_grid)
	period_grid[i] = P
	power_grid[i] = logp
	power_grid[i] = logp
    
  end
  resize!(period_grid,i)
  resize!(power_grid,i)
  println("# Last P = ", period_grid[end])
  return (period_grid,power_grid,logintegral)
end

function brute_force_over_periods_multiplanet_1d(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods_fixed::Vector{T}, Plo::Real, Phi::Real; samples_per_peak::Real = 2.0, calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, update_design_matrix::Function = update_design_matrix_circ, sigmaj::Real = -1.0, num_Sigmas::Integer = 20, verbose::Integer = 0 ) where T<:Real
  periods = zeros(length(periods_fixed)+1)
  periods[1:length(periods_fixed)] = periods_fixed

  # Compute how much signal is remaining so can choose appropriate grid density
  X = make_design_matrix(times,periods_fixed)
  sigmaj_opt = calc_opt_sigmaj(times, vels, sigma_obs, X, calclogprior_linear=calclogprior_linear).minimizer
  Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.))
  Sigma = Sigma0 + ScalMat(length(times),sigmaj_opt) # PDMat(make_Sigma(times,sigma_obs,sigmaj_opt))
  param_bf_linalg = calc_best_fit(vels,X,Sigma)
  predict = X*param_bf_linalg
  
  chisq = invquad(Sigma0,vels.-predict)
  rms_vel = std((vels.-predict)./sigma_obs)
  println("# RMS resid (uncor) = ", rms_vel)
  #rms_vel_eff = sqrt(chisq/length(times))
  #println("# RMS resid (corel) = ", rms_vel_eff)
  
  periods[end] = (Plo+Phi)/2
  X = make_design_matrix(times,periods)
  
  if num_Sigmas > 1
     #= 
	 nodes, weights = gausslegendre( num_Sigmas );
     weights *= 0.5;
     nodes = 0.5+0.5*nodes;
     sigmaj_list = sigma_j_0*(exp.(nodes*log1p(sigma_j_max/sigma_j_0)).-1);
     #Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.));
     SigmaList = map(sigmaj->Sigma0+ScalMat(length(times),sigmaj),sigmaj_list);
	 =#
	 (sigmaj_list,weights,SigmaList) = make_Sigma_cache(num_Sigmas,Sigma0)
  end

  # Now compute periodogram
  min_freq = 1/Phi
  max_freq = 1/Plo
  duration = maximum(times)-minimum(times)
  num_periods = convert(Int64,ceil(samples_per_peak*2pi*duration*rms_vel*(max_freq-min_freq)))
  println("# num_periods = ",num_periods)
  period_grid = Array{Float64}(2*num_periods)
  power_grid  = Array{Float64}(2*num_periods)
  deltaP_const = 1/(samples_per_peak*2pi*duration*rms_vel)
  P = Plo
  logintegral = -Inf
  i = 0
  while P<=Phi
    deltaP = deltaP_const*P^2
	if i >0 P += deltaP end
    i += 1
    periods[end] =  P
	#println("# size(X) = ",size(X), " len(times)= ", length(times), " len(periods) = ", length(periods))
	update_design_matrix(X,times,periods,length(periods))

    if num_Sigmas == 1
	   if sigmaj < 0
   	      logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,periods, calclogprior_linear=calclogprior_linear, X=X, Sigma0=Sigma0) # make_design_matrix=make_design_matrix)
	   else
	      logp = compute_marginalized_likelihood_fix_periods_sigmaj(times,vels,sigma_obs,periods,sigmaj, calclogprior_linear=calclogprior_linear, X=X, Sigma=Sigma) # make_design_matrix=make_design_matrix)
	   end
    else
		  logp = compute_marginalized_likelihood_fix_periods_sigmalist(times, vels, sigma_obs, periods, calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix, X=X, Sigma0=Sigma0,  num_Sigmas=num_Sigmas, Sigma_cache=(sigmaj_list,weights,SigmaList) )
       #=
	   marginalized_likelihood_fix_periods_sigmaj = map(k->compute_marginalized_likelihood_fix_periods_sigmaj(times, vels, sigma_obs, periods, sigmaj_list[k], Sigma=SigmaList[k],calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix, X=X),1:length(sigmaj_list))
       int_norm = maximum(marginalized_likelihood_fix_periods_sigmaj);
       logp = log(dot( weights, exp.(marginalized_likelihood_fix_periods_sigmaj-int_norm) ))+int_norm
	   =#
	end
	
	logp += log(deltaP/P)
    if verbose >=2 && i%10 == 0
	  println("# i=",i," P=",P," dP=",deltaP," logp=",logp," logI=",logintegral)
	end 
	if logintegral == -Inf
	  logintegral = logp
	else
	  logintegral = logsumlogs(logintegral,logp)
	end
	period_grid[i] = P
	power_grid[i] = logp
    
  end
  resize!(period_grid,i)
  resize!(power_grid,i)
  println("# Last P = ", period_grid[end])
  (period_grid,power_grid,logintegral)
end

  

# Optimize periods
function compute_marginalized_likelihood_near_periods_laplace(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods::Vector{T}; calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ ) where T<:Real
   X = make_design_matrix(times,periods)

   function optimize_jitter_helper(sigmaj::Real) 
     #println("# making Sigma")
	 Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))
	 #println("# making FIM")
     FIM = PDMat(Xt_invA_X(Sigma,X))
     #println("# approximating integral")
	 evid = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
 	 evid += logprior_jitter(sigmaj)
	 #println("# ",evid/log(10))
     -evid
   end
   Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.))
   result = optimize(optimize_jitter_helper,sigma_j_min,sigma_j_max,rel_tol=1e-6)
   #println("# Optimize jitter: ", result)
   sigmaj = result.minimizer
   Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))

   function optimize_period_helper(P::T) 
     X = make_design_matrix(times,[P])
     FIM = PDMat(Xt_invA_X(Sigma,X))
     evid = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
 	 evid += logprior_jitter(sigmaj)
     -evid
   end
   function optimize_period_helper(periods::Vector{T}) 
     X = make_design_matrix(times,periods)
     FIM = PDMat(Xt_invA_X(Sigma,X))
     evid = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
 	 evid += logprior_jitter(sigmaj)
     -evid
   end
   
   if(length(periods)==1)
      result2 = optimize(optimize_period_helper,periods[1]*0.98,periods[1]*1.02,rel_tol=1e-6) 
	  periods = [result2.minimizer]
   else
	  result2 = optimize(optimize_period_helper,periods,f_tol=1e-3) # abs(result.minimum)) 
	  periods = result2.minimizer
   end
   #println("# Optimize period: ", result2)
   best_period = copy(periods)
   X = make_design_matrix(times,periods)
   FIM = PDMat(Xt_invA_X(Sigma,X))
   (param_bf,sigma_param,logp) = compute_Laplace_Approx_param_and_integral(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
   logp += logprior_jitter(sigmaj)+ logprior_periods(periods)

   #println("# marginal at best sigma_j & P: ",logp/log(10))
   
   # Laplace approx to integrate over jitter parameter
   dSdj = make_dSigmadsigmaj(times,sigma_obs,sigmaj)
   log_int_over_sigma_j = 0.5*(log(2pi)-log(0.5*trace((Sigma \ full(dSdj))^2)))
   logp += log_int_over_sigma_j

   #println("# marginal over sigma_j at best P: ",logp/log(10))

   # Laplace approx to integrate over period 
   sigma_P = zeros(length(periods))
   log_int_over_periods = 0
   duration = maximum(times)-minimum(times)
   param_per_pl = (calclogprior_linear == logprior_sinusoids) ? 2 : 4
   for i in 1:length(best_period)
     periods = copy(best_period)
	 
     amp = sqrt(param_bf[1+(i-1)*param_per_pl]^2+param_bf[2+(i-1)*param_per_pl]^2)
     eps = periods[i]/(40*2pi*duration*amp)
	 periods[i] += eps
	 
     X = make_design_matrix(times,periods)
     logp_hi = compute_Laplace_Approx(vels,X,Sigma,calclogprior=calclogprior_linear)
     logp_hi += logprior_jitter(sigmaj)+ logprior_periods(periods)
     logp_hi += log_int_over_sigma_j  # Approximates this term as const
	 periods[i] -= 2*eps
     X = make_design_matrix(times,periods)
	 logp_lo = compute_Laplace_Approx(vels,X,Sigma,calclogprior=calclogprior_linear)
     logp_lo += logprior_jitter(sigmaj)+ logprior_periods(periods)
     logp_lo += log_int_over_sigma_j # Approximates this term as const
     d2logpdp2 = (logp_hi-2*logp+logp_lo)/eps^2
     sigma_P[i] = sqrt(abs(1/d2logpdp2))
     #println("# d^2 logp/dP_",i,"^2 = ",d2logpdp2/log(10))
     log_int_over_periods += 0.5*(log(2pi)-log(abs(d2logpdp2)))
   end
   logp += log_int_over_periods # Approx as diagonal
   println("# marginal over P at ",best_period," and sigma_j at ",sigmaj," (laplace): ",logp/log(10))
   if calclogprior_linear == logprior_epicycles
      frac_valid_e = generate_frac_valid_e(param_bf,sigma_param) 
      println("# frac w/ valid e = ", frac_valid_e)
      logp += log(frac_valid_e)
   end
   return (param_bf,sigma_param,logp,best_period,sigma_P)
end

function compute_marginalized_likelihood_near_periods_gauss(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods::Vector{T}; calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, num_Sigmas::Integer = 20 ) where T<:Real
   X = make_design_matrix(times,periods)

   function optimize_jitter_helper(sigmaj::Real) 
     #println("# making Sigma")
	 Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))
	 #println("# making FIM")
     FIM = PDMat(Xt_invA_X(Sigma,X))
     #println("# approximating integral")
	 evid = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
 	 evid += logprior_jitter(sigmaj)
	 #println("# ",evid/log(10))
     -evid
   end
   Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.))
   result = optimize(optimize_jitter_helper,sigma_j_min,sigma_j_max,rel_tol=1e-6)
   #println("# Optimize jitter: ", result)
   sigmaj = result.minimizer
   Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))

   function optimize_period_helper(P::T) 
     X = make_design_matrix(times,[P])
     FIM = PDMat(Xt_invA_X(Sigma,X))
     evid = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
 	 evid += logprior_jitter(sigmaj)
     -evid
   end
   function optimize_period_helper(periods::Vector{T}) 
     X = make_design_matrix(times,periods)
     FIM = PDMat(Xt_invA_X(Sigma,X))
     evid = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
 	 evid += logprior_jitter(sigmaj)
     -evid
   end
   
   if(length(periods)==1)
      result2 = optimize(optimize_period_helper,periods[1]*0.98,periods[1]*1.02,rel_tol=1e-4) 
	  periods = [result2.minimizer]
   else
	  result2 = optimize(optimize_period_helper,periods,f_tol=1e-4) # abs(result.minimum)) 
	  periods = result2.minimizer
   end
   #println("# Optimize period: ", result2)
   best_period = copy(periods)
   X = make_design_matrix(times,periods)
  (param_bf,sigma_param,logp) = compute_Laplace_Approx_param_and_integral(vels,X,Sigma,calclogprior=calclogprior_linear)

    (sigmaj_list,weights,SigmaList) = make_Sigma_cache(num_Sigmas,Sigma0)
     logp = compute_marginalized_likelihood_fix_periods_sigmalist(times, vels, sigma_obs, periods, calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix, X=X, Sigma0=Sigma0,  num_Sigmas=num_Sigmas, Sigma_cache=(sigmaj_list,weights,SigmaList) ) 

   # Laplace approx to integrate over period 
   sigma_P = zeros(length(periods))
   log_int_over_periods = 0
   duration = maximum(times)-minimum(times)
   param_per_pl = (calclogprior_linear == logprior_sinusoids) ? 2 : 4
   for i in 1:length(best_period)
     periods = copy(best_period)
	 amp = sqrt(param_bf[1+(i-1)*param_per_pl]^2+param_bf[2+(i-1)*param_per_pl]^2)
     eps = periods[i]/(40*2pi*duration*amp)
	 periods[i] += eps
	 
     X = make_design_matrix(times,periods)
	 logp_hi = compute_marginalized_likelihood_fix_periods_sigmalist(times, vels, sigma_obs, periods, calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix, X=X, Sigma0=Sigma0,  num_Sigmas=num_Sigmas, Sigma_cache=(sigmaj_list,weights,SigmaList) ) 
	 
	 periods[i] -= 2*eps
     X = make_design_matrix(times,periods)
	 logp_lo = compute_marginalized_likelihood_fix_periods_sigmalist(times, vels, sigma_obs, periods, calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix, X=X, Sigma0=Sigma0,  num_Sigmas=num_Sigmas, Sigma_cache=(sigmaj_list,weights,SigmaList) ) 
	 
     d2logpdp2 = (logp_hi-2*logp+logp_lo)/eps^2
     sigma_P[i] = sqrt(abs(1/d2logpdp2))
     #println("# d^2 logp/dP_",i,"^2 = ",d2logpdp2/log(10))
     log_int_over_periods += 0.5*(log(2pi)-log(abs(d2logpdp2)))
   end
   logp += log_int_over_periods # Approx as diagonal
   println("# marginal over P at ",best_period," and sigma_j at ",sigmaj," (gauss): ",logp/log(10))	 
   if calclogprior_linear == logprior_epicycles
      frac_valid_e = generate_frac_valid_e(param_bf,sigma_param) 
      println("# frac w/ valid e = ", frac_valid_e)
      logp += log(frac_valid_e)
   end
   return (param_bf,sigma_param,logp,best_period,sigma_P)
end

function find_n_peaks_in_pgram(period::Vector{Float64}, power::Vector{Float64}; num_peaks::Integer = 1, exclude_period_factor::Real = 1.2, exclude_peaks::Vector{Float64} = Float64[] )
  @assert num_peaks>= 1
  peak_periods = zeros(num_peaks)
  peak_logps = zeros(num_peaks)
  peaks_found = 1
  idx_peak = findmax(power)[2]
  peak_periods[peaks_found] = period[idx_peak]
  peak_logps[peaks_found] = power[idx_peak]
  idx_active = trues(length(period))
  for i in 1:length(exclude_peaks)
     for j in 1:length(period)
        if exclude_peaks[i]/exclude_period_factor <= period[j] <= exclude_peaks[i]*exclude_period_factor
           idx_active[j] = false
        end # if near an excluded peak
     end # for over periods
  end # for over excluded peaks
  while peaks_found < num_peaks
     peaks_found += 1
	 idx_peak = findmax(power[idx_active])[2]
     peak_periods[peaks_found] = period[idx_active][idx_peak]
	 peak_logps[peaks_found] = power[idx_active][idx_peak]

     for j in 1:length(period)
        if peak_periods[peaks_found]/exclude_period_factor <= period[j] <= peak_periods[peaks_found]*exclude_period_factor
           idx_active[j] = false
        end # blackout periods near most recent peak
     end # for over periods

	 end # while more peaks to be found
  (peak_periods,peak_logps)
end



function compute_evidences_laplace_circ{T}(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}; num_Sigmas::Integer = 20, samples_per_peak::Integer = 1, Pminsearch::Vector{Float64} = 10.0*ones(3) )
	 #(sigmaj_list,weights,SigmaList) = make_Sigma_cache(num_Sigmas,Sigma0)

runtimes = zeros(4)
tic()	
log10_evidences = zeros(4)
res0l = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,Float64[])
res0g = compute_marginalized_likelihood_const_sigmalist(times,vels,sigma_obs,num_Sigmas=num_Sigmas)
println("# Evidence for npl=0: ",res0l/log(10),", ", res0g/log(10))
#log10_evidences[1] = res0l/log(10)
log10_evidences[1] = res0g/log(10)
runtimes[1] = toq()
tic()

@time res1b = brute_force_over_periods_1d(times,vels,sigma_obs,Pminsearch[1],Pmax,samples_per_peak=samples_per_peak,num_Sigmas=num_Sigmas)
P1 = find_n_peaks_in_pgram(res1b[1],res1b[2])[1][1]

res1l = compute_marginalized_likelihood_near_periods_laplace(times,vels,sigma_obs, [P1] )
res1g = compute_marginalized_likelihood_near_periods_gauss(times,vels,sigma_obs, [P1],num_Sigmas=num_Sigmas )
#log10_evidences[2] = res1l[3] > res1b[3] ? res1l[3]/log(10) : res1b[3]/log(10)
log10_evidences[2] = res1g[3] > res1b[3] ? res1g[3]/log(10) : res1b[3]/log(10)
P1 = res1l[4][1]
runtimes[2] = toq()
tic()
@time res2b = brute_force_over_periods_multiplanet_1d(times,vels,sigma_obs,[P1],Pminsearch[2],Pmax,samples_per_peak=samples_per_peak,num_Sigmas=num_Sigmas)
P2 = find_n_peaks_in_pgram(res2b[1],res2b[2],exclude_peaks=[P1])[1][1]

res2l = compute_marginalized_likelihood_near_periods_laplace(times,vels,sigma_obs, [P1,P2] )
res2g = compute_marginalized_likelihood_near_periods_gauss(times,vels,sigma_obs, [P1,P2],num_Sigmas=num_Sigmas  )
#log10_evidences[3] = res2l[3] > res2b[3] ? res2l[3]/log(10) : res2b[3]/log(10)
log10_evidences[3] = res2g[3] > res2b[3] ? res2g[3]/log(10) : res2b[3]/log(10)
P1 = res2l[4][1]
P2 = res2l[4][2]
runtimes[3] = toq()
tic()
@time res3b = brute_force_over_periods_multiplanet_1d(times,vels,sigma_obs,[P1,P2],Pminsearch[3],Pmax,samples_per_peak=samples_per_peak,num_Sigmas=num_Sigmas)
P3 = find_n_peaks_in_pgram(res3b[1],res3b[2],num_peaks=10,exclude_peaks=[P1,P2])[1][1]

res3l = compute_marginalized_likelihood_near_periods_laplace(times,vels,sigma_obs, [P1,P2,P3] )
res3g = compute_marginalized_likelihood_near_periods_gauss(times,vels,sigma_obs, [P1,P2,P3],num_Sigmas=num_Sigmas  )
#log10_evidences[4] = res3l[3] > res3b[3] ? res3l[3]/log(10) : res3b[3]/log(10)
log10_evidences[4] = res3g[3] > res3b[3] ? res3g[3]/log(10) : res3b[3]/log(10)
runtimes[4] = toq()

  return (log10_evidences,[P1,P2,P3],[res1l,res2l,res3l],runtimes)
end

function make_kepler_param_from_linear_output(output, npl::Integer)
  evidence_linear = output[1][npl]
  periods = output[2][1:npl]
  param_linear = output[4][npl]
  sigmaj = output[6][npl]
  chisq_linear = output[10][npl]
  const nparam_per_pl = 5
  const nparam_non_pl = 2
  const nparam_linear_per_pl = 4
  const nparam = npl*nparam_per_pl + nparam_non_pl
  const tref = 300.0
  param = Array{Float64}(nparam)
  for i in 1:npl
      P = periods[i]
	  Ksinl = param_linear[1+(i-1)*nparam_linear_per_pl]
	  Kcosl = param_linear[2+(i-1)*nparam_linear_per_pl]
	  K = sqrt(Ksinl^2+Kcosl^2)
	  wpM = atan2(Ksinl,Kcosl)
	  h = K>0 ? param_linear[3+(i-1)*nparam_linear_per_pl]/K : 0
	  k = K>0 ? param_linear[4+(i-1)*nparam_linear_per_pl]/K : 0
	  w = atan2(h,k)
      param[1+(i-1)*nparam_per_pl] = P
	  param[2+(i-1)*nparam_per_pl] = K > 0 ? K : 0.5
	  param[3+(i-1)*nparam_per_pl] = h/100
	  param[4+(i-1)*nparam_per_pl] = k/100
	  param[5+(i-1)*nparam_per_pl] = mod2pi(wpM+2pi*tref/P)	  
  end
  param[1+npl*nparam_per_pl] = param_linear[1+npl*nparam_linear_per_pl]
  param[2+npl*nparam_per_pl] = sigmaj
  param
end

function optimize_phase{T}(param, times::Vector{T}, vels::Vector{T}, Sigma, plid::Integer)
  const nparam_per_pl = 5
  const nparam_non_pl = 2
  npl = convert(Int64,floor((length(param)-1)//nparam_per_pl))
  function optimize_phase_helper(x::Real)
    param[5+(plid-1)*nparam_per_pl] = x
    -loglikelihood_fixed_jitter(param,times,vels,Sigma)
  end
  result = optimize(optimize_phase_helper,0.0,2pi) 
  param[5+(plid-1)*nparam_per_pl] = mod2pi(result.minimizer)
  param
end

function optimize_omega{T}(param, times::Vector{T}, vels::Vector{T}, Sigma, plid::Integer)
  const nparam_per_pl = 5
  const nparam_non_pl = 2
  npl = convert(Int64,floor((length(param)-1)//nparam_per_pl))
  function optimize_omega_helper(x::Vector)
    param[3+(plid-1)*nparam_per_pl:4+(plid-1)*nparam_per_pl] = x
    -loglikelihood_fixed_jitter(param,times,vels,Sigma)
  end
  result = optimize(optimize_omega_helper, param[3+(plid-1)*nparam_per_pl:4+(plid-1)*nparam_per_pl] ) 
  if sum(result.minimizer.^2)<1
     param[3+(plid-1)*nparam_per_pl:4+(plid-1)*nparam_per_pl] = result.minimizer
  else
     param[3+(plid-1)*nparam_per_pl] = 0.0 #1e-6*param[3+(plid-1)*nparam_per_pl]
	 param[4+(plid-1)*nparam_per_pl] = 0.0 #1e-6*param[4+(plid-1)*nparam_per_pl]
  end
  param
end

function optimize_phases{T}(param, times::Vector{T}, vels::Vector{T}, Sigma)
  const nparam_per_pl = 5
  const nparam_non_pl = 2
  npl = convert(Int64,floor((length(param)-1)//nparam_per_pl))
  for plid in 1:npl
     optimize_phase(param,times,vels,Sigma,plid)
	 println("# param (after phase ",plid," ) = ", param)
	 optimize_omega(param,times,vels,Sigma,plid)
	 println("# param = (after omega ",plid," = ", param)
  end
  param
end


function analyze_dataset(filename::String; num_Sigmas::Integer = 20, samples_per_peak::Real = 1.0, Pminsearch::Vector{Float64} = 10.0*ones(3))
  # Read dataset
  data = readdlm(filename);
  times = data[:,1];
  vels = data[:,2];
  sigma_obs = data[:,3];
  nobs = length(vels)
  
   (evidence_circ_list, P_best, result_laplace_circ_list, runtimes_circ) = compute_evidences_laplace_circ(times,vels,sigma_obs,num_Sigmas=num_Sigmas, Pminsearch=Pminsearch)
   runtimes_kepler = zeros(runtimes_circ)
      
   param_bf_list = Vector{Vector{Float64}}(length(P_best))
   resid_list = Vector{Vector{Float64}}(length(P_best))
   rms_list = Vector{Float64}(length(P_best))
   chisq_list = Vector{Float64}(length(P_best))
   sigmaj_list = Vector{Float64}(length(P_best))
   sigma_P_list = Vector{Vector{Float64}}(length(P_best))
   sigma_sigmaj_list = Vector{Float64}(length(P_best))
   sigma_param_list = Vector(length(P_best))
   frac_valid_list = Vector{Float64}(length(P_best))
   evidence_kepler_list = Vector{Float64}(length(P_best))
   kepler_fit_list = Vector(length(P_best))
   for np in 1:length(P_best)  # Loop over number of planets to include
      sigma_P_list[np] = result_laplace_circ_list[3][5]
	  tic()
	  X = make_design_matrix_epicycle(times,P_best[1:np])
      sigmaj_opt = calc_opt_sigmaj(times, vels, sigma_obs, X, calclogprior_linear=logprior_epicycles).minimizer
      Sigma = PDMat(make_Sigma(times,sigma_obs,sigmaj_opt))
      param_bf_list[np] = calc_best_fit(vels,X,Sigma)
      predict = X*param_bf_list[np]
      # Estimate Uncertainties
      FIM = make_fischer_information_matrix(X,Sigma)
      sigma_param = inv(FIM).chol[:U]
      frac_valid_e = generate_frac_valid_e(param_bf_list[np],sigma_param) 
      # Uncertainty on sigmaj
      dSdj = make_dSigmadsigmaj(times,sigma_obs,sigmaj_opt)
      sigma_sigmaj = 1/sqrt(abs(0.5*trace((Sigma \ full(dSdj))^2)))
      println("# np=", np," sigma_sigmaj = ",sigma_sigmaj,"  frac w/ valid e = ", frac_valid_e)
      evidence_kepler_list[np] = compute_marginalized_likelihood_near_periods_gauss(times,vels,sigma_obs, P_best[1:np],num_Sigmas=num_Sigmas, calclogprior_linear=logprior_epicycles, make_design_matrix=make_design_matrix_epicycle )[3]/log(10)
	  
      resid_list[np] = vels.-predict
      rms_list[np] = std(resid_list[np])
      chisq_list[np] = invquad(Sigma,resid_list[np]) 
	  sigma_param_list[np] = sigma_param
	  sigmaj_list[np] = sigmaj_opt
	  sigma_sigmaj_list[np] = sigma_sigmaj
	  frac_valid_list[np] = frac_valid_e
	  runtimes_kepler[np] += toq()
  end
   
   result_linear = (evidence_circ_list, P_best,sigma_P_list, param_bf_list, sigma_param_list, sigmaj_list, sigma_sigmaj_list, resid_list, rms_list, chisq_list, frac_valid_list, runtimes_circ, evidence_kepler_list, runtimes_kepler)
   
   #=
   Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.0));
   for npl in 1:length(P_best)
     tic()
     param_tmp = make_kepler_param_from_linear_output(result_linear,npl)
     SigmaBig = PDMat(make_Sigma(times,sigma_obs,sigmaj_list[npl]));
     optimize_phases(param_tmp,times,vels,SigmaBig)
     kepler_fit_list[npl] = laplace_approximation_kepler_model_jitter_separate(param_tmp,times,vels,Sigma0)
     evidence_kepler_list[npl] = kepler_fit_list[npl][1]/log(10)
	 println("# logp npl=",npl," (laplace): ",evidence_kepler_list[npl])
     flush(STDOUT)
     evidence_kepler_list[npl] = laplace_approximation_kepler_model_jitter_separate_gauss(param_tmp,times,vels,Sigma0)[3]/log(10)
	 println("# logp npl=",npl," (gauss): ",evidence_kepler_list[npl])
	 flush(STDOUT)
     runtimes_kepler[np] += toq()
  end
  
   return (evidence_circ_list, P_best,sigma_P_list, param_bf_list, sigma_param_list, sigmaj_list, sigma_sigmaj_list, resid_list, rms_list, chisq_list, frac_valid_list,evidence_kepler_list,runtimes_kepler, kepler_fit_list,runtimes_circ,runtimes_kepler)
   =#
end


function write_evidences_file(filename::String,output; use_method::Integer = 1)
   if use_method==1
      evidence = output[1]
	  runtime = convert(Array{Int64,1},ceil.(output[12]))
   elseif use_method==2
	  evidence = vcat(output[1][1],output[13])
	  runtime = convert(Array{Int64,1},ceil.(output[14]))
   end
   output_circ_data = Array{Any}(4,8)
   output_circ_data[1,1] = 0
   output_circ_data[:,1] = runtime
   output_circ_data[:,2] = collect(0:3)
   for i in 3:8
      output_circ_data[:,i] = evidence
   end
   writedlm(filename, output_circ_data, ", ")
end

function write_evidences_file(filename::String,output; use_method::Integer = 1, input_filename::String="../data/rvs_0001.txt")
   if use_method==1
      evidence = output[1]
	  runtime = convert(Array{Int64,1},ceil.(output[12]))
   elseif use_method==2
	  evidence = vcat(output[1][1],output[13])
	  runtime = convert(Array{Int64,1},ceil.(output[14]))
   elseif use_method==3
       evidence = zeros(4)
	   runtime = zeros(4)
	   data = readdlm(input_filename)
       times = data[:,1];
       vels = data[:,2];
       sigma_obs = data[:,3];
	   Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.0));
	   evidence[1] = NaN
       for np in 1:3
	      tic()
		  param = make_kepler_param_from_linear_output(output,np)
	      SigmaBig = PDMat(make_Sigma(times,sigma_obs,param[end]));
		  optimize_phases(param,times,vels,SigmaBig)
	      param2 = laplace_approximation_kepler_model_jitter_separate(param,times,vels,Sigma0,opt_jitter=false)[2].minimizer
		  evidence[1+np] = laplace_approximation_kepler_model_jitter_separate(param2,times,vels,Sigma0,opt_jitter=true)[1]/log(10) 
		  runtime[1+np] = toc()
	   end
   end
   output_circ_data = Array{Any}(4,8)
   output_circ_data[1,1] = 0
   output_circ_data[:,1] = runtime
   output_circ_data[:,2] = collect(0:3)
   for i in 3:8
      output_circ_data[:,i] = evidence
   end
   writedlm(filename, output_circ_data, ", ")
end

#=
include("calc_evidence_laplace_lowe.jl")
output1 = analyze_dataset("../data/rvs_0001.txt",num_Sigmas = 4,samples_per_peak=1,Pminsearch=[30.0,10.0,10.])
write_evidences_file("laplace_linearized_circ/evidences_0001.txt",output1)
write_evidences_file("laplace_linearized_ecc/evidences_0001.txt",output1,use_method=2)
@save "output_kepler_1.jld" 

output1 = analyze_dataset("../data/rvs_0001.txt",num_Sigmas = 40,samples_per_peak=4,Pminsearch=[30.0,10.0,10.0])

output1 = analyze_dataset("../data/rvs_0001.txt",num_Sigmas = 4,samples_per_peak=1,Pminsearch=[10.0,10.0,1.2])
output2 = analyze_dataset("../data/rvs_0002.txt",num_Sigmas = 4,samples_per_peak=1,Pminsearch=[10.0,10.0,1.2])
output3 = analyze_dataset("../data/rvs_0003.txt",num_Sigmas = 4,samples_per_peak=1,Pminsearch=[10.0,10.0,1.2])
output4 = analyze_dataset("../data/rvs_0004.txt",num_Sigmas = 4,samples_per_peak=1,Pminsearch=[10.0,10.0,1.2])
output5 = analyze_dataset("../data/rvs_0005.txt",num_Sigmas = 4,samples_per_peak=1,Pminsearch=[10.0,10.0,1.2])
output6 = analyze_dataset("../data/rvs_0006.txt",num_Sigmas = 4,samples_per_peak=1,Pminsearch=[10.0,10.0,1.2])
@save "output_kepler_4.jld" 
=#
#=  
include("calc_evidence_laplace_lowe.jl")
output1 = analyze_dataset("../data/rvs_0001.txt",num_Sigmas = 40,samples_per_peak=4,Pminsearch=[10.0,10.,1.2])
write_evidences_file("laplace_linearized_circ/evidences_0001.txt",output1)
write_evidences_file("laplace_linearized_ecc/evidences_0001.txt",output1,use_method=2)
@save "output_kepler_40_4_10.jld" 
output2 = analyze_dataset("../data/rvs_0002.txt",num_Sigmas = 40,samples_per_peak=4,Pminsearch=[10.0,10.,1.2])
write_evidences_file("laplace_linearized_circ/evidences_0002.txt",output2)
write_evidences_file("laplace_linearized_ecc/evidences_0002.txt",output2,use_method=2)
output3 = analyze_dataset("../data/rvs_0003.txt",num_Sigmas = 40,samples_per_peak=4,Pminsearch=[10.0,10.,1.2])
write_evidences_file("laplace_linearized_circ/evidences_0003.txt",output3)
write_evidences_file("laplace_linearized_ecc/evidences_0003.txt",output3,use_method=2)
output4 = analyze_dataset("../data/rvs_0004.txt",num_Sigmas = 40,samples_per_peak=4,Pminsearch=[10.0,10.,1.2])
write_evidences_file("laplace_linearized_circ/evidences_0004.txt",output4)
write_evidences_file("laplace_linearized_ecc/evidences_0004.txt",output4,use_method=2)
output5 = analyze_dataset("../data/rvs_0005.txt",num_Sigmas = 40,samples_per_peak=4,Pminsearch=[10.0,10.,1.2])
write_evidences_file("laplace_linearized_circ/evidences_0005.txt",output5)
write_evidences_file("laplace_linearized_ecc/evidences_0005.txt",output5,use_method=2)
output6 = analyze_dataset("../data/rvs_0006.txt",num_Sigmas = 40,samples_per_peak=4,Pminsearch=[10.0,10.,1.2])
write_evidences_file("laplace_linearized_circ/evidences_0006.txt",output6)
write_evidences_file("laplace_linearized_ecc/evidences_0006.txt",output6,use_method=2)
@save "output_kepler_40_4_10.jld" 
=#


#=
data = readdlm("../data/rvs_0001.txt")
times = data[:,1];
vels = data[:,2];
sigma_obs = data[:,3];
output1 = analyze_dataset("../data/rvs_0001.txt",num_Sigmas = 40,Pminsearch=[30.0,10.0,10.0])
param_tmp = make_kepler_param_from_linear_output(output1,1)
SigmaBig = PDMat(make_Sigma(times,sigma_obs,param_tmp[end]));
optimize_phases(param_tmp,times,vels,SigmaBig)
res1 = laplace_approximation_kepler_model_jitter_separate(param_tmp,times,vels,Sigma0)
res1[1]/log(10)

param_tmp = make_kepler_param_from_linear_output(output1,2)
SigmaBig = PDMat(make_Sigma(times,sigma_obs,param_tmp[end]));
optimize_phases(param_tmp,times,vels,SigmaBig)
res2 = laplace_approximation_kepler_model_jitter_separate(param_tmp,times,vels,Sigma0)
res2[1]/log(10)

param_tmp = make_kepler_param_from_linear_output(output1,3)
SigmaBig = PDMat(make_Sigma(times,sigma_obs,param_tmp[end]));
optimize_phases(param_tmp,times,vels,SigmaBig)
res3 = laplace_approximation_kepler_model_jitter_separate(param_tmp,times,vels,Sigma0)
res3[1]/log(10)

=#

#=
param1a = [42.0,5.0,0.01,0.01,1.,0.]
res = laplace_approximation_kepler_model(param1a,times,vels,SigmaBig,loglikelihood_fixed_jitter)
param1a = res[2].minimizer;
laplace_approximation_kepler_model_jitter_separate(param1a,times,vels,Sigma0)[1]/log(10)
=#


function compute_marginalized_likelihood_cuba(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, min_param::Vector{T}, max_param::Vector{T}; num_Sigmas::Integer = 20 ) where T<:Real
  @assert length(min_param)==length(max_param)
  log_normalization = 0.0
  Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.))
  Sigma_cache = make_Sigma_cache(num_Sigmas,Sigma0)
  function log_integrand_cuba(x::Vector)
    param = (max_param.-min_param).*x.+min_param
	
	logp = loglikelihood_marinalize_jitter(param,times,vels,Sigma_cache=Sigma_cache)
	#println("# param= ",param," logp= ",logp)
	logp
  end
  function integrand_cuba(x::Vector, output::Vector)
    output[1] = exp(log_integrand_cuba(x)-log_normalization)
  end

  log_normalization = log_integrand_cuba(0.5*ones(min_param))
  result = suave(integrand_cuba,length(min_param),1,reltol=1e-3,maxevals=10000)
  #result = log(result.integral[1]) + log(prod(max_param.-min_param))
  #result+log_normalization
end

function compute_marginalized_likelihood_cuba_near(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, param::Vector{T}; num_Sigmas::Integer = 20 ) where T<:Real
  const nparam_per_pl = 5
  const nparam_non_pl = 2
  npl = convert(Int64,floor((length(param)-1)//nparam_per_pl))
  param_min = similar(param); param_max = similar(param);
  for p in 1:npl
    param_min[1+(p-1)*nparam_per_pl] = param[1+(p-1)*nparam_per_pl]*0.98
    param_max[1+(p-1)*nparam_per_pl] = param[1+(p-1)*nparam_per_pl]*1.02
	param_min[2+(p-1)*nparam_per_pl] = param[2+(p-1)*nparam_per_pl]*0.5
    param_max[2+(p-1)*nparam_per_pl] = param[2+(p-1)*nparam_per_pl]*2.00
	param_min[3+(p-1)*nparam_per_pl] = -1.0
    param_max[3+(p-1)*nparam_per_pl] = 1.0
	param_min[4+(p-1)*nparam_per_pl] = -1.0
    param_max[4+(p-1)*nparam_per_pl] = 1.0
	param_min[5+(p-1)*nparam_per_pl] = 0.0
    param_max[5+(p-1)*nparam_per_pl] = 2pi
  end
  param_min[1+npl*nparam_per_pl] = param[1+npl*nparam_per_pl]-2.0
  param_max[1+npl*nparam_per_pl] = param[1+npl*nparam_per_pl]+2.0
  param_min[1+npl*nparam_per_pl] = 0.0
  param_max[1+npl*nparam_per_pl] = 2*param[2+npl*nparam_per_pl]
  res = compute_marginalized_likelihood_cuba(times,vels,sigma_obs,param_min,param_max,num_Sigmas=num_Sigmas)
end

#param_tmp = make_kepler_param_from_linear_output(output1,1)
#compute_marginalized_likelihood_cuba_near(times,vels,sigma_obs,param_tmp,num_Sigmas=3)
  

#=  
write_evidences_file("laplace_keplerian/evidences_0001.txt",output1,use_method=3,input_filename="../data/rvs_0001.txt")
write_evidences_file("laplace_keplerian/evidences_0002.txt",output1,use_method=3,input_filename="../data/rvs_0002.txt")
write_evidences_file("laplace_keplerian/evidences_0003.txt",output1,use_method=3,input_filename="../data/rvs_0003.txt")
write_evidences_file("laplace_keplerian/evidences_0004.txt",output1,use_method=3,input_filename="../data/rvs_0004.txt")
write_evidences_file("laplace_keplerian/evidences_0005.txt",output1,use_method=3,input_filename="../data/rvs_0005.txt")
write_evidences_file("laplace_keplerian/evidences_0006.txt",output1,use_method=3,input_filename="../data/rvs_0006.txt")
=#
