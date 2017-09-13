using PDMats  # Positive Definite Matrices
using QuadGK  # 1-d integration
using Cuba    # multi-d integration
using Optim   # Optimizing over non-linear parameters


# Read dataset
data = readdlm("../data/rvs_0001.txt");
times = data[:,1]
vels = data[:,2]
sigma_obs = data[:,3]
nobs = length(vels)

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

function calc_chisq(data::Vector{T}, model::Array{T,2}, Sigma::SigmaT; FIM::PDMat{T} = PDMat(Xt_invA_X(Sigma,model)), calclogprior::Function = x->(0.0,0.0) ) where {T<:Real, SigmaT<:AbstractPDMat{T} }
  @assert length(data) == size(model,1) == size(Sigma,1)
  nobs = length(data)
  nparam = size(model,2)

  param_bf_linalg = calc_best_fit(data,model,Sigma)
  predict = model*param_bf_linalg
  delta = data.-predict
  chisq = invquad(Sigma,delta) # sum(delta'*Sigma^-1*delta)
end

function compute_Laplace_Approx_param_and_integral(data::Vector{T}, model::Array{T,2}, Sigma::SigmaT; FIM::PDMat{T} = PDMat(Xt_invA_X(Sigma,model)), calclogprior::Function = x->(0.0,0.0) ) where {T<:Real, SigmaT<:AbstractPDMat{T} }
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

function compute_Laplace_Approx(data::Vector{T}, model::Array{T,2}, Sigma::SigmaT; FIM::PDMat{T} = PDMat(Xt_invA_X(Sigma,model)), calclogprior::Function = x->(0.0,0.0) ) where {T<:Real, SigmaT<:AbstractPDMat{T} }
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

#=
function compute_log_evidence(sigma_j::Real)   
  Sigma = make_Sigma(times,sigma_obs,sigma_j)
  log_prior_outside = logprior_jitter(sigma_j) + logprior_periods(periods)
  log_likelihood = compute_Laplace_Approx(vels,X,Sigma,calclogprior=logprior_epicycles)
  log_prior_outside+log_likelihood
end
  
function compute_evidence(sigma_j::Real; log_normalization::Real=0.0)   
  exp(compute_log_evidence(sigma_j)-log_normalization)
end
=#

#=
periods = [42.0]
X = make_design_matrix_epicycle(times,periods);
Sigma = PDMat(make_Sigma(times,sigma_obs,sigma_j_0))
FIM = PDMat(Xt_invA_X(Sigma,X))
(param_bf,sigma_param,evid) = compute_Laplace_Approx_param_and_integral(vels,X,Sigma,FIM=FIM,calclogprior=logprior_epicycles)

(Ks,es) = generate_Ke_samples(param_bf,sigma_param,1000)
=#

#=
lognorm = compute_log_evidence(sigma_j_0)  # can help to prevent underflow
integrand(x) = compute_evidence(x,log_normalization=lognorm)
result = QuadGK.quadgk(integrand,sigma_j_min,sigma_j_max).*(exp(lognorm)/log(10))
=#

function calc_opt_sigmaj(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, model::Array{T,2}; calclogprior_linear::Function = logprior_sinusoids) where T<:Real
  function calc_laplace_approx_arg_jitter(sigmaj::Real) 
    Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))
    FIM = PDMat(Xt_invA_X(Sigma,model))
    evid = compute_Laplace_Approx(vels,model,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
	evid += logprior_jitter(sigmaj)
    -evid
  end
  Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.))
  result = optimize(calc_laplace_approx_arg_jitter,sigma_j_min,sigma_j_max,rel_tol=1e-6)
end

#= 
periods = [42.0]
X = make_design_matrix_sinusoids(times,periods);
res = calc_opt_sigmaj(times,vels,sigma_obs,X)
dSdj = make_dSigmadsigmaj(times,sigma_obs,res.minimizer)
(compute_log_evidence(res.minimizer)+0.5*(log(2pi)-log(0.5*trace((Sigma \ full(dSdj))^2))))/log(10)
=#

#=
function make_matrix_pd(A::Array{Float64,2}; epsabs::Float64 = 0.0, epsfac::Float64 = 1.0e-6)
  @assert(size(A,1)==size(A,2))
  B = copy(A)
  itt = 1
  while !isposdef(B)
	eigvalB,eigvecB = eig(B)
	neweigval = (epsabs == 0.0) ? epsfac*minimum(eigvalB[eigvalB.>0]) : epsabs
	eigvalB[eigvalB.<0] = neweigval
	B = eigvecB *diagm(eigvalB)*eigvecB'
#	println(itt,": ",B)
	itt +=1
	if itt>size(A,1) 
	  error("There's a problem in make_matrix_pd.\n")
  	  break 
	end
  end
  return B
end
=#

# Integrate over period
function compute_marginalized_likelihood_fix_periods(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods::Vector{T}; calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, X = make_design_matrix(times,periods), Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.0)) ) where T<:Real
   const nparam = size(X,2)

   function optimize_jitter_helper(sigmaj::Real) 
     Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))
	 FIM = Xt_invA_X(Sigma,X)
	 if !isposdef(FIM)
	   FIM += diagm(1e-3*ones(nparam))
	 end
	 FIM = PDMat(FIM)
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
   FIM = Xt_invA_X(Sigma,X)
	 if !isposdef(FIM)
	   FIM += diagm(1e-3*ones(nparam))
	 end
   FIM = PDMat(FIM)

   logp = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=calclogprior_linear)
   logp += logprior_jitter(sigmaj)+ logprior_periods(periods)
   logp += 0.5*(log(2pi)-log(0.5*trace((Sigma \ full(dSdj))^2)))
   #println("# logp = ",logp)
   return logp 
end


function compute_marginalized_likelihood_fix_periods_sigmaj(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods::Vector{T}, sigmaj::T; calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, Sigma=PDMat(make_Sigma(times,sigma_obs,sigmaj)), X= make_design_matrix(times,periods) ) where T<:Real
   #X = make_design_matrix(times,periods)
   const nparam = size(X,2)
   #Sigma = PDMat(make_Sigma(times,sigma_obs,sigmaj))
   #dSdj = make_dSigmadsigmaj(times,sigma_obs,sigmaj)
   FIM = Xt_invA_X(Sigma,X)
	 if !isposdef(FIM)
	   FIM += diagm(1e-3*ones(nparam))
	 end
   FIM = PDMat(FIM)

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

compute_marginalized_likelihood(times,vels,sigma_obs,[11.9],[12.1])./log(10)

#=
compute_marginalized_likelihood(times,vels,sigma_obs,[11.9],[12.1],calclogprior_linear=logprior_epicycles,make_design_matrix=make_design_matrix_epicycle)./log(10)

compute_marginalized_likelihood(times,vels,sigma_obs,[10.],[2400.])./log(10)
=#



function logsumlogs(x::T,y::T) where T<:Real
  (x<=y) ? y + log1p(exp(x-y)) : x + log1p(exp(y-x)) 
end

#=
function brute_force_over_periods_1d_old(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, Plo::Real, Phi::Real; samples_per_peak::Real = 4.0, calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ ) where T<:Real
  min_freq = 1/Phi
  max_freq = 1/Plo
  duration = maximum(times)-minimum(times)
  delta_freq = 1/(samples_per_peak * duration)
  log_delta_freq = log(delta_freq)
  range = min_freq:delta_freq:max_freq
  logintegral = -Inf
  for f in range
    P = 1/f
	logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,[P], calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix)
	logp -= log(f)
	if logintegral == -Inf
	  logintegral = logp+log_delta_freq
	else
	  logintegral = logsumlogs(logintegral,logp+log_delta_freq)
	end
  end
  logintegral
end

function brute_force_over_periods_1d(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, Plo::Real, Phi::Real; samples_per_peak::Real = 1.0, calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, sigmaj::Real = -1.0 ) where T<:Real
  min_freq = 1/Phi
  max_freq = 1/Plo
  duration = maximum(times)-minimum(times)
  mean_vel = sum(vels./sigma_obs.^2)/sum(sigma_obs.^(-2))
  rms_vel = std((vels.-mean_vel)./sigma_obs)
  num_periods = convert(Int64,ceil(samples_per_peak*2pi*duration*rms_vel*(max_freq-min_freq)))
  println("# num_periods = ",num_periods)
  deltaP_const = 1/(samples_per_peak*2pi*duration*rms_vel)
  P = Plo
  periods = Array(Float64,2*num_periods)
  powers  = Array(Float64,2*num_periods)
  logintegral = -Inf
  i = 0
  while P<=Phi
    deltaP = deltaP_const*P^2
	if i >0 P += deltaP end
    i += 1
	if sigmaj < 0
	   logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,[P], calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix)
	else
	   logp = compute_marginalized_likelihood_fix_periods_sigmaj(times,vels,sigma_obs,[P],sigmaj, calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix)
	   

	   logp += logprior_jitter(sigmaj)
	end
	logp += log(deltaP/P)
    if i%10 == 0
	  println("#i=",i," P=",P," dP=",deltaP," logp=",logp," logI=",logintegral)
	end
	if logintegral == -Inf
	  logintegral = logp
	else
	  logintegral = logsumlogs(logintegral,logp)
	end
	@assert i<=length(periods)
	periods[i] = P
	powers[i] = logp
    
  end
  resize!(periods,i)
  resize!(powers,i)
  println("# Last P = ", P)
  return (periods,powers,logintegral)
end

res = brute_force_over_periods_1d(times,vels,sigma_obs,20.0,10000.0,samples_per_peak=1,sigmaj = 1.5)


result = QuadGK.quadgk(x->exp(brute_force_over_periods_1d(times,vels,sigma_obs,20.0,10000.0,samples_per_peak=1,sigmaj=x)),0.0,10.0,reltol=1e-2)

res = brute_force_over_periods_1d(times,vels,sigma_obs,20.0,10000.0,samples_per_peak=1)

=#

using FastGaussQuadrature
num_Sigmas = 40
nodes, weights = gausslegendre( num_Sigmas );
weights *= 0.5;
nodes = 0.5+0.5*nodes;
sigmaj_list = sigma_j_0*(exp.(nodes*log1p(sigma_j_max/sigma_j_0)).-1);
Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.));
SigmaList = map(sigmaj->Sigma0+ScalMat(length(times),sigmaj),sigmaj_list);

marginalized_likelihood_fix_periods_sigmaj = map(i->compute_marginalized_likelihood_fix_periods_sigmaj(times, vels, sigma_obs, [12.1], sigmaj_list[i], Sigma=SigmaList[i]),1:length(sigmaj_list))
int_norm = maximum(marginalized_likelihood_fix_periods_sigmaj);
integral = log(dot( weights, exp.(marginalized_likelihood_fix_periods_sigmaj-int_norm) ))+int_norm
compute_marginalized_likelihood_fix_periods(times, vels, sigma_obs, [12.1] )  

function brute_force_over_periods_1d_new(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, Plo::Real, Phi::Real; samples_per_peak::Real = 1.0, calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, sigmaj::Real = -1.0, num_Sigmas::Integer = 40 ) where T<:Real
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
     nodes, weights = gausslegendre( num_Sigmas );
     weights *= 0.5;
     nodes = 0.5+0.5*nodes;
     sigmaj_list = sigma_j_0*(exp.(nodes*log1p(sigma_j_max/sigma_j_0)).-1);
     SigmaList = map(s->Sigma0+ScalMat(length(times),s),sigmaj_list);
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
	   marginalized_likelihood_fix_periods_sigmaj = map(k->compute_marginalized_likelihood_fix_periods_sigmaj(times, vels, sigma_obs, periods, sigmaj_list[k], Sigma=SigmaList[k],calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix, X=X),1:length(sigmaj_list))
       int_norm = maximum(marginalized_likelihood_fix_periods_sigmaj);
       logp = log(dot( weights, exp.(marginalized_likelihood_fix_periods_sigmaj-int_norm) ))+int_norm

    else
	   if sigmaj < 0
	      logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,periods,Sigma0=Sigma0, calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix,X=X)  
	   else
	      logp = compute_marginalized_likelihood_fix_periods_sigmaj(times,vels,sigma_obs,periods,sigmaj, Sigma=Sigma, calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix,X=X)  
	      logp += logprior_jitter(sigmaj)
	   end

    end
	logp += log(deltaP/P)
    if i%10 == 0
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
    
  end
  resize!(period_grid,i)
  resize!(power_grid,i)
  println("# Last P = ", period_grid[end])
  return (period_grid,power_grid,logintegral)
end

res = brute_force_over_periods_1d_new(times,vels,sigma_obs,20.0,10000.0,samples_per_peak=1,num_Sigmas=1)





function brute_force_over_periods_multiplanet_1d_old(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods_fixed::Vector{T}, Plo::Real, Phi::Real; samples_per_peak::Real = 4.0, calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, update_design_matrix::Function = update_design_matrix_circ ) where T<:Real
  periods = zeros(length(periods_fixed)+1)
  periods[1:length(periods_fixed)] = periods_fixed
  periods[end] = (Plo+Phi)/2
  X = make_design_matrix(times,periods)
  min_freq = 1/Phi
  max_freq = 1/Plo
  duration = maximum(times)-minimum(times)
  delta_freq = 1/(samples_per_peak * duration)
  log_delta_freq = log(delta_freq)
  range = min_freq:delta_freq:max_freq
  logintegral = -Inf
  for f in range
    periods[end] =  1/f
	update_design_matrix(X,times,periods,length(periods))
	logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,periods, calclogprior_linear=calclogprior_linear, X=X)
	logp -= log(f)
	if logintegral == -Inf
	  logintegral = logp+log_delta_freq
	else
	  logintegral = logsumlogs(logintegral,logp+log_delta_freq)
	end
  end
  logintegral
end

function brute_force_over_periods_multiplanet_1d(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods_fixed::Vector{T}, Plo::Real, Phi::Real; samples_per_peak::Real = 2.0, calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ, update_design_matrix::Function = update_design_matrix_circ, num_Sigmas::Integer = 40 ) where T<:Real
  periods = zeros(length(periods_fixed)+1)
  periods[1:length(periods_fixed)] = periods_fixed

  # Compute how much signal is remaining so can choose appropriate grid density
  X = make_design_matrix(times,periods_fixed)
  sigmaj = calc_opt_sigmaj(times, vels, sigma_obs, X, calclogprior_linear=calclogprior_linear).minimizer
  Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.))
  Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))
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
     nodes, weights = gausslegendre( num_Sigmas );
     weights *= 0.5;
     nodes = 0.5+0.5*nodes;
     sigmaj_list = sigma_j_0*(exp.(nodes*log1p(sigma_j_max/sigma_j_0)).-1);
     Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.));
     SigmaList = map(sigmaj->Sigma0+ScalMat(length(times),sigmaj),sigmaj_list);
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
	   #logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,periods, calclogprior_linear=calclogprior_linear, X=X, Sigma0=Sigma0) # make_design_matrix=make_design_matrix)
	   logp = compute_marginalized_likelihood_fix_periods_sigmaj(times,vels,sigma_obs,periods,sigmaj, calclogprior_linear=calclogprior_linear, X=X, Sigma=Sigma) # make_design_matrix=make_design_matrix)
	else
	   marginalized_likelihood_fix_periods_sigmaj = map(k->compute_marginalized_likelihood_fix_periods_sigmaj(times, vels, sigma_obs, periods, sigmaj_list[k], Sigma=SigmaList[k],calclogprior_linear=calclogprior_linear, make_design_matrix=make_design_matrix, X=X),1:length(sigmaj_list))
       int_norm = maximum(marginalized_likelihood_fix_periods_sigmaj);
       logp = log(dot( weights, exp.(marginalized_likelihood_fix_periods_sigmaj-int_norm) ))+int_norm
	end
	
	logp += log(deltaP/P)
    if i%100 == 0
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
  println("# Last P = ", P)
  (period_grid,power_grid,logintegral)
end

  

# Optimzie periods
function compute_marginalized_likelihood_near_periods(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, periods::Vector{T}; calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ ) where T<:Real
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
   #=
   X = make_design_matrix(times,periods)
   result = optimize(optimize_jitter_helper,sigma_j_min,sigma_j_max,rel_tol=1e-6)
   #println("# Optimize jitter: ", result)
   sigmaj = result.minimizer
   #Sigma = PDMat(make_Sigma(times,sigma_obs,sigmaj))
   Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))
   result2 = optimize(optimize_period_helper,periods[1]*0.98,periods[1]*1.02,rel_tol=1e-6)
   println("# Optimize period: ", result2)
   best_period = result2.minimizer
   =#
   
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

   # Laplace approx to integrate over period (one peak only)
   sigma_P = zeros(length(periods))
   log_int_over_periods = 0
   for i in 1:length(best_period)
     periods = copy(best_period)
     eps = 1e-5*periods[i]
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
     sigma_P[i] = sqrt(-1/d2logpdp2)
     println("# d^2 logp/dP_",i,"^2 = ",d2logpdp2/log(10))
     log_int_over_periods += 0.5*(log(2pi)-log(abs(d2logpdp2)))
   end
   logp += log_int_over_periods # Approx as diagonal
   println("# marginal over P at ",best_period," and sigma_j at ",sigmaj," : ",logp/log(10))
   
   return (param_bf,sigma_param,logp,best_period,sigma_P)
end

res0 = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,Float64[])/log(10)
res1  = compute_marginalized_likelihood_near_periods(times,vels,sigma_obs,[42.0])[3] /log(10)
res2  = compute_marginalized_likelihood_near_periods(times,vels,sigma_obs,[42.0,12.1])[3]/log(10)

res1e = compute_marginalized_likelihood_near_periods(times,vels,sigma_obs,[42.0],calclogprior_linear=logprior_epicycles,make_design_matrix=make_design_matrix_epicycle)  

res2e = compute_marginalized_likelihood_near_periods(times,vels,sigma_obs,[42.0,12.0],calclogprior_linear=logprior_epicycles,make_design_matrix=make_design_matrix_epicycle)  

res2b =  brute_force_over_periods_multiplanet_1d(times,vels,sigma_obs,[42.0777],10.0,2400.0,samples_per_peak=8)
 
 
function compute_marginalized_likelihood_cuba(times::Vector{T}, vels::Vector{T}, sigma_obs::Vector{T}, minperiods::Vector{T}, maxperiods::Vector{T}; calclogprior_linear::Function = logprior_sinusoids, make_design_matrix::Function = make_design_matrix_circ  ) where T<:Real
  @assert length(minperiods)==length(maxperiods)
  log_normalization = 0.0
  function helper_1d(period::T) # Note, this exp's unlike multi-d helper
    #println("# Evaluating P= ",period) 
	#flush(STDOUT)
    logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,[period],calclogprior_linear=calclogprior_linear,make_design_matrix=make_design_matrix)
	#println("#      ", logp) 
	#flush(STDOUT)
	return (logp == -Inf) ? 0. : exp(logp-log_normalization)
  end
  function helper(periods::Vector{T})
    #println("# Evaluating P= ",periods) 
	#flush(STDOUT)
     logp = compute_marginalized_likelihood_fix_periods(times,vels,sigma_obs,periods,calclogprior_linear=calclogprior_linear,make_design_matrix=make_design_matrix)
	#println("#      ", logp) 
	#flush(STDOUT)
	return logp
	# (logp == -Inf) ? 0. : exp(logp-log_normalization)
  end
  function integrand_cuba(x::Vector, output::Vector)   
    @assert length(output) == 1
	#println("# x=",x)
    periods = (maxperiods.-minperiods).*x.+minperiods
    output[1] = exp(helper(periods)-log_normalization)
	println("# P= ",periods," logp= ",output[1])
  end
  
  log_normalization = helper(0.5*(minperiods.+maxperiods))
  if length(minperiods)==1
     result = QuadGK.quadgk(helper_1d,minperiods[1],maxperiods[1],reltol=1e-3)
	 result = log(result[1])
     #result = divonne(integrand_cuba,1,1,reltol=1e-3,maxevals=100)*prod(maxperiods.-minperiods) # Problem?
  else
  
     result = divonne(integrand_cuba,length(minperiods),1,reltol=1e-3,maxevals=1000)
	 result = log(result.integral[1]) + log(prod(maxperiods.-minperiods))
  end 
  result+log_normalization
end

res = compute_marginalized_likelihood_cuba(times,vels,sigma_obs,[41.9,11.9],[42.3,12.2])

#=
@time vegas(integrand_cuba,2,1,reltol=1e-3,maxevals=1e5)
@time suave(integrand_cuba,2,1,reltol=1e-3,maxevals=1e5)
@time divonne(integrand_cuba,2,1,reltol=1e-3,maxevals=1e5)
@time cuhre(integrand_cuba,2,1,reltol=1e-3,minevals=1e3,maxevals=1e5)
=#



#=
# Brute force 2-d integration
using Cubature

function log_target(x::Vector)
  offset = x[1]
  sigma_j = x[2]
  logprior = logprior_offset([offset]) + logprior_jitter(sigma_j)  
  Sigma = make_Sigma(times,sigma_obs,sigma_j)
  delta = vels.-offset
  chisq = invquad(Sigma,delta) 
  loglikelihood = 0.5*(-chisq -logdet(Sigma)-nobs*log(2pi))
  logprior + loglikelihood 
end

function integrand(x::Vector; log_normalization::Real=0.0)   
  exp(log_target(x)-log_normalization)
end
@time hcubature(integrand, [-Cmax/100,sigma_j_min],[Cmax/100,sigma_j_max],reltol=1e-2)
#@time pcubature(integrand, [-Cmax/100,sigma_j_min],[Cmax/100,sigma_j_max],reltol=1e-2)
=#
#=
using Cuba
function integrand_cuba(x::Vector, output::Vector; log_normalization::Real=0.0)   
  xx = zeros(2)
  xx[1] = x[1]*2*Cmax/100-Cmax/100
  xx[2] = x[2]*(sigma_j_max-sigma_j_min)+sigma_j_min
  output[1] = exp(log_target(xx)-log_normalization)*(2*Cmax/100)*(sigma_j_max-sigma_j_min)
end
@time vegas(integrand_cuba,2,1,reltol=1e-3,maxevals=1e5)
@time suave(integrand_cuba,2,1,reltol=1e-3,maxevals=1e5)
@time divonne(integrand_cuba,2,1,reltol=1e-3,maxevals=1e5)
@time cuhre(integrand_cuba,2,1,reltol=1e-3,minevals=1e3,maxevals=1e5)
=#
#=
# Brute Force:  Sequence of Nested 1-d integrations
function compute_log_target_fixed_offset(offset::Real)   
  delta = vels.-offset
  function integrand(sigma_j::Real)
    logprior = logprior_offset([offset]) + logprior_jitter(sigma_j)  
    Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))
    chisq = invquad(Sigma,delta) 
    loglikelihood = 0.5*(-chisq -logdet(Sigma)-nobs*log(2pi))
    logtarget = logprior + loglikelihood 
	exp(logtarget) # +log(1e210))
  end
  Sigma0 = make_Sigma(times,sigma_obs,0.)
  result = QuadGK.quadgk(integrand,sigma_j_min,sigma_j_0,sigma_j_max,reltol=1e-2)[1]
end
@time result = QuadGK.quadgk(x->begin p=compute_log_target_fixed_offset(x); println(x," ",p); p end,-Cmax/100,Cmax/100,reltol=1e-2)./log(10)

  
# Brute Force: 2-d integration  
function log_target(x::Vector)
  offset = x[1]
  sigma_j = x[2]
  logprior = logprior_offset([offset]) + logprior_jitter(sigma_j)  
  Sigma = make_Sigma(times,sigma_obs,sigma_j)
  delta = vels.-offset
  chisq = invquad(Sigma,delta) 
  loglikelihood = 0.5*(-chisq -logdet(Sigma)-nobs*log(2pi))
  logprior + loglikelihood 
end
=#

#=
# Fit one planet model
periods = [42.0]
nparam = 1+2*length(periods)
freqs = 2pi./periods
X = [sin(freqs[1]*times) cos(freqs[1]*times) ones(times)];
Sigma = PDMat(make_Sigma(times,sigma_obs,sigma_j_0))
FIM = PDMat(Xt_invA_X(Sigma,X))
(param_bf,sigma_param,evid) = compute_Laplace_Approx_param_and_integral(vels,X,Sigma,FIM=FIM,calclogprior=logprior_sinusoids)

function calc_opt_sigmaj(data::Vector{T}, model::Array{T,2}) where T<:Real
  function calc_laplace_approx_arg_jitter(sigmaj::Real) 
	Sigma = Sigma0 + ScalMat(length(times),sigmaj) # PDMat(make_Sigma(times,sigma_obs,sigmaj))
    FIM = PDMat(Xt_invA_X(Sigma,X))
    evid = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=logprior_sinusoids)
    evid += logprior_jitter(sigmaj)
	-evid
  end
  Sigma0 = make_Sigma(times,sigma_obs,0.)
  result = optimize(calc_laplace_approx_arg_jitter,sigma_j_min,sigma_j_max,rel_tol=1e-4)
end
res = calc_opt_sigmaj(vels,X)
Sigma = PDMat(make_Sigma(times,sigma_obs,res.minimizer))
dSigmadsigmaj = make_dSigmadsigmaj(times,sigma_obs,res.minimizer)
FIM = PDMat(Xt_invA_X(Sigma,X))
(param_bf,sigma_param,evid) = compute_Laplace_Approx_param_and_integral(vels,X,Sigma,FIM=FIM,calclogprior=logprior_sinusoids)
evid += 0.5*(log(2pi)-log(0.5*trace((Sigma \ full(dSdj))^2)))



Ks = generate_K_samples(param_bf,sigma_param,1000);
m = mean(Ks,1);
sig = std(Ks,1,mean=m);
m-sig,m,m+sig


function profile_sigmaj(data::Vector{T}, model::Array{T,2}) where T<:Real
  function calc_laplace_approx_arg_jitter(sigmaj::Real) #param::Vector{T})
    #sigmaj = param[1]
	Sigma = PDMat(make_Sigma(times,sigma_obs,sigmaj))
    FIM = PDMat(Xt_invA_X(Sigma,X))
    evid = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=logprior_sinusoids)
	evid += logprior_jitter(sigmaj)
    evid
  end
  #result = optimize(calc_laplace_approx_arg_jitter,[sigma_j_0])
  result = optimize(calc_laplace_approx_arg_jitter,sigma_j_min,sigma_j_max,rel_tol=1e-6)
end
res = calc_opt_sigmaj(vels,X)


# Fit two planet model
periods = [42.0,12.0]
X = make_design_matrix(times,periods)
Sigma = PDMat(make_Sigma(times,sigma_obs,0.68442348767604010984))
FIM = PDMat(Xt_invA_X(Sigma,X))
(param_bf,sigma_param,evid) = compute_Laplace_Approx_param_and_integral(vels,X,Sigma,FIM=FIM,calclogprior=logprior_sinusoids)

Ks = generate_K_samples(param_bf,sigma_param,1000)
m = mean(Ks,1)
sig = std(Ks,1,mean=m)
m-sig,m+sig

=#