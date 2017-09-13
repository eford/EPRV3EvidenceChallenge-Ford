using PDMats  # Positive Definite Matrices
using QuadGK  # 1-d integration
using Optim

# Read dataset
data = readdlm("../data/rvs_0001.txt");
times = data[:,1]
vels = data[:,2]
sigma_obs = data[:,3]
nobs = length(vels)

#
function compute_Fischer_matrix_offset{T<:Real}(data::Vector{T}, model::Array{T,2}, Sigma::PDMat{T})
  @assert length(data) == size(model,1) == size(Sigma,1)
  nparam = size(model,2)
  FIM = Array{Float64}(nparam,nparam)
  FIM[1,1] = sum(1.0./diag(Sigma))
  return FIM
end

function compute_Fischer_matrix_1sin{T<:Real}(data::Vector{T}, model::Array{T,2}, Sigma::PDMat{T})
  @assert length(data) == size(model,1) == size(Sigma,1)
  nparam = size(model,2)
  FIM = Array{Float64}(nparam,nparam)
  FIM[1,1] = sum(1.0./diag(Sigma))
  return FIM
end

const Cmax = 1000.0
function logprior_offset(param)
  -log(2*Cmax)
end

const Kmax = 999.0
const K0 = 1.0
function logprior_sinusoids(param)
  nparam = length(param)
  nsin = convert(Int64,floor((nparam-1)//2))
  logp = 0
  for i in 0:(nsin-1)
    K = sqrt(param[1+2*i]^2+param[2+2*i]^2)
    logp -= log1p(K/K0)+log(log1p(Kmax/K0))
  end
  C = param[nparam]
  if !(-Cmax<=C<=Cmax) logp -=Inf
  else  logp -= log(2*Cmax) end
  return logp
end

function calc_best_fit(data::Vector{T}, model::Array{T,2}, Sigma::PDMat{T}) where T<:Real
  inv_chol_Sigma = inv(Sigma.chol[:L])
  param_bf_linalg = (inv_chol_Sigma*model) \ (inv_chol_Sigma*data)
end

function calc_best_fit(data::Vector{T}, model::Array{T,2}, Sigma::PDiagMat{T}) where T<:Real
  inv_chol_Sigma = sqrt.(Sigma.inv_diag)
  param_bf_linalg = (inv_chol_Sigma.*model) \ (inv_chol_Sigma.*data)
end

function compute_Laplace_Approx(data::Vector{T}, model::Array{T,2}, Sigma::SigmaT; FIM::PDMat{T} = PDMat(Xt_invA_X(Sigma,X)), calclogprior::Function = x->0.0 ) where {T<:Real, SigmaT<:AbstractPDMat{T} }
  @assert length(data) == size(model,1) == size(Sigma,1)
  nobs = length(data)
  nparam = size(model,2)

  param_bf_linalg = calc_best_fit(data,model,Sigma)
  predict = X*param_bf_linalg
  
  #FIM = compute_Fischer_matrix_offset(data,model,Sigma)
  #FIM = compute_Fischer_matrix_1sin(data,model,Sigma)
  #chol_FIM = chol(FIM)
  #chol_FIM = chol(full(FIM))
  #sigma_param = inv(chol_FIM)
  sigma_param = inv(FIM).chol[:U]

  delta = data.-predict
  chisq = invquad(Sigma,delta) # sum(delta'*Sigma^-1*delta)

  loglikelihood_mode = 0.5*(-chisq-logdet(Sigma)-nobs*log(2pi))
  logprior_mode = calclogprior(param_bf_linalg) 
  log_mode = logprior_mode + loglikelihood_mode
  LaplaceApprox = 0.5*nparam*log(2pi)-0.5*logdet(FIM)+log_mode
  #LaplaceApprox = 0.5*nparam*log(2pi)-logdet(chol_FIM)+log_mode
  return (param_bf_linalg,sigma_param,LaplaceApprox)
end

function generate_K_samples(bf::Vector, sigma, nsamples::Integer = 1)
  nparam = length(bf)
  theta = bf .+ sigma*rand(nparam,nsamples)
  #theta = param_bf'.+rand(nsamples,nparam)*sigma_param
  namps = convert(Int64,floor((nparam-1)//2))

  Ks = zeros(nsamples,namps)
  for i in 1:namps
    Ks[:,i] = sqrt.(theta[1+2*(i-1),:].^2+theta[2+2*(i-1),:].^2)
  end
  return Ks
end

nparam = 1
X = ones(nobs,nparam); 
Sigma=PDiagMat(data[:,3].^2)
compute_Laplace_Approx(vels,X,Sigma) 

# Simplified zero planet model to test linear algebra
# Brute force best-fit C and sigma for comparison
C_bf = sum(data[:,2]./data[:,3].^2)/sum(data[:,3].^(-2))
sigma_bf_0 = sqrt(1/sum(data[:,3].^(-2)))

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
  PDMat( [ 0.0 + (i==j ? 2*sigma_j : 0.0) for i in 1:length(t), j in 1:length(t)] )
end

const sigma_j_min = 0.0
const sigma_j_max = 99.0
const sigma_j_0 = 1.0
function logprior_jitter(sigma_j::Real)
  -log1p(sigma_j/sigma_j_0)-log(sigma_j_0)-log(log1p(sigma_j_max/sigma_j_0))
end

function compute_log_evidence(sigma_j::Real)   
  Sigma = make_Sigma(times,sigma_obs,sigma_j)
  log_prior = logprior_jitter(sigma_j) + logprior_offset([])
  log_likelihood = compute_Laplace_Approx(vels,X,Sigma)[3]
  log_prior+log_likelihood
end
  
function compute_evidence(sigma_j::Real; log_normalization::Real=0.0)   
  exp(compute_log_evidence(sigma_j)-log_normalization)
end

#lognorm = compute_log_evidence(sigma_j_0)  # needed if underflow
integrand(x) = compute_evidence(x) # ,log_normalization=0.0)
result = QuadGK.quadgk(integrand,sigma_j_min,sigma_j_max)./log(10)

function calc_opt_sigmaj(data::Vector{T}, model::Array{T,2}) where T<:Real
  function calc_laplace_approx_arg_jitter(sigmaj::Real) 
  Sigma = PDMat(make_Sigma(times,sigma_obs,sigmaj))
    FIM = PDMat(Xt_invA_X(Sigma,X))
    (param_bf,sigma_param,evid) = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=logprior_sinusoids)
	evid += logprior_jitter(sigmaj)
    -evid
  end
  result = optimize(calc_laplace_approx_arg_jitter,sigma_j_min,sigma_j_max,rel_tol=1e-6)
end
res = calc_opt_sigmaj(vels,X)
dSigmadsigmaj = make_dSigmadsigmaj(times,sigma_obs,res.minimizer)
evid = (compute_log_evidence(res.minimizer)+0.5*(log(2pi)-log(0.5*trace((Sigma \ full(dSdj))^2))))/log(10)



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

using Cuba
function integrand_vegas(x::Vector, output::Vector; log_normalization::Real=0.0)   
  xx = zeros(2)
  xx[1] = x[1]*2*Cmax/100-Cmax/100
  xx[2] = x[2]*(sigma_j_max-sigma_j_min)+sigma_j_min
  output[1] = exp(log_target(xx)-log_normalization)*(2*Cmax/100)*(sigma_j_max-sigma_j_min)
end
@time vegas(integrand_vegas,2,1,reltol=1e-3,maxevals=1e5)
@time suave(integrand_vegas,2,1,reltol=1e-3,maxevals=1e5)
@time divonne(integrand_vegas,2,1,reltol=1e-3,maxevals=1e5)
@time cuhre(integrand_vegas,2,1,reltol=1e-3,minevals=1e3,maxevals=1e5)

#=
# Brute Force:  Sequence of Nested 1-d integrations
function compute_log_target_fixed_offset(offset::Real)   
  delta = vels.-offset
  function integrand(sigma_j::Real)
    logprior = logprior_offset([offset]) + logprior_jitter(sigma_j)  
    Sigma = make_Sigma(times,sigma_obs,sigma_j)
    chisq = invquad(Sigma,delta) 
    loglikelihood = 0.5*(-chisq -logdet(Sigma)-nobs*log(2pi))
    logtarget = logprior + loglikelihood 
	exp(logtarget) # +log(1e210))
  end
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

# Fit planet models
function make_design_matrix(times::Vector{T},periods::Vector{T}) where T<:Real
  nparam = 1+2*length(periods)
  freqs = 2pi./periods
  X = Array{T}(length(times),nparam)
  for i in 1:length(periods)
     X[:,1+(i-1)*2] = sin(freqs[i]*times)
	 X[:,2+(i-1)*2] = cos(freqs[i]*times)
  end
  X[:,nparam] = 1.0
  return X
end

# Fit one planet model
periods = [42.0]
nparam = 1+2*length(periods)
freqs = 2pi./periods
X = [sin(freqs[1]*times) cos(freqs[1]*times) ones(times)];
Sigma = PDMat(make_Sigma(times,sigma_obs,sigma_j_0))
FIM = PDMat(Xt_invA_X(Sigma,X))
(param_bf,sigma_param,evid) = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=logprior_sinusoids)

function calc_opt_sigmaj(data::Vector{T}, model::Array{T,2}) where T<:Real
  function calc_laplace_approx_arg_jitter(sigmaj::Real) 
	Sigma = PDMat(make_Sigma(times,sigma_obs,sigmaj))
    FIM = PDMat(Xt_invA_X(Sigma,X))
    (param_bf,sigma_param,evid) = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=logprior_sinusoids)
    evid += logprior_jitter(sigmaj)
	-evid
  end
  result = optimize(calc_laplace_approx_arg_jitter,sigma_j_min,sigma_j_max,rel_tol=1e-4)
end
res = calc_opt_sigmaj(vels,X)
Sigma = PDMat(make_Sigma(times,sigma_obs,res.minimizer))
dSigmadsigmaj = make_dSigmadsigmaj(times,sigma_obs,res.minimizer)
FIM = PDMat(Xt_invA_X(Sigma,X))
(param_bf,sigma_param,evid) = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=logprior_sinusoids)
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
    (param_bf,sigma_param,evid) = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=logprior_sinusoids)
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
(param_bf,sigma_param,evid) = compute_Laplace_Approx(vels,X,Sigma,FIM=FIM,calclogprior=logprior_1sin)

Ks = generate_K_samples(param_bf,sigma_param,1000)
m = mean(Ks,1)
sig = std(Ks,1,mean=m)
m-sig,m+sig
