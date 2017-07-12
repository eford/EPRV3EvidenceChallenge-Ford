# Statistical model for analyzing Radial Velocity observations
# Assumes existance of data in arrays times, obs, sigma_obs
# Assumes existance of functions: TODO

export loglikelihood, logprior_noise, is_valid_noise

num_noise_param() = 4

if(!isdefined(:PDMats))        using PDMats       end

function is_valid_noise(p::Vector)
  param = extract_noise_param(p)
  @assert length(param) == num_noise_param()
  sigma_sq_cor = param[1]
  tau_evolve   = param[2]
  lambda_p     = param[3]
  tau_rotate   = param[4]
  if !(0<=sigma_sq_cor) || !(0<tau_evolve) || !(0<lambda_p) || !(0<tau_rotate)
     return false
  end
  return true
end

function kernel_quaesiperiodic(dt::Float64,param::Vector)
  sigma_sq_cor = param[1]
  tau_evolve   = param[2]
  lambda_p     = param[3]
  tau_rotate   = param[4]
  sigma_sq_cor*exp(-0.5*((dt/tau_evolve)^2 + (sin(pi*dt/tau_rotate)/lambda_p)^2) )
end

function loglikelihood(p::Vector)
  num_pl = num_planets(p)
  @assert num_pl >= 1
  if !is_valid(p) return -Inf end  # prempt model evaluation
  # Set t, o, and so to point to global arrays with observational data, while enforcing types
  t::Array{Float64,1} = times
  o::Array{Float64,1} = obs
  so::Array{Float64,1} = sigma_obs
  @assert length(times) == length(obs) == length(sigma_obs)
  #gp_param = [3.0, 50., 0.5, 20.]   # TODO: allow to be parameters
  gp_param = extract_noise_param(p)
  model_minus_data = map(i->calc_model_rv(p,t[i])-o[i],1:length(t))
  Sigma = PDMat( [ kernel_quaesiperiodic(t[i]-t[j],gp_param) + (i==j ? so[i]^2 : 0.0) for i in 1:length(t), j in 1:length(t)] )

  # chisq = model_minus_data'*(Sigma*model_minus_data)
  chisq = invquad(Sigma, model_minus_data)
  log_normalization = -0.5*( length(t)*log(2pi) + logdet(Sigma) )
  return -0.5*chisq+log_normalization
end

function logprior_noise(p::Vector)
  logp = zero(eltype(p))
  if !is_valid_noise(p)
     logp -= Inf
     return logp
  end
  param = extract_noise_param(p)
  @assert length(param) == num_noise_param()
  sigma_cor = sqrt(param[1])
  # tau_evolve   = param[2]
  # lambda_p     = param[3]
  # tau_rotate   = param[4]
  const Jitter0 = 1.0
  const max_jitter = 10000.0
  const jitter_prior_norm = log(log1p(max_jitter/Jitter0))
  logp += -log1p(sigma_cor/Jitter0) + jitter_prior_norm
  # TODO: Add prior for other GP parameters
  return logp::eltype(p)
end

