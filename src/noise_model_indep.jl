# Statistical model for analyzing Radial Velocity observations
# Assumes existance of data in arrays times, obs, sigma_obs
# Assumes existance of functions: TODO 

export loglikelihood, logprior_noise, is_valid_noise

num_noise_param() = 0    # Set to 1 for using jitter, 0 for none
const Jitter0 = 1.0  # units of m/s
const max_jitter = 10000.0
const jitter_prior_norm = -log(log1p(max_jitter/Jitter0))
jitter_offset(theta::Vector) = noise_param_offset(theta)

function set_jitter(theta::Vector, jitter) 
  if num_noise_param() == 1
     theta[jitter_offset(theta)] = log1p(jitter/Jitter0)
  end
  return
end

function extract_jitter(theta::Vector) 
  if num_noise_param() == 1
     return Jitter0*(exp(theta[jitter_offset(theta)])-1)
  else
     return 0
  end
end

function is_valid_noise(p::Vector)
  param = extract_noise_param(p)
  @assert length(param) == num_noise_param()
  if num_noise_param() == 1 
     jitter = extract_jitter(p)
     if !(0<=jitter<=max_jitter)
       return false
     end
  end
  return true
end

function loglikelihood(p::Vector)
  num_pl = num_planets(p)
  if !is_valid(p) return -Inf end  # prempt model evaluation
  # Set t, o, and so to point to global arrays with observational data, while enforcing types
  t::Array{Float64,1} = times
  o::Array{Float64,1} = obs
  so::Array{Float64,1} = sigma_obs
  @assert length(times) == length(obs) == length(sigma_obs)
  jitter_sq = num_noise_param() ==1 ? extract_jitter(p)^2 : 0.
  chisq = zero(eltype(p))
  log_normalization = -0.5*length(t)*log(2pi)
  for i in 1:length(t)
    model::eltype(p) = calc_model_rv(p,t[i])
    sigma_eff_sq = so[i]^2+jitter_sq
    chisq += abs2(model-o[i])/sigma_eff_sq
    log_normalization -= 0.5*log(sigma_eff_sq)
  end
  return -0.5*(chisq)+log_normalization
end

function logprior_noise(p::Vector)
  logp = zero(eltype(p))
  if num_noise_param() == 1 
    if !is_valid_noise(p) 
      logp -=Inf          # prempt model evaluation
      return logp
    end
    jitter::eltype(p) = extract_jitter(p)
    logp += -log1p(jitter/Jitter0) + jitter_prior_norm
  end
  return logp::eltype(p)
end

