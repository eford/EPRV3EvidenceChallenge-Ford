# Statistical model for analyzing Radial Velocity observations
# Assumes existance of data in arrays times, obs, sigma_obs
# Assumes existance of functions: loglikelihood(theta), logprior_noise(theta), and is_valid_noise(theta)

export logprior_planets, logprior, logtarget, negative_logtarget
export set_times, set_obs, set_sigma_obs, num_noise_param

# Observational data to compare model to
global times # ::Array{Float64,1}
global obs # ::Array{Float64,1}
global sigma_obs #::Array{Float64,1}

# Functions to set global data within module
function set_times(t::Array{Float64,1}) global times = t   end
function set_obs(o::Array{Float64,1})  global obs = o   end
function set_sigma_obs(so::Array{Float64,1}) global sigma_obs = so  end

function is_valid(p::Vector)
   is_valid_noise(p) && is_valid_planets(p)
end

function logprior_period(P::Real)
  const min_period = 1.0
  const max_period = 10000.0
  const norm = -log(max_period/min_period)
  if !(min_period<=P<=max_period)
     return -Inf
  end
  return norm-log(P)
end

function logprior_amplitude(K::Real)
  const min_K = 0.0
  const max_K = 999.0
  const K0 = 1.0
  const norm = -log(K0*log1p(maxK/min_K))
  if !(min_K<=K<=max_K)
     return -Inf
  end
  return norm-log1p(K/K0)
end

function logprior_planets(p::Vector) 
  if !is_valid_planets(p) return -Inf end  # prempt model evaluation
  num_pl = num_planets(p)
  logp = zero(eltype(p))
  if num_pl <= 0 return logp end
  if num_param_per_planet == 3
     logp -= num_pl*log(2pi)
  elseif num_param_per_planet == 5
     logp -= 2*num_pl*log(2pi)
  end
  for plid in 1:num_pl
    P::eltype(p) = extract_period(p,plid=plid)
    K::eltype(p) = extract_amplitude(p,plid=plid)
    logp += logprior_period(P) + logprior_amplitude(K)
  end
  return logp::eltype(p)
end

function logprior_offset(C::Real)
  const max_C = 1000.0
  if !(-max_C<=C<=max_C) 
     return -Inf
  end
  return -log(2*max_C)
end

logprior_offset(p::Vector) = logprior_offset(extract_rvoffset(p))

function logprior(p::Vector) 
  logprior_noise(p) + logprior_offset(p) + logprior_planets(p) 
end

function logtarget(p::Vector)
  val = logprior(p)
  if val==-Inf
     return val
  end
  val += loglikelihood(p)
  return val
end

negative_logtarget(p::Vector) = -logtarget(p)



