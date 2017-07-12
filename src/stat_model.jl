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

function logprior_planets(p::Vector) 
  if !is_valid_planets(p) return -Inf end  # prempt model evaluation
  num_pl = num_planets(p)
  @assert num_pl >= 1
  logp = zero(eltype(p))
  if num_param_per_planet == 5
     logp -= 2*num_pl*log(2pi)
  end
  const max_period = 10000.0
  const max_amplitude = 10000.0
  for plid in 1:num_pl
    P::eltype(p) = extract_period(p,plid=plid)
    K::eltype(p) = extract_amplitude(p,plid=plid)
    logp += -log((1+P/P0::Float64)*log1p(max_period/P0::Float64)* 
                 (1+K/K0::Float64)*log1p(max_amplitude/K0::Float64) )
  end
  return logp::eltype(p)
end

function logprior(p::Vector) 
  logprior_noise(p) + logprior_planets(p) 
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



