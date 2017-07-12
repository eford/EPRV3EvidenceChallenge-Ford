# Code to extract parameters from unlabeled parameter vector

# model assumes model parameters = [ Period, K, M0 ] for each planet followed by each RV offset and noise parameters
const num_param_per_planet = 3
pl_offset(plid::Integer) = (plid-1)*num_param_per_planet
num_planets(theta::Vector) = floor(Int64,(length(theta)-num_noise_param())//num_param_per_planet)  # WARNING: Assumes num_rvoffsets<num_param_per_planet
num_obs_offsets(theta::Vector) = length(theta)-num_planets(theta)*num_param_per_planet-num_noise_param()
obs_offset(theta::Vector,obsid::Integer = 1) = num_planets(theta)*num_param_per_planet+obsid
noise_param_offset(theta::Vector) = length(theta)-(num_noise_param()-1)

# constants defining unit system/scale for modified Jeffrys priors
const P0 = 1.0  # units of days
const K0 = 1.0  # units of m/s

transform_period(P) = log1p(P/P0)
transform_amplitude(K) = log1p(K/K0)
inv_transform_period(x) = P0*(exp(x)-1)
inv_transform_amplitude(x) = K0*(exp(x)-1)

set_period(theta::Vector, P; plid::Integer = 1) = theta[1+pl_offset(plid)] = transform_period(P)
set_amplitude(theta::Vector, K; plid::Integer = 1) = theta[2+pl_offset(plid)] = transform_amplitude(K)
set_mean_anomaly_at_t0(theta::Vector, M0; plid::Integer = 1) = theta[3+pl_offset(plid)] = M0
set_rvoffset(theta::Vector, C; obsid::Integer = 1) = theta[obs_offset(theta,obsid)] = C
function set_noise_param(theta::Vector, noise_param::Vector; obsid::Integer = 1) 
  @assert length(noise_param) == num_noise_param()
  theta[noise_param_offset(theta):end] = noise_param
  return
end
function set_PKM0(theta::Vector, P, K, M0; plid::Integer = 1) 
  set_period(theta,P,plid=plid)
  set_amplitude(theta,K,plid=plid) 
  set_mean_anomaly_at_t0(theta,M0,plid=plid)
  return
end


extract_period(theta::Vector; plid::Integer = 1) = inv_transform_period(theta[1+pl_offset(plid)])
extract_amplitude(theta::Vector; plid::Integer = 1) = inv_transform_amplitude(theta[2+pl_offset(plid)])
extract_mean_anomaly_at_t0(theta::Vector; plid::Integer = 1) = theta[3+pl_offset(plid)]
extract_rvoffset(theta::Vector; obsid::Integer = 1) = theta[obs_offset(theta,obsid)]
extract_noise_param(theta::Vector; obsid::Integer = 1) = theta[noise_param_offset(theta):end]

function extract_PKM0(theta::Vector; plid::Integer = 1) 
  local P,K,M0
  P = extract_period(theta, plid=plid)
  K = extract_amplitude(theta, plid=plid)
  M0 = extract_mean_anomaly_at_t0(theta, plid=plid)
  return (P,K,M0)
end

# check if parameters are valid
function is_valid_planets(p::Vector)
  local num_pl
  num_pl = num_planets(p)
  @assert length(num_pl) >= 1
  return true
end

export num_planets, num_obs_offsets
export set_period, set_amplitude, set_mean_anomaly_at_t0, set_rvoffset, set_noise_param
export extract_period, extract_amplitude, extract_mean_anomaly_at_t0, extract_PKM, extract_rvoffset, extract_noise_param
export transform_period, inv_transform_period, transform_amplitude, inv_transform_amplitude
export is_valid_planets

