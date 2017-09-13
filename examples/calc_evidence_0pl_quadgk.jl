ENV["PLOTS_DEFAULT_BACKEND"] = "Plotly"
data = readdlm("../data/rvs_0001.txt");

want_to_plot_pgram = false
if want_to_plot_pgram
  if !isdefined(:Plots) using Plots end 
  if !isdefined(:LombScargle) using LombScargle end
end 

num_planets = 0
include("../src/rv_model.jl")    # provides logtarget

# Now with Keplerian model & GP noise
import RvModelKeplerianGPNoise
RvModelKeplerianGPNoise.set_times(data[:,1]);
RvModelKeplerianGPNoise.set_obs(data[:,2]);
RvModelKeplerianGPNoise.set_sigma_obs(data[:,3]);
num_offsets = 1
param_init_kepler_gp = zeros(num_planets*RvModelKeplerianGPNoise.num_param_per_planet+num_offsets+RvModelKeplerianGPNoise.num_noise_param()) 
for p in 1:num_planets
  (P,K,M0) = RvModelCircularGPNoise.extract_PKM0(param_init_circ_gp, plid=p)
  e = 0.01
  w = 0.0
  RvModelKeplerianGPNoise.set_PKewM0(param_init_kepler_gp, P, K, e, w, M0; plid=p)  
end
RvModelKeplerianGPNoise.set_rvoffset(param_init_kepler_gp, 0.0)
RvModelKeplerianGPNoise.set_noise_param(param_init_kepler_gp, [0.0] )

if !isdefined(:QuadGK) using QuadGK end
logtarget_ref = RvModelKeplerianGPNoise.logtarget(param_init_kepler_gp)

param_kepler_gp = param_init_kepler_gp
function integrand_rvoffset(x::Real)
  RvModelKeplerianGPNoise.set_rvoffset(param_kepler_gp, x)
  exp(RvModelKeplerianGPNoise.logtarget(param_kepler_gp)-logtarget_ref)
end
function integrand_jitter(x::Real)
  RvModelKeplerianGPNoise.set_noise_param(param_kepler_gp, [x])
  exp(RvModelKeplerianGPNoise.logtarget(param_kepler_gp)-logtarget_ref)
 end

function marginalized_over_rvoffset(x::Real)
  RvModelKeplerianGPNoise.set_noise_param(param_kepler_gp, [x])
  f(y) = integrand_rvoffset(y)
  QuadGK.quadgk(f,-1000.0,1000.0)
end

g(z) = marginalized_over_rvoffset(z::Real)[1]

logevidence = log(QuadGK.quadgk(g,0.0,99.0)[1])
result = (logtarget_ref+logevidence)/log(10.0)

