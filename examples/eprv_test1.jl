ENV["PLOTS_DEFAULT_BACKEND"] = "Plotly"
data = readdlm("../data/eprv3_rvs_1.txt");

want_to_plot_pgram = false
if want_to_plot_pgram
  if !isdefined(:Plots) using Plots end 
  if !isdefined(:LombScargle) using LombScargle end
  plot(periodpower(pgram),xscale=:log)
end 

include("find_peaks.jl")
num_planets = 2
periods_to_try = find_n_peaks(data[:,1],data[:,2],data[:,3],num_peaks=num_planets)

try_period_near_rotation = false
if try_extra_period_near_rotation 
  rotation_period = 20.0
  push!(periods_to_try,rotation_period)
  num_planets += 1 
end

include("rv_model.jl")    # provides logtarget
import RvModelCircularIndepNoise
RvModelCircularIndepNoise.set_times(data[:,1]);
RvModelCircularIndepNoise.set_obs(data[:,2]);
RvModelCircularIndepNoise.set_sigma_obs(data[:,3]);
num_offsets = 1
param_init_circ_indep = zeros(num_planets*RvModelCircularIndepNoise.num_param_per_planet+num_offsets+RvModelCircularIndepNoise.num_noise_param())
for p in 1:num_planets
  amp = p==1 ? std(data[:,2]) : 0.
  RvModelCircularIndepNoise.set_PKM0(param_init_circ_indep,periods_to_try[p], amp, 1*pi, plid=p)
end
RvModelCircularIndepNoise.set_rvoffset(param_init_circ_indep,0.)
RvModelCircularIndepNoise.set_jitter(param_init_circ_indep, 1.0)

RvModelCircularIndepNoise.logtarget(param_init_circ_indep)

if !isdefined(:Optim) using Optim end

# First optimize just phase
planet_to_optimize = 1
function negative_logtarget_fixed_PK(p::Float64)
  RvModelCircularIndepNoise.set_mean_anomaly_at_t0(param_init_circ_indep,p,plid=planet_to_optimize)
  RvModelCircularIndepNoise.negative_logtarget(param_init_circ_indep)
end

result_circ_phase_indep = optimize(negative_logtarget_fixed_PK, 0, 2pi)
result_circ_phase_indep.minimizer
result_circ_phase_indep.minimum/size(data,1) 
RvModelCircularIndepNoise.set_mean_anomaly_at_t0(param_init_circ_indep,mod2pi(result_circ_phase_indep.minimizer[1]))

for planet_to_optimize in 2:num_planets
  amp = std(map(i->data[i,2]- RvModelCircularIndepNoise.calc_model_rv(param_init_circ_indep,data[i,1]), 1:size(data,1)))
  RvModelCircularIndepNoise.set_PKM0(param_init_circ_indep,periods_to_try[planet_to_optimize], amp, 1*pi, plid=planet_to_optimize)
  
  result_circ_phase_indep = optimize(negative_logtarget_fixed_PK, 0, 2pi)
  result_circ_phase_indep.minimizer
  result_circ_phase_indep.minimum/size(data,1) 
  RvModelCircularIndepNoise.set_mean_anomaly_at_t0(param_init_circ_indep,mod2pi(result_circ_phase_indep.minimizer[1]),plid=planet_to_optimize)
end

# Now setup model with GP noise
import RvModelCircularGPNoise
RvModelCircularGPNoise.set_times(data[:,1]);
RvModelCircularGPNoise.set_obs(data[:,2]);
RvModelCircularGPNoise.set_sigma_obs(data[:,3]);
num_offsets = 1
param_init_circ_gp = vcat(param_init_circ_indep[1:num_planets*RvModelCircularIndepNoise.num_param_per_planet+num_offsets],zeros(RvModelCircularGPNoise.num_noise_param()) ) 
RvModelCircularGPNoise.set_noise_param(param_init_circ_gp, [3.0, 50., 0.5, 20.])
RvModelCircularGPNoise.logtarget(param_init_circ_gp)

# Refitting with a GP model isn't really necessary, but I wrote it to make sure things were working
function negative_logtarget_fixed_PK(p::Float64)
  RvModelCircularGPNoise.set_mean_anomaly_at_t0(param_init_circ_gp,p)
  RvModelCircularGPNoise.negative_logtarget(param_init_circ_gp)
end

if !isdefined(:Optim) using Optim end
result_circ_phase_gp = optimize(negative_logtarget_fixed_PK, result_circ_phase_indep.minimizer-pi, result_circ_phase_indep.minimizer+pi)
result_circ_phase_gp.minimizer
result_circ_phase_gp.minimum/size(data,1) 
RvModelCircularGPNoise.set_mean_anomaly_at_t0(param_init_circ_gp,mod2pi(result_circ_phase_gp.minimizer[1]))

# This optimization is slow and still may not complete
# result_circ = optimize(RvModelCircularGPNoise.negative_logtarget, param_init_circ_gp, ConjugateGradient())

plot_model = false
if plot_model
  pred_obs = map(t->RvModelCircularGPNoise.calc_model_rv(param_init_circ_gp,t),data[:,1]);
  times_plot = linspace(minimum(data[:,1]),maximum(data[:,1]),800);
  pred_plot = map(t->RvModelCircularGPNoise.calc_model_rv(param_init_circ_gp,t),times_plot);

  if !isdefined(:Plots) using Plots end 
  plot(times_plot,pred_plot)
  scatter!(data[:,1],pred_obs)
end

plot_residuals_periodogram = false
if plot_residuals_periodogram
  residuals = map(i->data[i,2]- RvModelCircularGPNoise.calc_model_rv(param_init_circ_gp,data[i,1]), 1:size(data,1));
  if !isdefined(:LombScargle) using LombScargle end 
  resid_pgram = lombscargle(data[:,1],residuals,data[:,3])
  if !isdefined(:Plots) using Plots end 
  plot(periodpower(resid_pgram),xscale=:log)
end

include("demcmc.jl")

perturb_scale = fill(0.01,length(param_init_circ_gp))
for p in 1:num_planets 
  perturb_scale[1+RvModelCircularGPNoise.pl_offset(p)] = 1e-3
end
pop_init_circ = param_init_circ_gp .+ perturb_scale.*randn(length(param_init_circ_gp),3*length(param_init_circ_gp))
burnin_circ = run_demcmc( pop_init_circ, RvModelCircularGPNoise.logtarget, num_gen = 100, epsilon = 0.01 );

#result_circ = copy(burnin_circ)
result_circ = run_demcmc( burnin_circ["theta_last"], RvModelCircularGPNoise.logtarget, num_gen = 1000, epsilon = 0.01 );

for p in 1:num_planets
  tmp = RvModelCircularGPNoise.inv_transform_period.(result_circ["theta_all"][1+RvModelCircularGPNoise.pl_offset(p),:,:])
  println("param P_",p,": ",mean(tmp), " +/- ", std(tmp))
  tmp = RvModelCircularGPNoise.inv_transform_amplitude.(result_circ["theta_all"][2+RvModelCircularGPNoise.pl_offset(p),:,:])
  println("param K_",p,": ",mean(tmp), " +/- ", std(tmp))
  tmp = result_circ["theta_all"][3+RvModelCircularGPNoise.pl_offset(p),:,:]
  println("param M_",p,": ",mean(tmp), " +/- ", std(tmp))
end
for i in (num_planets*RvModelCircularGPNoise.num_param_per_planet+1):length(param_init_circ_gp)
  println("param #",i,": ",mean(result_circ["theta_all"][i,:,:]), " +/- ", std(result_circ["theta_all"][i,:,:]))
end


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
RvModelKeplerianGPNoise.set_rvoffset(param_init_kepler_gp, RvModelCircularGPNoise.extract_rvoffset(param_init_circ_gp) )
RvModelKeplerianGPNoise.set_noise_param(param_init_kepler_gp, [3.0, 50., 0.5, 20.])
RvModelKeplerianGPNoise.logtarget(param_init_kepler_gp)


perturb_scale = fill(0.01,length(param_init_kepler_gp))
for p in 1:num_planets 
  perturb_scale[1+RvModelCircularGPNoise.pl_offset(p)] = 1e-3
end
pop_init_kepler = param_init_kepler_gp .+ perturb_scale.*randn(length(param_init_kepler_gp),3*length(param_init_kepler_gp))
burnin_kepler = run_demcmc( pop_init_kepler, RvModelKeplerianGPNoise.logtarget, num_gen = 100, epsilon = 0.01 );

result_kepler = copy(burnin_kepler)
#result_kepler = run_demcmc( burnin_kepler["theta_last"], RvModelKeplerianGPNoise.logtarget, num_gen = 1000, epsilon = 0.01 );


for p in 1:num_planets
  tmp = RvModelKeplerianGPNoise.inv_transform_period.(result_kepler["theta_all"][1+RvModelKeplerianGPNoise.pl_offset(p),:,:])
  println("param P_",p,": ",mean(tmp), " +/- ", std(tmp))
  tmp = RvModelKeplerianGPNoise.inv_transform_amplitude.(result_kepler["theta_all"][2+RvModelKeplerianGPNoise.pl_offset(p),:,:])
  println("param K_",p,": ",mean(tmp), " +/- ", std(tmp))
  tmp = result_kepler["theta_all"][3+RvModelKeplerianGPNoise.pl_offset(p),:,:]
  println("param ecosw_",p,": ",mean(tmp), " +/- ", std(tmp))
  tmp = result_kepler["theta_all"][4+RvModelKeplerianGPNoise.pl_offset(p),:,:]
  println("param esinw_",p,": ",mean(tmp), " +/- ", std(tmp))
  tmp = result_kepler["theta_all"][5+RvModelKeplerianGPNoise.pl_offset(p),:,:]
  println("param M_",p,": ",mean(tmp), " +/- ", std(tmp))
end
for i in (num_planets*RvModelKeplerianGPNoise.num_param_per_planet+1):length(param_init_kepler_gp)
  println("param #",i,": ",mean(result_kepler["theta_all"][i,:,:]), " +/- ", std(result_kepler["theta_all"][i,:,:]))
end

