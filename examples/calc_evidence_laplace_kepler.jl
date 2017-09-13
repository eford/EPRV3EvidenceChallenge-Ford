include("../src/rv_model_keplerian_simple.jl")

if !isdefined(:PDMats) using PDMats  end # Positive Definite Matrices
if !isdefined(:Optim)  using Optim   end # Optimizing over non-linear parameters
if !isdefined(:QuadGK) using QuadGK  end # 1-d integration
if !isdefined(:Cuba)   using Cuba    end # multi-d integration

# Read dataset
data = readdlm("../data/rvs_0001.txt");
times = data[:,1]
vels = data[:,2]
sigma_obs = data[:,3]

Sigma0 = PDMat(make_Sigma(times,sigma_obs,0.0))
SigmaBig = PDMat(make_Sigma(times,sigma_obs,std(vels)))
param0 = [0.]
res = laplace_approximation_kepler_model(param0,times,vels,SigmaBig,loglikelihood_fixed_jitter)
param0 = res[2].minimizer;
laplace_approximation_kepler_model_jitter_separate(param0,times,vels,Sigma0)[1]/log(10)

param1a = [42.0,5.0,0.01,0.01,1.,0.]
res = laplace_approximation_kepler_model(param1a,times,vels,SigmaBig,loglikelihood_fixed_jitter)
param1a = res[2].minimizer;
laplace_approximation_kepler_model_jitter_separate(param1a,times,vels,Sigma0)[1]/log(10)

param1b = [12.1,3.0,0.01,0.01,1.,0.]
res = laplace_approximation_kepler_model(param1b,times,vels,SigmaBig,loglikelihood_fixed_jitter)
param1b = res[2].minimizer;
laplace_approximation_kepler_model_jitter_separate(param1b,times,vels,Sigma0)[1]/log(10)

#param2 = [42.0,5.0,0.01,0.01,1.,12.1,3.0,0.01,0.01,1.,0.]
param2 = vcat(param1a[1:5], param1b[1:5], 0.0)
res = laplace_approximation_kepler_model(param2,times,vels,SigmaBig,loglikelihood_fixed_jitter)
param2 = res[2].minimizer;
laplace_approximation_kepler_model_jitter_separate(param2,times,vels,Sigma0)[1]/log(10)
