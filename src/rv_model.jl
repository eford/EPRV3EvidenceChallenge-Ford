#=
   Likelihhod is chi^2 comparing observations of the radial velocity of the star to a model, 
   where the model computes the velocity of the star as the linear superposition of the Keplerian orbit 
   induced by each planet, i.e., neglecting mutual planet-planet interactions 
   Priors based on SAMSI reference priors from Ford & Gregory 2006 
=#

include("utils_internal.jl")                 # Manually add code to support autodiff of mod2pi

module RvModelCircularIndepNoise

include("rv_model_circ_param.jl")       # Code to convert parameter vector to named parameters
include("rv_model_circ.jl")             # Code to compute RV model for circular orbits
include("stat_model.jl")                # Common code to compute target density, logtarget, logprior_planets
include("noise_model_indep.jl")      # Code to compute target density, loglikelihood, logprior_noise

end # module RvModelCircularIndepNoise

module RvModelCircularGPNoise

include("rv_model_circ_param.jl")       # Code to convert parameter vector to named parameters
include("rv_model_circ.jl")             # Code to compute RV model for circular orbits
include("stat_model.jl")                # Common code to compute target density, logtarget, logprior_planets
include("noise_model_gp.jl")             # Code to compute target density, loglikelihood, logprior_noise

end # module RvModelCircularGPNoise

module RvModelKeplerianIndepNoise

include("rv_model_keplerian_param.jl")  # Code to convert parameter vector to named parameters
include("rv_model_keplerian.jl")        # Code to compute RV model for Keplerian orbits
include("stat_model.jl")                # Common code to compute target density, logtarget, logprior_planets
include("noise_model_indep.jl")      # Code to compute target density, loglikelihood, logprior_noise

end # module RvModelKeplerianIndepNoise

module RvModelKeplerianGPNoise

include("rv_model_keplerian_param.jl")  # Code to convert parameter vector to named parameters
include("rv_model_keplerian.jl")        # Code to compute RV model for Keplerian orbits
include("stat_model.jl")                # Common code to compute target density, logtarget, logprior_planets
include("noise_model_gp.jl")             # Code to compute target density, loglikelihood, logprior_noise

end # module RvModelKeplerianGPNoise


