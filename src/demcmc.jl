# Return log density for population of states
function eval_population( theta::Array{Float64,2}, calc_log_pdf::Function )
   popsize = size(theta,2)
   logpdf = Array{Float64}( popsize )
   for i in 1:popsize
      logpdf[i] = calc_log_pdf(theta[:,i])
   end
   return logpdf
end

## Differential Evolution Sampler 
#    based on ter Braak (2006) and Nelson et al. (2014)

# Generate ensemble of trial states according to DEMCMC 
function generate_trial_states_demcmc( theta::Array{Float64,2}; gamma_o::Float64 = 2.38/sqrt(2*size(theta,1)), epsilon = 0.01  )
   num_param = size(theta,1)
   num_pop = size(theta,2)
   theta_trial = similar(theta)
   
   for i in 1:num_pop
      # Choose head (k) and tail (j) for perturbation vector
  j = rand(1:(num_pop-1))
  if( j >= i ) j = j+1 end
  k = rand(1:(num_pop-2))
  if( k >= i ) k = k+1 end
  if( k >= j ) k = k+1 end
  
  # Choose scale factor
  scale = (1.0 + epsilon*randn()) * gamma_o
  
  theta_trial[:,i] = theta[:,i] + scale * (theta[:,k] - theta[:,j])
   end
   return theta_trial
end

# Evolve population of states (theta_init) with target density
function run_demcmc( theta_init::Array{Float64,2}, calc_log_pdf::Function; num_gen::Integer = 100, epsilon::Float64 = 0.01 )
   @assert(num_gen >=1 )
   num_param = size(theta_init,1)
   num_pop = size(theta_init,2)
   @assert(num_pop > num_param)
    
   # Allocate arrays before starting loop
   pop = Array{Float64}(num_param, num_pop, num_gen)
   poplogpdf = Array{Float64}(num_pop, num_gen )
   accepts_chain = zeros(Int64,num_pop )
   rejects_chain = zeros(Int64,num_pop )
   accepts_generation = zeros(Int64,num_gen )
   rejects_generation = zeros(Int64,num_gen )
    
   # Make a working copy of the current population of parameter values
   theta = copy(theta_init)
   logpdf = eval_population( theta, calc_log_pdf )
    
   for g in 1:num_gen # first(pop.range):last(pop.range)  # loop over generations
      # Generate population of trial sets of model parameters
      gamma_o = (mod(g,10)!=0) ? 2.38/sqrt(2*num_param) : 1.0 # every 10th generation try full-size steps
      theta_trial = generate_trial_states_demcmc( theta, gamma_o=gamma_o, epsilon=epsilon )
        
      # Evaluate model for each set of trial parameters
      logpdf_trial = eval_population( theta_trial, calc_log_pdf )
      
      # For each member of population  
      for i in 1:num_pop
        log_pdf_ratio = logpdf_trial[i] - logpdf[i]     
        if( (log_pdf_ratio>0) || (log_pdf_ratio>log(rand())) )     # Decide whether to Accept
            theta[:,i] = theta_trial[:,i]
            logpdf[i] = logpdf_trial[i]
            accepts_chain[i] = accepts_chain[i]+1
            accepts_generation[g] = accepts_generation[g]+1
        else
            rejects_chain[i] = rejects_chain[i]+1
            rejects_generation[g] = rejects_generation[g]+1
        end
      end
  
      # Log results
      if(true)  # (in(g,pop.range))
        pop[:,:,g] = copy(theta)
        poplogpdf[:,g] = copy(logpdf)
      end 
   end
   return Dict("theta_last"=>theta, "logpdf_last"=>logpdf, "theta_all"=>pop, "logpdf_all"=>poplogpdf, 
           "accepts_chain"=>accepts_chain, "rejects_chain"=>rejects_chain, "accepts_generation"=>accepts_generation, "rejects_generation"=>rejects_generation )
end


