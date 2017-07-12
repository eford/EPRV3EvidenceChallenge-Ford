#=
   Provides calc_model_rv(theta, time) 
   Computes the velocity of the star due to the perturbations of multiple planets, as the linear superposition of the circular orbit induced by each planet, i.e., neglecting eccentricity and mutual planet-planet interactions
=#

# Calculate Keplerian velocity of star due to one planet (with parameters displaced by offset)
function calc_rv_circ_one_planet{T1,T2}( theta::Array{T1,1}, time::T2; plid::Integer = 1 )
  (P,K,M0) = extract_PKM0(theta,plid=plid)
  #(P,Kc,Ks) = extract_PKcKs(theta,plid=plid)
  n = 2pi/P
  #M = mod2pi(time*n-M0)
  M = time*n-M0
  return K*cos(M)
  #return Kc*cos(M)+Ks*sin(M)
end

# Calculate Keplerian velocity of star due to num_pl planets
function calc_rv_circ_multi_planet{T1,T2}( theta::Array{T1,1}, time::T2)
  zdot = zero(T1)
  for plid in 1:num_planets(theta)
      zdot += calc_rv_circ_one_planet(theta,time,plid=plid)
  end
  return zdot
end

# Assumes model parameters = [ Period_1, K_1, M0_1,   Period_2, K_2, M0_2, ...  C ] 
function calc_model_rv{T1,T2}( theta::Array{T1,1}, t::T2; obsid::Integer = 1, tol::Real = 1.e-8)
  Offset = extract_rvoffset(theta, obsid=obsid)
  calc_rv_circ_multi_planet(theta, t) + Offset
end

export calc_model_rv


