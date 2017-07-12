if !isdefined(:LombScargle) using LombScargle end 

function find_n_peaks_in_pgram(period::Vector, power::Vector; num_peaks::Integer = 1, exclude_period_factor::Real = 1.2 )
  @assert num_peaks>= 1
  peak_periods = zeros(num_peaks)
  peaks_found = 1
  peak_periods[peaks_found] = period[findmax(power)[2]]
  while peaks_found < num_peaks
     idx_active = trues(length(period))
 for j in 1:length(period)
 for i in 1:peaks_found
        if peak_periods[i]/exclude_period_factor <= period[j] <= peak_periods[i]*exclude_period_factor
   idx_active[j] = false
end # if near a previous peak
 end # for over peaks
 end # for over periods
 peaks_found += 1
 peak_periods[peaks_found] = period[idx_active][findmax(power[idx_active])[2]]
  end # while more peaks to be found
  peak_periods
end

function find_n_peaks(time::Vector, obs::Vector, sigma::Vector; num_peaks::Integer = 1, exclude_period_factor::Real = 1.2 )
  pgram = lombscargle(time, obs, sigma)
  find_n_peaks_in_pgram(period(pgram), power(pgram), num_peaks=num_peaks, exclude_period_factor=exclude_period_factor )
end



