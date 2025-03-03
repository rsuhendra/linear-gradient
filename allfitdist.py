import scipy.stats as stats
import numpy as np
import pandas as pd

# GET DISTRIBUTIONS

common_cont_dist_names = (
	'alpha',            # Alpha continuous random variable
	'beta',             # Beta continuous random variable
	'cauchy',           # Cauchy continuous random variable
	'chi2',             # Chi-squared continuous random variable
	'expon',            # Exponential continuous random variable
	'exponweib',        # Exponentiated Weibull continuous random variable
	'f',                # F continuous random variable
	'fatiguelife',      # Fatigue-life (Birnbaum-Sanders) continuous random variable
	'genextreme',       # Generalized extreme value continuous random variable
	'genpareto', 		# Generalized Pareto
	'gamma',            # Gamma continuous random variable
	'laplace',          # Laplace continuous random variable
	'lognorm',          # Lognormal continuous random variable
	'norm',             # Normal continuous random variable
	'powerlognorm',     # Power log-normal continuous random variable
	'powernorm',        # Power normal continuous random variable
	't',                # Student's T continuous random variable
	'tukeylambda',      # Tukey-Lamdba continuous random variable
	'uniform',          # Uniform continuous random variable
	'invgauss',			# Inverse Gaussian 
	'logistic',			# Logistic
	'fisk',				# Log logistic
	'rice',
	'rayleigh',			
	'weibull_min'
)

# Get all continuous distributions
all_continuous_distributions = [dist for dist in dir(stats) 
						if isinstance(getattr(stats, dist), stats.rv_continuous)]

# There were some that took a while to run
disclude = ['levy_stable', 'studentized_range']
subset_continuous_distributions = [d for d in all_continuous_distributions if d not in disclude]

# FUNCTIONS

def compute_ic(log_likelihood, num_params, num_data_points, mode = 'bayesian'):
	if mode == 'aikake':
		return 2* num_params * np.log(num_data_points) - 2 * log_likelihood
	elif mode == 'bayesian':
		return num_params * np.log(num_data_points) - 2 * log_likelihood

def allfitdist(data, dists_to_test, sortby = 'BIC'):
	df = []
	for dist_name in dists_to_test:
		try:
			# Get the distribution object
			dist = getattr(stats, dist_name)
			
			# Fit the data to the distribution
			params = dist.fit(data)
			
			# Create a frozen distribution with the fitted parameters
			fitted_dist = dist(*params)
			
			# Compute the log-likelihood
			log_likelihood = np.sum(fitted_dist.logpdf(data))
			
			# Compute BIC and AIC
			bic = compute_ic(log_likelihood, len(params), len(data), mode = 'bayesian')
			aic = compute_ic(log_likelihood, len(params), len(data), mode = 'aikake')

			# Kolmogorov-Smirnov GoF
			ks_stat, ks_pval = stats.kstest(data, dist_name, args=params)

			# Print the results
			# print(f"Distribution: {dist_name}")
			# print(f"Fitted Parameters: {params}")
			# print(f"BIC  {bic}, KS {ks_stat}, KS_PVAL {ks_p_value}")
			# print("-" * 40)
	
			df.append({'Distribution': dist_name, 'Params': params, 'BIC': bic, 'AIC': aic, 'KS_STAT': ks_stat, 'KS_PVAL': ks_pval})

		except Exception as e:
			print(f"Failed to fit {dist_name}: {e}")

	df = pd.DataFrame(df)
	df = df.sort_values(by=sortby)
	return df

# # generate Example data
# data = np.random.normal(loc=10, scale=2, size=1000)

# df = allfitdist(data, common_cont_dist_names, sortby = 'BIC')

# print(df)
