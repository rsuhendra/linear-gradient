import scipy.stats as stats
import numpy as np
import pandas as pd

# ===== GET DISTRIBUTIONS =====
# Tuple of commonly used continuous probability distributions to fit data against

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

# Exclude distributions that take too long to fit/compute
disclude = ['levy_stable', 'studentized_range']
subset_continuous_distributions = [d for d in all_continuous_distributions if d not in disclude]

# ===== FUNCTIONS =====

# Computes information criteria (AIC or BIC) to compare goodness of fit across distributions
# Parameters:
#   log_likelihood: sum of log probabilities from fitted distribution
#   num_params: number of parameters in the fitted distribution
#   num_data_points: total number of data points
#   mode: 'bayesian' for BIC (default) or 'aikake' for AIC
# Returns: information criterion value (lower is better fit)
def compute_ic(log_likelihood, num_params, num_data_points, mode = 'bayesian'):
	# AIC (Akaike Information Criterion) formula
	if mode == 'aikake':
		return 2* num_params * np.log(num_data_points) - 2 * log_likelihood
	# BIC (Bayesian Information Criterion) formula
	elif mode == 'bayesian':
		return num_params * np.log(num_data_points) - 2 * log_likelihood

# Fits multiple probability distributions to data and compares their goodness of fit
# Parameters:
#   data: array of data points to fit
#   dists_to_test: list of distribution names (strings) to fit
#   sortby: metric to sort results by (default 'BIC')
# Returns: DataFrame with results including fitted parameters, BIC, AIC, and KS test statistics
def allfitdist(data, dists_to_test, sortby = 'BIC'):
	df = []
	# Iterate through each distribution to fit
	for dist_name in dists_to_test:
		try:
			# Retrieve the scipy distribution object by name
			dist = getattr(stats, dist_name)
			
			# Fit the distribution parameters to the data using maximum likelihood estimation
			params = dist.fit(data)
			
			# Create a frozen distribution object with the fitted parameters
			fitted_dist = dist(*params)
			
			# Calculate the sum of log probabilities (log-likelihood) for the fitted distribution
			log_likelihood = np.sum(fitted_dist.logpdf(data))
			
			# Compute information criteria for model comparison
			bic = compute_ic(log_likelihood, len(params), len(data), mode = 'bayesian')
			aic = compute_ic(log_likelihood, len(params), len(data), mode = 'aikake')

			# Perform Kolmogorov-Smirnov goodness of fit test
			ks_stat, ks_pval = stats.kstest(data, dist_name, args=params)

			# Print statements for debugging (commented out)
			# print(f"Distribution: {dist_name}")
			# print(f"Fitted Parameters: {params}")
			# print(f"BIC  {bic}, KS {ks_stat}, KS_PVAL {ks_p_value}")
			# print("-" * 40)
	
			# Add results to list as dictionary
			df.append({'Distribution': dist_name, 'Params': params, 'BIC': bic, 'AIC': aic, 'KS_STAT': ks_stat, 'KS_PVAL': ks_pval})

		# Handle exceptions for distributions that fail to fit
		except Exception as e:
			print(f"Failed to fit {dist_name}: {e}")

	# Convert results list to DataFrame and sort by specified metric
	df = pd.DataFrame(df)
	df = df.sort_values(by=sortby)
	return df

# Example usage (commented out):
# Generate sample data from a normal distribution
# data = np.random.normal(loc=10, scale=2, size=1000)

# Fit all distributions and sort results by BIC
# df = allfitdist(data, common_cont_dist_names, sortby = 'BIC')

# Display the results
# print(df)
