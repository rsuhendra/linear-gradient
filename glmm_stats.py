import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.othermod.betareg import BetaModel
from statsmodels.genmod import families
import os
import matplotlib.pyplot as plt

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri  # Import pandas2ri correctly
from pymer4.models import Lmer, Lm

# Function to look through all groups and find the relevant ones easily
def groupNameFinder(keyword):
	all_genotypes = os.listdir('data')    
	relevant = list(filter(lambda name: keyword.lower() in name.lower(), all_genotypes))
	# relevant_genotypes = [name for name in all_genotypes if keyword in name]
	if len(relevant) != 2:
		print('There was an error in finding groupNames!')
		return 0
	else:
		if '+' in relevant[0] and 'kir' in relevant[1].lower():
			return relevant
		elif '+' in relevant[1] and 'kir' in relevant[0].lower():
			return [relevant[1], relevant[0]]
		else:
			print('There was an error in finding groupNames!')
			return 0

# Binomial GLM for reaching/or not
def reachOrNot_glm(groupNames, mode = 'regular'):
	dfs = []
	fig, ax = plt.subplots()

	for k, groupName in enumerate(groupNames):
		df = pd.read_csv(f'data/{groupName}/reachOrNot_{groupName}.csv', index_col=None)
		q, r = divmod(k, 2)
		df['KIR'] = r
		df['UAS'] = q
			

		dfs.append(df)
		ax.bar(groupName, np.average(df.reachOrNot), color='blue')

	dfs = pd.concat(dfs, axis=0, ignore_index=True)
 
	# Save figure
	ax.set_ylim([0, 1])
	ax.set_ylabel('P(Reach End)')
	ax.set_xlabel('Genotype')
	fig.savefig('Test.png')
	plt.close('all')
 
	if mode == 'regular':
		# Fit logistic regression model with interaction term
		model = smf.glm("reachOrNot ~ KIR * UAS", data=dfs, family=sm.families.Binomial()).fit()

		# Model summary
		print(model.summary())

		# # Predict probabilities (the predicted P(reachOrNot = 1))
		# new_data = pd.DataFrame({'KIR': 0, 'UAS': 1}, index=[0])
		# predicted_probs = model.predict(new_data)
		# print(predicted_probs)
 
	elif mode == 'firth':
		
		# Import R packages
		pandas2ri.activate()
		base = importr('base')
		logistf = importr('logistf')
	
		# Convert the pandas DataFrame to an R data frame
		r_df = pandas2ri.py2rpy(dfs)

		# Fit the logistic regression model using logistf
		formula = robjects.Formula('reachOrNot ~ KIR * UAS')
		fit = logistf.logistf(formula, data=r_df)
		
		# This thing prints twice if I print
		# print(base.summary(fit))
		test = str(base.summary(fit))

def turns_glmm(groupNames, bins_redux = False):
	
	if bins_redux:
		l1, u1 = np.pi/3, 2*np.pi/3
		l2, u2 = 4*np.pi/3, 5*np.pi/3
	else:
		l1, u1 = np.pi/4, 3*np.pi/4
		l2, u2 = 5*np.pi/4, 7*np.pi/4

	dfs = []
	fig, ax = plt.subplots()
	for k, groupName in enumerate(groupNames):
		turn_df = pd.read_csv(f'data/{groupName}/turns_{groupName}.csv', index_col=None)

		# Fix angles to [0, 2pi], extract correct turns
		turn_df.angle1 = turn_df.angle1 % (2*np.pi)
		turn_df = turn_df[turn_df.all_turns!=0]
		turn_df['correct_turn'] = 1*(turn_df.all_turns > 0)

		df1 = turn_df[(turn_df.angle1 >= l1) & (turn_df.angle1 <= u1)]
		df2 = turn_df[(turn_df.angle1 >= l2) & (turn_df.angle1 <= u2)]
		df1.loc[:, 'correct_turn'] = 1 - df1['correct_turn']
		
		df = pd.concat([df1, df2], axis=0)

		q, r = divmod(k, 2)
		df['KIR'] = r
		df['UAS'] = q
  
		# Plotting bar of 
		ax.bar(groupName, np.average(df.correct_turn), color='green')

		print(groupName, np.average(df.correct_turn))
			
		dfs.append(df)

	dfs = pd.concat(dfs, axis=0, ignore_index=True)
 
	# Save figure
	ax.set_ylim([0.5, 1])
	ax.set_ylabel('P(Correct Turn)')
	ax.set_xlabel('Genotype')
	fig.savefig('Test2.png')
	plt.close('all')

	# Fit GLMM
	model = Lmer("correct_turn ~ KIR * UAS + (1 | all_fnames)", data=dfs, family='binomial')
	model.fit()

	print(model.summary())
 
	# new_data = pd.DataFrame({'KIR': 0, 'UAS': 1, 'all_fnames':'' }, index=[0])
	# predicted_probs = model.predict(new_data)
	# print(predicted_probs)
 


def effx_betareg(groupNames):
	dfs = []
	for k, groupName in enumerate(groupNames):
		df = pd.read_csv(f'data/{groupName}/effx_{groupName}.csv', index_col=None)
		q, r = divmod(k, 2)
		df['KIR'] = r
		df['UAS'] = q
			
		dfs.append(df)

	dfs = pd.concat(dfs, axis=0, ignore_index=True)
 
	# model = BetaModel.from_formula("effx ~ KIR * UAS", dfs, link_precision=families.links.identity()).fit()
	model = BetaModel.from_formula("effx ~ KIR * UAS", dfs).fit()

	print(model.summary())
 
 
#  Order should be ['WT', 'Kir/+', 'UAS/+', 'UAS/Kir']

groupNames = ['WT', 'Kir-+', 'HdC+', 'HdC_Kir']
groupNames = ['WT', 'Kir-+'] + groupNameFinder('hdc')

turns_glmm(groupNames, bins_redux=True)
reachOrNot_glm(groupNames, mode='firth')