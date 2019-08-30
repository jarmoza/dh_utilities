#!/usr/bin/env python

# Author: Jonathan Armoza
# Project: Digital Humanities Utilities (github.com/jarmoza/dh_utilities)
# Date: August 29, 2019

# Count holdouts
from collections import Counter

# Classifier for differentiating between item types
from sklearn.linear_model import LogisticRegression

# Splitting up training and test data sets
from sklearn.model_selection import train_test_split

class LogReg(object):

	def __init__(self,
		p_training_and_validation_set, p_training_and_validation_labels,
		p_holdout_set=None, p_holdout_labels=None, p_holdout_titles=None,
		p_test_size=0.25, p_random_state=0,
		p_solver="lbfgs", p_multi_class="auto", p_max_iter=500,
		p_verbose=False):

		# 1. Save parameters for modeling
		self.m_training_and_validation_data = p_training_and_validation_set
		self.m_training_and_validation_labels = p_training_and_validation_labels
		self.m_parameters = {

			"train_test_split": {

				"test_size": p_test_size,
				"random_state": p_random_state
			},
			
			"fit": {

				"solver": p_solver,
				"multi_class": p_multi_class,
				"max_iter": p_max_iter
			}
		}

		# 2. Save given verbosity
		self.m_verbose = p_verbose

		# 3. Calculate holdout label counts (for holdout prediction scoring), if given
		self.m_holdout_data = p_holdout_set
		self.m_holdout_labels = p_holdout_labels
		self.m_holdout_titles = p_holdout_titles
		self.m_holdout_counts = None
		if None != p_holdout_labels:
			self.m_holdout_counts = Counter(self.m_holdout_labels)

		# 4. Create the scikit logreg object with given parameters
		self.m_logistic_regression = LogisticRegression(solver=self.m_parameters["fit"]["solver"],
			multi_class=self.m_parameters["fit"]["multi_class"], max_iter=self.m_parameters["fit"]["max_iter"])

	def fit(self, p_verbose=None):

		verbose = LogReg.verbose_status(p_verbose, self.m_verbose)
		if verbose:
			print("Training model...")

		# 1. Fit training data given the labels
		self.m_logistic_regression.fit(self.m_training_data, self.m_training_labels)

	def holdout_score(self, p_verbose=None, p_print_predictions=False):

		verbose = LogReg.verbose_status(p_verbose, self.m_verbose)
		if verbose:
			print("Making predictions of holdout data set...")

		# 1. Make holdout predictions
		predicted_labels = []
		for index in range(len(self.m_holdout_data)):
			
			# A. Make a prediction for this item
			predicted_label = self.m_logistic_regression.predict([self.m_holdout_data[index]])
			predicted_labels.append(predicted_label)

			# B. Output labels for respective titles if no holdout labels given
			if verbose and None != self.m_holdout_titles and p_print_predictions:
				print("{0} - Prediction: {1}".format(self.m_holdout_titles[index], predicted_label))

		# 2. Score predictions on holdout set, if holdout labels were given
		self.m_holdout_prediction_scores = { item_type: 0 for item_type in self.m_holdout_counts }

		for index in range(len(self.m_holdout_data)):
			
			# A. Make prediction about this item
			predicted_label = self.m_logistic_regression.predict([self.m_holdout_data[index]])
			holdout_label = self.m_holdout_labels[index]
			
			# B. Output info about this prediction, if verbose requested
			if verbose and None != self.m_holdout_labels and p_print_predictions:
				if None != self.m_holdout_titles:
					print("{0} - Label: {1} Prediction: {2}".format(self.m_holdout_titles[index],
																	holdout_label, predicted_label))
				else:
					print("{0} - Label: {1} Prediction: {2}".format(index, holdout_label, predicted_label))

			# C. Accrue totals for each holdout type correctly predicted
			if predicted_label == holdout_label:
				self.m_holdout_prediction_scores[holdout_label] += 1

		# 3. Calculate prediction accuracy scores for each holdout item
		if verbose:
			print("Holdout prediction accuracy:")
		for item_type in self.m_holdout_prediction_scores:
			self.m_holdout_prediction_scores[item_type] /= float(self.m_holdout_counts[item_type])

			if verbose:
				print("\t{0}: {1}%%".format(item_type,
					self.m_holdout_prediction_scores[item_type] * 100))

		return self.m_holdout_prediction_scores

	def split_training_and_validation(self, p_verbose=None):

		verbose = LogReg.verbose_status(p_verbose, self.m_verbose)
		if verbose:
			print("Splitting up training and validation data sets...")

		# 1. Split the given data set into training and test sets and labels
		self.m_training_data, self.m_validation_data, self.m_training_labels, self.m_validation_labels = \
			train_test_split(self.m_training_and_validation_data, self.m_training_and_validation_labels,
							 test_size=self.m_parameters["train_test_split"]["test_size"],
							 random_state=self.m_parameters["train_test_split"]["random_state"])

		return self.m_training_data, self.m_training_labels, self.m_validation_data, self.m_validation_labels

	def validation_score(self, p_verbose=None):

		# 1. Determine model validation score
		self.m_validation_score = self.m_logistic_regression.score(self.m_validation_data,
																   self.m_validation_labels)
		
		verbose = LogReg.verbose_status(p_verbose, self.m_verbose)
		if verbose:
			print("Validation score: {0}".format(self.m_validation_score))

		return self.m_validation_score


	@staticmethod
	def split_data_and_holdouts_half(p_dataset):
		return p_dataset[0:int(len(p_dataset) / 2.0)], p_dataset[int(len(p_dataset) / 2.0) + 1:]

	@staticmethod
	def verbose_status(p_given_verbosity, p_instance_verbosity):
		return p_given_verbosity if None != p_given_verbosity else p_instance_verbosity



