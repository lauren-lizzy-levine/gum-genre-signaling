import pandas as pd
import matplotlib.pyplot as plt


def signal_proportion_by_genre(datafile, relation_group_name, outfile, relation_list=None, coarse=False, normalize=False):
	df = pd.read_csv(datafile, sep="\t")
	# Change oph label to dm
	df['SIGNAL_TYPE'].replace('orp', 'dm', inplace=True)
	# Filter to have only relation in relation_list
	if relation_list is not None:
		if coarse:
			df = df[df['COARSE_RELATION'].isin(relation_list)]
		else:
			df = df[df['RELATION'].isin(relation_list)]
	# Get value counts
	value_counts = df['GENRE'].value_counts()
	group_names = value_counts.index.tolist()
	if normalize:
		fine_relations = df['RELATION'].unique()
		column_names = list(df.columns)
		new_df = pd.DataFrame(columns=column_names + ["Value", "Total", "Proportion"])
		for relation in fine_relations:
			# make sub data frame
			temp_df = df[df["RELATION"] == relation].copy()
			# make proportions
			# Add constant value for each row to count the occurrences
			temp_df['Value'] = 1
			# Calculate proportions
			temp_df['Total'] = temp_df.groupby('GENRE')['Value'].transform('sum')
			temp_df['Proportion'] = (temp_df['Value'] / temp_df['Total']) #/ temp_df['Relation_Count']
			# concat onto data frame
			new_df = pd.concat([new_df, temp_df], ignore_index=True)
		new_df['Relation_Count'] = new_df.groupby('GENRE')['RELATION'].transform('nunique')
		new_df['Proportion'] = new_df['Proportion'] / new_df['Relation_Count']
		df = new_df
	else:
		# Add constant value for each row to count the occurrences
		df['Value'] = 1
		# Calculate proportions
		df['Total'] = df.groupby('GENRE')['Value'].transform('sum')
		df['Proportion'] = df['Value'] / df['Total']
	# Pivot the DataFrame for plotting
	pivot_df = df.pivot_table(index='GENRE', columns='SIGNAL_TYPE', values='Proportion', aggfunc='sum')
	# Update pivot rows (genres) with occurrence counts
	new_row_names = {}
	for genre in group_names:
		new_row_names[genre] = genre + ' (' + str(value_counts[genre]) + ')'
	pivot_df.rename(index=new_row_names, inplace=True)
	# Plot
	pivot_df.plot(kind='bar', stacked=True)
	plt.xlabel('Genre')
	plt.ylabel('Proportion')
	plt.title('Proportion of Signal Type per GUM Genre for ' + relation_group_name + ' Relations')
	plt.legend(title='Signal Type', bbox_to_anchor=(1, 1))
	plt.xticks(rotation=60)
	plt.savefig('visualizations/' + outfile, bbox_inches='tight')  # Save the plot as a PNG file with tight bounding box
	#plt.show()

	return


def signal_proportion_by_relation(datafile, genre_group_name, outfile, genre_list=None):
	df = pd.read_csv(datafile, sep="\t")
	if genre_list is not None:
		# Filter to have only genres in genre_list
		df = df[df['GENRE'].isin(genre_list)]
	# Change oph label to dm
	df['SIGNAL_TYPE'].replace('orp', 'dm', inplace=True)
	# Get value counts
	value_counts = df['COARSE_RELATION'].value_counts()
	group_names = value_counts.index.tolist()
	# Add constant value for each row to count the occurrences
	df['Value'] = 1
	# Calculate proportions
	df['Total'] = df.groupby('COARSE_RELATION')['Value'].transform('sum')
	df['Proportion'] = df['Value'] / df['Total']
	# Pivot the DataFrame for plotting
	pivot_df = df.pivot_table(index='COARSE_RELATION', columns='SIGNAL_TYPE', values='Proportion', aggfunc='sum')
	# Update pivot rows (coarse_relation) with occurrence counts
	new_row_names = {}
	for relation in group_names:
		new_row_names[relation] = relation + ' (' + str(value_counts[relation]) + ')'
	pivot_df.rename(index=new_row_names, inplace=True)
	# Plot
	pivot_df.plot(kind='bar', stacked=True)
	plt.xlabel('Coarse Relation')
	plt.ylabel('Proportion')
	plt.title('Proportion of Signal Type per Coarse Relation for GUM ' + genre_group_name)
	plt.legend(title='Signal Type', bbox_to_anchor=(1, 1))
	plt.xticks(rotation=60)
	plt.savefig('visualizations/' + outfile, bbox_inches='tight')  # Save the plot as a PNG file with tight bounding box
	#plt.show()

	return


def create_signal_proportion_genre_graphs():
	datafile = "GUM_signals.tsv"
	signal_proportion_by_genre(datafile, "All", "all_relations_genre_signal.png",
							   coarse=True, normalize=False)
	signal_proportion_by_genre(datafile, "All", "all_relations_genre_signal_normal.png",
							   coarse=True, normalize=True)
	signal_proportion_by_genre(datafile, "Explanation-Evidence", "explanation-evidence_relations_genre_signal.png",
							   relation_list=["explanation-evidence"], coarse=False, normalize=False)
	signal_proportion_by_genre(datafile, "Adversative-Antithesis", "adversative-antithesis_relations_genre_signal.png",
							   relation_list=["adversative-antithesis"], coarse=False, normalize=False)
	signal_proportion_by_genre(datafile, "Evaluation-Comment", "evaluation-comment_relations_genre_signal.png",
							   relation_list=["evaluation-comment"], coarse=False, normalize=False)
	signal_proportion_by_genre(datafile, "Organization", "organization_relations_genre_signal_normal.png",
							   relation_list=["organization"], coarse=True, normalize=True)
	signal_proportion_by_genre(datafile, "Causal", "causal_relations_genre_signal_normal.png",
							   relation_list=["causal"], coarse=True, normalize=True)
	signal_proportion_by_genre(datafile, "Evaluation", "evaluation_relations_genre_signal_normal.png",
							   relation_list=["evaluation"], coarse=True, normalize=True)
	signal_proportion_by_genre(datafile, "Restatement", "restatement_relations_genre_signal_normal.png",
							   relation_list=["restatement"], coarse=True, normalize=True)
	signal_proportion_by_genre(datafile, "Explanation", "explanation_relations_genre_signal_normal.png",
							   relation_list=["explanation"], coarse=True, normalize=True)

	return


def create_signal_proportion_relation_graphs():
	datafile = "GUM_signals.tsv"
	signal_proportion_by_relation(datafile, "All Genres", "all_genres_relation_signal.png", genre_list=None)
	return


if __name__ == "__main__":
	create_signal_proportion_genre_graphs()
	create_signal_proportion_relation_graphs()
