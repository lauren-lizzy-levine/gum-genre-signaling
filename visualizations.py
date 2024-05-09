import pandas as pd
import matplotlib.pyplot as plt


def signal_proportion_by_genre(datafile, relation_list, relation_group_name, outfile, coarse=False):
	df = pd.read_csv(datafile, sep="\t")
	# Filter to have only genres in genre_list
	if coarse:
		df = df[df['COARSE_RELATION'].isin(relation_list)]
	else:
		df = df[df['RELATION'].isin(relation_list)]
	# Change oph label to dm
	df['SIGNAL_TYPE'].replace('orp', 'dm', inplace=True)
	# Get value counts
	value_counts = df['GENRE'].value_counts()
	group_names = value_counts.index.tolist()
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
	#print(pivot_df)
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


def fine_grained_signal_proportion_genre_graphs():
	datafile = "GUM_signals.tsv"
	signal_proportion_by_genre(datafile, ["organization-heading"], "Organization-heading", "organization-heading_genre_signal.png")
	signal_proportion_by_genre(datafile, ["organization-phatic"], "Organization-phatic", "organization-phatic_genre_signal.png")
	signal_proportion_by_genre(datafile, ["organization-preparation"], "Organization-preparation", "organization-preparation_genre_signal.png")
	signal_proportion_by_genre(datafile, ["explanation-evidence"], "Explanation-evidence", "explanation-evidence_genre_signal.png")
	signal_proportion_by_genre(datafile, ["explanation-justify"], "Explanation-justify", "explanation-justify_genre_signal.png")
	signal_proportion_by_genre(datafile, ["explanation-motivation"], "Explanation-motivation", "explanation-motivation_genre_signal.png")
	signal_proportion_by_genre(datafile, ["causal-cause"], "Causal-cause", "causal-cause_genre_signal.png")
	signal_proportion_by_genre(datafile, ["causal-result"], "Causal-result", "causal-result_genre_signal.png")
	return


def create_signal_proportion_genre_graphs():
	datafile = "GUM_signals.tsv"
	signal_proportion_by_genre(datafile, ["adversative"], "Adversative", "adversative_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["attribution"], "Attribution", "attribution_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["causal"], "Causal", "causal_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["context"], "Context", "context_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["contingency"], "Contingency", "contingency_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["elaboration"], "Elaboration", "elaboration_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["evaluation"], "Evaluation", "evaluation_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["explanation"], "Explanation", "explanation_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["joint"], "Joint", "jont_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["mode"], "Mode", "mode_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["organization"], "Organization", "organization_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["purpose"], "Purpose", "purpose_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["restatement"], "Restatement", "restatement_genre_signal.png", coarse=True)
	signal_proportion_by_genre(datafile, ["topic"], "Topic", "topic_genre_signal.png", coarse=True)
	return


def signal_proportion_by_relation(datafile, genre_list, genre_group_name, outfile):
	df = pd.read_csv(datafile, sep="\t")
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
	#print(pivot_df)
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

def create_signal_proportion_graphs():
	datafile = "GUM_signals.tsv"
	signal_proportion_by_relation(datafile, ["academic"], "Academic Writing", "academic_relation_signal.png")
	signal_proportion_by_relation(datafile, ["bio"], "Biographies", "bio_relation_signal.png")
	signal_proportion_by_relation(datafile, ["court"], "Courtroom Transcripts", "court_relation_signal.png")
	signal_proportion_by_relation(datafile, ["essay"], "Essays", "essay_relation_signal.png")
	signal_proportion_by_relation(datafile, ["fiction"], "Fiction", "fiction_relation_signal.png")
	signal_proportion_by_relation(datafile, ["whow"], "How-to Guides", "whow_relation_signal.png")
	signal_proportion_by_relation(datafile, ["interview"], "Interviews", "interview_relation_signal.png")
	signal_proportion_by_relation(datafile, ["letter"], "Letters", "letter_relation_signal.png")
	signal_proportion_by_relation(datafile, ["news"], "News", "news_relation_signal.png")
	signal_proportion_by_relation(datafile, ["reddit"], "Reddit", "reddit_relation_signal.png")
	signal_proportion_by_relation(datafile, ["podcast"], "Podcasts", "podcast_relation_signal.png")
	signal_proportion_by_relation(datafile, ["speech"], "Political Speeches", "speech_relation_signal.png")
	signal_proportion_by_relation(datafile, ["conversation"], "Conversations", "conversation_relation_signal.png")
	signal_proportion_by_relation(datafile, ["textbook"], "Textbooks", "textbook_relation_signal.png")
	signal_proportion_by_relation(datafile, ["voyage"], "Travel Guides", "voyage_relation_signal.png")
	signal_proportion_by_relation(datafile, ["vlog"], "Vlogs", "vlog_relation_signal.png")
	written_genres = ["academic", "bio", "essay", "fiction", "whow", "letter", "news", "reddit", "textbook", "voyage"]
	spoken_genres = ["court", "interview", "podcast", "speech", "conversation", "vlog"]
	signal_proportion_by_relation(datafile, written_genres, "Written", "written_relation_signal.png")
	signal_proportion_by_relation(datafile, spoken_genres, "Spoken", "spoken_relation_signal.png")
	signal_proportion_by_relation(datafile, written_genres + spoken_genres, "All Data", "everything_relation_signal.png")
	return

if __name__ == "__main__":
	#create_signal_proportion_graphs()
	#create_signal_proportion_genre_graphs()
	fine_grained_signal_proportion_genre_graphs()