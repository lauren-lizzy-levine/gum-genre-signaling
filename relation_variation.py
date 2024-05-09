from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


def frequency_counts_rel_genre(infile, coarse=False):
	lines = []
	with open(infile, "r") as file:
		reader = csv.reader(file, delimiter='\t')
		next(reader, None)  # skip the headers
		for line in reader:
			lines.append(line)
	freq_counts = {}
	for line in lines:
		genre = line[1]
		if coarse:
			relation = line[2] # coarse relation
		else:
			relation = line[3]
		signal_type = line[4]
		if signal_type == 'orp':
			signal_type = 'dm'
		if relation in freq_counts:
			if genre in freq_counts[relation]:
				if signal_type in freq_counts[relation][genre]:
					freq_counts[relation][genre][signal_type] += 1
				else:
					freq_counts[relation][genre][signal_type] = 1
			else:
				freq_counts[relation][genre] = {signal_type: 1}
		else:
			freq_counts[relation] = {genre: {signal_type: 1}}

	# fill in zeros
	singal_types = ['dm', 'grf', 'lex', 'mrf', 'num', 'ref', 'sem', 'syn']
	for relation in freq_counts:
		for genre in freq_counts[relation]:
			for sig_type in singal_types:
				if sig_type not in freq_counts[relation][genre]:
					freq_counts[relation][genre][sig_type] = 0

	# eliminate topic-solutionhood because it only occura 8 times
	if "topic-solutionhood" in freq_counts:
		freq_counts.pop("topic-solutionhood")

	return freq_counts

def pairwise_jsd(freq_counts):
	pair_distances = {}
	singal_types = ['dm', 'grf', 'lex', 'mrf', 'num', 'ref', 'sem', 'syn']
	for relation in freq_counts:
		pair_distances[relation] = {}
		genre_pairs_seen = set()
		for genre1 in freq_counts[relation]:
			for genre2 in freq_counts[relation]:
				if (genre2, genre1) not in genre_pairs_seen and genre1 != genre2:
					genre1_counts = []
					genre2_counts = []
					for sig in singal_types:
						genre1_counts.append(freq_counts[relation][genre1][sig])
						genre2_counts.append(freq_counts[relation][genre2][sig])
					pair_distances[relation][(genre2, genre1)] = distance.jensenshannon(genre1_counts, genre2_counts)
					genre_pairs_seen.add((genre2, genre1))
					genre_pairs_seen.add((genre1, genre2))

	return pair_distances

def rank_variation(pair_distances):
	# Which relations show the most inter-genre variation?
	# For each relation, calculate pair-wise avg of JSD, then sort in decending order
	relation_avg_jsd = []
	for relation in pair_distances:
		pairs_count = len(pair_distances[relation])
		jsd_sum = 0
		for pair in pair_distances[relation]:
			jsd_sum += pair_distances[relation][pair]
		relation_avg_jsd.append((relation, jsd_sum / pairs_count))
	relation_jsd_sorted_descending = sorted(relation_avg_jsd, key=lambda x: x[1], reverse=True)
	for rel_jsd in relation_jsd_sorted_descending:
		print(rel_jsd)
		#('organization', 0.3575705520997728)
		#('explanation', 0.34831146431631826)
		#('restatement', 0.2965876654702925)
		#('context', 0.2741118748162802)
		#('joint', 0.2352734773629788)
		#('mode', 0.19919851956411835)
		#('causal', 0.19592911381159392)
		#('contingency', 0.18613163957265172)
		#('elaboration', 0.17102395611252416)
		#('purpose', 0.11921745935527214)
		#('attribution', 0.11417348527138776)
		#('adversative', 0.08437153764023812)
		#('evaluation', 0.02908569613791581)

	return relation_jsd_sorted_descending

def visualize_ranking(ranking, title, outfile):
	categories = [x[0] for x in ranking]
	values = [x[1] for x in ranking]
	# bar plot
	plt.figure(figsize=(8, 6))  # Optional: adjust the size of the plot
	plt.bar(categories, values, color='skyblue')

	# title and labels
	plt.title(title)
	plt.xlabel('Relations')
	plt.ylabel('Avg. Pairwise JSD between Genres')
	plt.xticks(rotation=90)
	plt.tight_layout()

	#plt.savefig('visualizations/' + outfile, bbox_inches='tight')  # Save the plot as a PNG file with tight bounding box
	#plt.show()

	return

def make_distance_matrix(distances):
	#genres = ["academic", "bio", "conversation", "court", "essay", "fiction", "interview", 
	#"letter", "news", "podcast", "reddit", "speech", "textbook", "vlog", "voyage", "whow"]
	attested_genres = sorted(list(set([x[0] for x in distances.keys()] + [x[1] for x in distances.keys()])))

	distance_matrix = []
	for genre1 in attested_genres:
		row = []
		for genre2 in attested_genres:
			if genre1 == genre2:
				row.append(0)
			elif (genre1, genre2) in distances:
				row.append(distances[(genre1, genre2)])
			elif (genre2, genre1) in distances:
				row.append(distances[(genre2, genre1)])
			else:
				print("DANGER")
		distance_matrix.append(row)

	return np.array(distance_matrix), attested_genres


def make_dendrogram(relation, pair_distances, outfile):
	# Will illustrate which genres signal the gives relation most similarly/differently
	# Make distance matrix
	distance_matrix, genres = make_distance_matrix(pair_distances[relation])
	dists = squareform(distance_matrix)
	linkage_matrix = linkage(dists, "average")
	# Make visualizaion
	dendrogram(linkage_matrix, labels=genres)
	plt.title('Signaling Similarity between Genres for Relation: ' + relation.capitalize())
	plt.xticks(rotation=45)
	plt.tight_layout()
	#plt.savefig('visualizations/' + outfile)  # Save the plot as a PNG file with tight bounding box
	plt.show()
	return

def coarse_level_variation():
	datafile = "GUM_signals.tsv"
	freq_counts = frequency_counts_rel_genre(datafile, coarse=True)
	print(freq_counts["context"])
	pair_distances = pairwise_jsd(freq_counts)
	ranking = rank_variation(pair_distances)
	visualize_ranking(ranking, "Inter-Genre Variation of Coarse Relations", "coarse_rel_inter_genre_var.png")
	dendro_relations = ["organization", "explanation", "causal"]
	for rel in dendro_relations:
		make_dendrogram(rel, pair_distances, "dendrogram_" + rel + ".png")
	return


def fine_grained_level_variation():
	datafile = "GUM_signals.tsv"
	freq_counts = frequency_counts_rel_genre(datafile, coarse=False)
	pair_distances = pairwise_jsd(freq_counts)
	ranking = rank_variation(pair_distances)
	visualize_ranking(ranking, "Inter-Genre Variation of Fine-Grained Relations", "fine_grained_rel_inter_genre_var.png")
	organization_distances = {"organization-heading": pair_distances["organization-heading"], "organization-phatic": pair_distances["organization-phatic"], 
							"organization-preparation": pair_distances["organization-preparation"]}
	organization_ranking = rank_variation(organization_distances)
	visualize_ranking(organization_ranking, "Inter-Genre Variation of Organization Relations", "organization_rel_inter_genre_var.png")
	explanation_distances = {"explanation-evidence": pair_distances["explanation-evidence"], "explanation-justify": pair_distances["explanation-justify"], 
							"explanation-motivation": pair_distances["explanation-motivation"]}
	explanation_ranking = rank_variation(explanation_distances)
	visualize_ranking(explanation_ranking, "Inter-Genre Variation of Explanation Relations", "explanation_rel_inter_genre_var.png")
	causal_distances = {"causal-cause": pair_distances["causal-cause"], "causal-result": pair_distances["causal-result"]}
	causal_ranking = rank_variation(causal_distances)
	visualize_ranking(causal_ranking, "Inter-Genre Variation of Causal Relations", "causal_rel_inter_genre_var.png")
	dendro_relations = ["organization-heading", "organization-phatic", "organization-preparation", "explanation-evidence",
						 "explanation-justify", "explanation-motivation", "causal-cause", "causal-result"]
	for rel in dendro_relations:
		make_dendrogram(rel, pair_distances, "dendrogram_" + rel + ".png")

	return


def test():
	categories = ['dm', 'grf', 'lex', 'mrf', 'num', 'ref', 'sem', 'syn']
	values = [24, 2, 11, 1, 0, 17, 91, 8]
	# bar plot
	plt.figure(figsize=(8, 6))  # Optional: adjust the size of the plot
	#plt.bar(categories, values, color='skyblue')
	plt.pie(values, labels=categories)

	# title and labels
	plt.title('Context Academic Proportions')
	#plt.xlabel('Signal Types')
	#plt.ylabel('Counts')
	#plt.xticks(rotation=90)
	plt.tight_layout()

	#plt.savefig('visualizations/' + outfile, bbox_inches='tight')  # Save the plot as a PNG file with tight bounding box
	plt.show()
	return

if __name__ == "__main__":
	test()
	#coarse_level_variation()
	#fine_grained_level_variation()
