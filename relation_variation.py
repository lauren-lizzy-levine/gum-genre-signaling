from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import csv
import numpy as np
import matplotlib.pyplot as plt


def get_colors():
	colors = {"organization": "red", "restatement": "darkorange", "explanation": "gold", "context": "yellow",
			  "topic": "yellowgreen", "joint": "green", "causal": "turquoise", "contingency": "teal", "mode": "skyblue",
			  "elaboration": "dodgerblue", "purpose": "blue", "attribution": "darkviolet", "adversative": "fuchsia",
			  "evaluation": "hotpink"}

	return colors


def frequency_counts_rel_genre(infile, coarse=False):
	lines = []
	coarse_relations_set = set()
	with open(infile, "r") as file:
		reader = csv.reader(file, delimiter='\t')
		next(reader, None)  # skip the headers
		for line in reader:
			lines.append(line)
	freq_counts = {}
	for line in lines:
		genre = line[1]
		coarse_relations_set.add(line[2])
		relation = line[3] # fine grained relation
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

	# eliminate topic-solutionhood because it only occurs 8 times
	if "topic-solutionhood" in freq_counts:
		freq_counts.pop("topic-solutionhood")

	if coarse:
		# normalize each fine grained relation
		freq_counts_normalized = {}
		for relation in freq_counts:
			freq_counts_normalized[relation] = {}
			for genre in freq_counts[relation]:
				freq_counts_normalized[relation][genre] = {}
				total = sum(freq_counts[relation][genre].values())
				for sig_type in freq_counts[relation][genre]:
					freq_counts_normalized[relation][genre][sig_type] = freq_counts[relation][genre][sig_type] / total
		# collapse into coarse relations by averaging proportions of fine grained
		collaposed_avg = {}
		for relation in list(coarse_relations_set):
			collaposed_avg[relation] = {}
			n = 0
			for rel in freq_counts_normalized:
				if relation in rel:
					n += 1
					for genre in freq_counts_normalized[rel]:
						if genre in collaposed_avg[relation]:
							for sig_type in freq_counts_normalized[rel][genre]:
								collaposed_avg[relation][genre][sig_type] += freq_counts_normalized[rel][genre][sig_type]
						else:
							collaposed_avg[relation][genre] = freq_counts_normalized[rel][genre]
			# divide by n
			for genre in collaposed_avg[relation]:
				for sig_type in collaposed_avg[relation][genre]:
					collaposed_avg[relation][genre][sig_type] = collaposed_avg[relation][genre][sig_type] / n
		freq_counts = collaposed_avg

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

	return relation_jsd_sorted_descending


def visualize_ranking(ranking, title, outfile):
	categories = [x[0] for x in ranking]
	values = [x[1] for x in ranking]
	# make colors input
	color_assignment = get_colors()
	colors = []
	for relation in categories:
		if relation not in color_assignment:
			relation = relation.split("-")[0]
		colors.append(color_assignment[relation])
	# bar plot
	plt.figure(figsize=(8, 6))  # Optional: adjust the size of the plot
	plt.bar(categories, values, color=colors)

	# title and labels
	plt.title(title)
	plt.xlabel('Relations')
	plt.ylabel('Avg. Pairwise JSD between Genres')
	plt.xticks(rotation=90)
	plt.tight_layout()

	#plt.savefig('visualizations/' + outfile, bbox_inches='tight')  # Save the plot as a PNG file with tight bounding box
	plt.show()

	return


def make_distance_matrix(distances):
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
	# Illustrates which genres signal the given relation most similarly/differently
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
	pair_distances = pairwise_jsd(freq_counts)
	ranking = rank_variation(pair_distances)
	visualize_ranking(ranking, "Inter-Genre Variation of Coarse Relations", "coarse_rel_inter_genre_var.png")
	dendro_relations = ["organization", "restatement", "explanation"]
	for rel in dendro_relations:
		make_dendrogram(rel, pair_distances, "dendrogram_" + rel + ".png")

	return


def fine_grained_level_variation():
	datafile = "GUM_signals.tsv"
	freq_counts = frequency_counts_rel_genre(datafile, coarse=False)
	pair_distances = pairwise_jsd(freq_counts)
	ranking = rank_variation(pair_distances)
	visualize_ranking(ranking, "Inter-Genre Variation of Fine-Grained Relations", "fine_grained_rel_inter_genre_var.png")

	return


if __name__ == "__main__":
	coarse_level_variation()
	fine_grained_level_variation()
