from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import random
import csv


def get_colors():
	colors = {"organization": "red", "restatement": "darkorange", "explanation": "gold", "context": "yellow",
			  "topic": "yellowgreen", "joint": "green", "causal": "turquoise", "contingency": "teal", "mode": "skyblue",
			  "elaboration": "dodgerblue", "purpose": "blue", "attribution": "darkviolet", "adversative": "fuchsia",
			  "evaluation": "hotpink"}

	return colors


def get_data(infile, n=None):
	data = []
	with open(infile, "r") as file:
		reader = csv.reader(file, delimiter='\t')
		next(reader, None)  # skip the headers
		for line in reader:
			data.append(line)
	if n is not None:
		genre_doc_dict = {}
		doc_list = []
		new_data = []
		# make dictionary of genres and their doc
		for line in data:
			genre = line[1]
			doc_name = line[0]
			if genre in genre_doc_dict:
				genre_doc_dict[genre].add(doc_name)
			else:
				genre_doc_dict[genre] = set([doc_name])
		# for each genre, randomly select n docs and add to list
		for genre in genre_doc_dict:
			doc_names = random.sample(list(genre_doc_dict[genre]), n)
			doc_list += doc_names
		# for line in data, add to new data if it is from a doc on the doc list
		for line in data:
			doc = line[0]
			if doc in doc_list:
				new_data.append(line)
		data = new_data

	return data


def frequency_counts_rel_genre(lines, coarse=False):
	coarse_relations_set = set()
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

	# fill in zeros with epsilon = 1e-10
	epsilon = 1e-5
	singal_types = ['dm', 'grf', 'lex', 'mrf', 'num', 'ref', 'sem', 'syn']
	for relation in freq_counts:
		for genre in freq_counts[relation]:
			for sig_type in singal_types:
				if sig_type not in freq_counts[relation][genre]:
					freq_counts[relation][genre][sig_type] = epsilon

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
			for rel in freq_counts_normalized:
				if relation in rel:
					for genre in freq_counts_normalized[rel]:
						if genre in collaposed_avg[relation]:
							for sig_type in freq_counts_normalized[rel][genre]:
								collaposed_avg[relation][genre][sig_type] += freq_counts_normalized[rel][genre][sig_type]
						else:
							collaposed_avg[relation][genre] = freq_counts_normalized[rel][genre]
			# divide by n
			for genre in collaposed_avg[relation]:
				n = round(sum(collaposed_avg[relation][genre].values()))
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
	#for rel_jsd in relation_jsd_sorted_descending:
	#	print(rel_jsd)

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
	data = get_data(datafile)
	freq_counts = frequency_counts_rel_genre(data, coarse=True)
	pair_distances = pairwise_jsd(freq_counts)
	ranking = rank_variation(pair_distances)
	visualize_ranking(ranking, "Inter-Genre Variation of Coarse Relations", "coarse_rel_inter_genre_var.png")
	dendro_relations = ["organization", "restatement", "explanation"]
	for rel in dendro_relations:
		make_dendrogram(rel, pair_distances, "dendrogram_" + rel + ".png")

	return


def fine_grained_level_variation():
	datafile = "GUM_signals.tsv"
	data = get_data(datafile)
	freq_counts = frequency_counts_rel_genre(data, coarse=False)
	pair_distances = pairwise_jsd(freq_counts)
	ranking = rank_variation(pair_distances)
	visualize_ranking(ranking, "Inter-Genre Variation of Fine-Grained Relations", "fine_grained_rel_inter_genre_var.png")

	return


def ranking_consistency(iterations=10, doc_sample_size=5, coarse=False):
	datafile = "GUM_signals.tsv"
	rankings = []
	pair_count = 0
	kendalltau_sum = 0
	spearmanr_sum = 0
	pearsonr_sum = 0
	for i in range(iterations):
		data = get_data(datafile, n=doc_sample_size)
		freq_counts = frequency_counts_rel_genre(data, coarse=coarse)
		pair_distances = pairwise_jsd(freq_counts)
		ranking1 = rank_variation(pair_distances)
		# make rank
		for ranking2 in rankings:
			pair_count += 1
			relation_info = {}
			rank = 0
			for line in ranking1:
				rank += 1
				relation_info[line[0]] = {"rank1": rank, "score1": line[1]}
			rank = 0
			for line in ranking2:
				rank += 1
				relation_info[line[0]]["rank2"] = rank
				relation_info[line[0]]["score2"] = line[1]

			rank1, rank2, score1, score2 = [], [], [], []
			for relation in relation_info:
				rank1.append(relation_info[relation]["rank1"])
				rank2.append(relation_info[relation]["rank2"])
				score1.append(relation_info[relation]["score1"])
				score2.append(relation_info[relation]["score2"])

			tau, p_value = stats.kendalltau(rank1, rank2)
			#print("Kendall's Tau:", tau, "p_value:", p_value)
			kendalltau_sum += tau

			corr, p_value = stats.spearmanr(score1, score2)
			#print("Spearman's Rank:", corr, "p_value:", p_value)
			spearmanr_sum += corr

			corr, p_value = stats.pearsonr(score1, score2)
			#print("Pearson's:", corr, "p_value:", p_value)
			pearsonr_sum += corr
		rankings.append(ranking1)

	print("Configuration:", "Iterations:", iterations, "Doc Sample Size:", doc_sample_size, "Coarse:", coarse)
	print("Ranking Compared:", pair_count)
	print("Avg. Kendall's Tau:", kendalltau_sum / pair_count)
	print("Avg. Spearman's Rank:", spearmanr_sum / pair_count)
	print("Avg. Pearson's:", pearsonr_sum / pair_count)

	return


if __name__ == "__main__":
	coarse_level_variation()
	fine_grained_level_variation()
	ranking_consistency(iterations=50, doc_sample_size=5, coarse=False)
	ranking_consistency(iterations=50, doc_sample_size=5, coarse=True)
