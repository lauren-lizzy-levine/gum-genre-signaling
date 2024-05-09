import pandas as pd
import conllu
import glob


def process_gum_signals():
    # read in gum files
    gum_file_list = glob.glob("../gum/dep/*.conllu")
    file_token_id = set()
    gum_signals = []

    for file in gum_file_list:
        file_name = file.split("/")[-1].split(".")[0]
        genre = file_name.split("_")[1]
        # load data
        with open(file, "r") as f:
            file_text = f.read()
            sentences = conllu.parse(file_text)

        for sentence in sentences:
            for token in sentence:
                if token["misc"] and "Discourse" in token["misc"]:
                    token_discourse = token["misc"]["Discourse"]
                    discourse_relations = token_discourse.split(";")
                    for relation in discourse_relations:
                        # extract relation
                        relation_split = relation.split(":")
                        if len(relation_split[-1]) == 1:
                        	# no signal
                        	continue
                        relation = relation_split[0]
                        if "_" in relation:
                            relation = relation[:-2]
                        coarse_relation = relation.split("-")[0]
                        # extract signal information
                        signal_info = relation_split[-1]
                        signal_info_list = signal_info.split("+")
                        for info in signal_info_list:
                            info_split = info.split("-")
                            signal_type = info_split[0]
                            signal_subtype = info_split[1]
                            token_ids = str("-".join(info_split[2:]))

                            gum_signals.append([file_name, genre, coarse_relation, relation, signal_type, signal_subtype, token_ids])

    # write to file with pandas
    df = pd.DataFrame(gum_signals, columns =["DOC_NAME", "GENRE", "COARSE_RELATION", "RELATION", "SIGNAL_TYPE", "SIGNAL_SUBTYPE", "TOKEN_IDS"])
    df.to_csv("GUM_signals.tsv", index=False, sep="\t")

    return

if __name__ == "__main__":
	process_gum_signals()