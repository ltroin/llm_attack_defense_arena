import argparse
import json
import logging
import pathlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from jailbroken.utils import init_experiment, save_experiment_json

parser = argparse.ArgumentParser()
parser.add_argument("prompt_file", type=str)
parser.add_argument("--handle", type=str, default="dedup_generated")


def main():
    args = parser.parse_args()
    init_experiment(args.handle)
    save_experiment_json("args", vars(args))

    # Load prompts

    prompts_path = pathlib.Path(args.prompt_file)
    with open(prompts_path, "r") as f:
        prompt_keys, prompts = zip(*json.load(f).items())

    # Deduplicate prompts via TF-IDF cosine similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(prompts)
    similarity = cosine_similarity(vectors)

    # Find duplicates
    duplicates = []
    adjacency_list = {}
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            if similarity[i, j] > 0.3:
                duplicates.append((i, j, similarity[i, j]))
                adjacency_list.setdefault(prompt_keys[i], []).append(prompt_keys[j])
                adjacency_list.setdefault(prompt_keys[j], []).append(prompt_keys[i])

    # Print duplicates
    duplicates = sorted(duplicates, key=lambda x: x[2], reverse=True)
    for i, j, s in duplicates[:20]:
        logging.info(f"Similarity {s}:\n{prompts[i]}\n{prompts[j]}\n")
    for i, j, s in duplicates[-20:]:
        logging.info(f"Similarity {s}:\n{prompts[i]}\n{prompts[j]}\n")
    logging.info(
        f"Found {len(duplicates)} duplicates involving {len(adjacency_list)} keys"
    )

    removed_keys = []
    while adjacency_list:
        # Find most duplicated
        max_key = max(adjacency_list, key=lambda key: len(adjacency_list[key]))
        logging.info(f"Removing key {max_key}")
        if len(adjacency_list[max_key]) > 0:
            removed_keys.append(max_key)
            for neighbor in adjacency_list[max_key]:
                adjacency_list[neighbor].remove(max_key)
        del adjacency_list[max_key]
    logging.info(f"Removed {len(removed_keys)} keys: {sorted(removed_keys)}")

    # Save keys to remove
    with open(prompts_path.parent / "duplicate_keys.json", "w") as f:
        json.dump(removed_keys, f)
    logging.info(f"Saved removed keys to {prompts_path.parent / 'duplicate_keys.json'}")


if __name__ == "__main__":
    main()
