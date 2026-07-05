import pickle

with open("models/product_similarity.pkl", "rb") as f:
    similarity_df = pickle.load(f)

similarity_df = similarity_df.astype("float32")

with open("models/product_similarity.pkl", "wb") as f:
    pickle.dump(similarity_df, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done shrinking!")