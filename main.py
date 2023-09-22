# Databricks notebook source
# MAGIC %run ./read_data

# COMMAND ----------

# MAGIC %run ./preprocessing

# COMMAND ----------

# MAGIC %run ./training_model

# COMMAND ----------

# MAGIC %run ./return_embed

# COMMAND ----------

# MAGIC %run ./top_n

# COMMAND ----------

import pandas as pd

# You should import your necessary libraries and functions here (e.g., read_data, model_train, output_text, return_embed, top_n).

def load_model(model, column_name, vector_size, window_size):
  df = read_data()
  x = output_text(df, column_name)
  word2vec_model = model_train(x, vector_size, window_size, model)
  vectors = return_embed(word2vec_model, df, column_name)
  Vec = pd.DataFrame(vectors).transpose()  # Saving vectors of each abstract in data frame
  # Save the vectors to different files based on the selected model
#  if model == 'Skipgram':
#    Vec.to_csv('/dbfs/mnt/data/data/output/Skipgram_vec.csv')
#  else:
#    Vec.to_csv('/dbfs/mnt/data/data/output/Fasttext_vec.csv')

if __name__ == '__main__':
  # Load and save Skipgram model
  load_model('Skipgram', 'Abstract', 100, 3)

  # Load and save Fasttext model
  load_model('Fasttext', 'Abstract', 100, 3)

  # Assuming you have a function top_n to find top similar words
  results, sim = top_n('Coronavirus', 'Skipgram', 'Abstract')
  results1, sim1 = top_n('Coronavirus', 'Fasttext', 'Abstract')

