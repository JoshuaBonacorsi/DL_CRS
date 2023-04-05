import torch
import pandas as pd
import numpy as np
import random
import time
import scipy

def neg_sampling(ratings_df, n_neg=1, neg_val=0, pos_val=1, percent_print=5):
  """version 1.2: 1 positive 1 neg (2 times bigger than the original dataset by default)
    Parameters:
    input rating data as pandas dataframe: userId|movieId|rating
    n_neg: take n_negative / 1 positive
    Returns:
    negative sampled set as pandas dataframe
            userId|movieId|interact (implicit)
  """

  ratings_df.userId = ratings_df.userId.astype('category').cat.codes.values
  ratings_df.movieId = ratings_df.movieId.astype('category').cat.codes.values
  
  sparse_mat = scipy.sparse.coo_matrix((ratings_df.rating, (ratings_df.userId, ratings_df.movieId)))
  dense_mat = np.asarray(sparse_mat.todense())
  print(dense_mat.shape)
  nsamples = ratings_df[['userId', 'movieId']]
  nsamples['interact'] = nsamples.apply(lambda row: 1, axis=1)
  length = dense_mat.shape[0]
  printpc = int(length * percent_print/100)
  nTempData = []
  i = 0
  start_time = time.time()
  stop_time = time.time()
  extra_samples = 0
  for row in dense_mat:
    if(i%printpc==0):
      stop_time = time.time()
      print("processed ... {0:0.2f}% ...{1:0.2f}secs".format(float(i)*100 / length, stop_time - start_time))
      start_time = stop_time
    n_non_0 = len(np.nonzero(row)[0])
    zero_indices = np.where(row==0)[0]
    if(n_non_0 * n_neg + extra_samples > len(zero_indices)):
      print(i, "non 0:", n_non_0,": len ",len(zero_indices))
      neg_indices = zero_indices.tolist()
      extra_samples = n_non_0 * n_neg + extra_samples - len(zero_indices)
    else:
      neg_indices = random.sample(zero_indices.tolist(), n_non_0 * n_neg + extra_samples)
      extra_samples = 0
    nTempData.extend([(uu, ii, rr) for (uu, ii, rr) in zip(np.repeat(i, len(neg_indices))
                    , neg_indices, np.repeat(neg_val, len(neg_indices)))])
    i+=1
  nsamples=nsamples.append(pd.DataFrame(nTempData, columns=["userId","movieId", "interact"]),ignore_index=True)
  return nsamples

class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
    # len(movie_dataset)
    def __len__(self):
        return len(self.users)
    # movie_dataset[1] 
    def __getitem__(self, item):

        users = self.users[item] 
        movies = self.movies[item]
        ratings = self.ratings[item]
        
        return {
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.long),
        }