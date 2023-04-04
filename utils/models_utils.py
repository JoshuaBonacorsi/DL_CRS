import torch
import torch.nn as nn

class DLCRS(nn.Module):

    def __init__(self,n_users,n_movies,n_latents):

        super().__init__()
        
        # Latent vectors
        self.user_embed = nn.Embedding(num_embeddings=n_users, embedding_dim=n_latents)
        self.movie_embed = nn.Embedding(num_embeddings=n_movies, embedding_dim=n_latents)

        # Concatenation of latent vectors
        self.out = nn.Linear(2*n_latents,1)
        
    def forward(self, users, movies, ratings=None):

        users_embed = self.user_embed(users)
        movies_embed = self.movie_embed(movies)
        output = torch.cat([users_embed,movies_embed],dim=1)
        output = self.out(output)

        return(output)