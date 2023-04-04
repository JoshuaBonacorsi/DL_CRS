import torch
import torch.nn as nn

class DLCRS(nn.Module):

    def __init__(self,
                 n_users, # liste des users id
                 n_movies, # liste des movies id
                 n_latents, # nombre de latents factors 
                 n_layers,
                 dropout=None): # dictionnaire de configuration pour les layers fully connected

        super().__init__()
        
        # Embedding part
        self.user_embed = nn.Embedding(num_embeddings=n_users, embedding_dim=n_latents)
        self.movie_embed = nn.Embedding(num_embeddings=n_movies, embedding_dim=n_latents)

        # Fully connected layers part

        self.fc_activation = nn.ReLU()
        self.fc_layers = torch.nn.ModuleList()

        if dropout :
            # Add dropout layers 
            self.do_layers = nn.ModuleList()

        in_size, out_size = 2*n_latents,50*n_layers

        for i in range(n_layers) :
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.do_layers.append(torch.nn.Dropout(dropout))
            in_size = out_size
            out_size -= 50 

        # Output layer part
        self.fc_layers.append(nn.Linear(50,2))
        
    def forward(self, users, movies, ratings=None):

        users_embed = self.user_embed(users)
        movies_embed = self.movie_embed(movies)

        # concat both vectors
        output = torch.cat([users_embed,movies_embed],dim=1)

        # Fully connected layers loop
        for fc_layer, dropout_layer in zip(self.fc_layers[:-1], self.do_layers):
            output = fc_layer(output)
            output = dropout_layer(output)
            output = self.fc_activation(output)

        # Output layer
        output = self.fc_layers[-1](output)

        return(output)