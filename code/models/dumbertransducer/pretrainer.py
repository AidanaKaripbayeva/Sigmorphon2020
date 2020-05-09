import time
import logging
import consts

import torch
import torch.nn

from experiments import *
from experiments.experiment import *

class TWGPretrainer(object):
    
    class PretrainEmbedding(torch.nn.Module):
        def __init__(self, a_model):
            super().__init__()
            self.ff = torch.nn.Sequential(a_model.encoder.embedding.ff,
                torch.nn.Linear(a_model.encoder.embedding_dim, a_model.alphabet_size),
                torch.nn.Softmax()
            )
            
            #This choice makes it a bit fragile when I decide on a different way to decode.
            #self.ff = torch.nn.Sequential(a_model.encoder.embedding.ff,
            #    a_model.decoder.ff
            #)
            
        
        def forward(self, data):
            probs = list()
            for d in data:
                probs.append(self.ff(d))
            return probs
    
    def __init__(self, config, model, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pretrain_this_model = model
        self.model = None #The model we actually train
        self.max_epochs = 20
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.model = self.PretrainEmbedding(self.pretrain_this_model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=5.0, momentum=0.1)
        self.schedule = None
    
    def run(self):
            
        """
        Train the model for a single epoch.

        :return: The average training loss in this epoch, as a floating point number.
        """
        # Put the model in training mode.
        self.pretrain_this_model.train()
        self.model.train()
        
        total_loss = 0.0
        batch_start_time = time.time()
        
        for e in range(self.max_epochs):
            # For each batch of data ...
            for batch_idx, one_batch in enumerate(self.train_loader):
                # Zero out the previous gradient information.
                self.optimizer.zero_grad()
                input_batch, output_batch = one_batch
                
                # Run the model on this batch of data.
                probabilities = self.model(input_batch.lemma) #with teacher-forcing
                
                # Compute the batch loss.
                batch_loss = 0.0
                batch_size = len(input_batch.tags)
                for i in range(batch_size):
                    batch_loss += self.loss_function(probabilities[i], input_batch.lemma[i])
                batch_loss /= batch_size
                
                # Update model parameter.
                batch_loss.backward()
                self.optimizer.step()
            
            if self.schedule is not None:
                self.schedule.step()
                
            logging.getLogger(consts.MAIN).info("Pretraining epoch {} done. ".format(e))
            #TODO: Something to decide when to accept.
        
        logging.getLogger(consts.MAIN).info("Done pretraining.")
