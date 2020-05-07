_ = """def pretrain(self, dataloader):
    import time
    import logging
    import consts
    # Put the model in training mode.
    self.train()
    
    pretrain_loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
    pretrain_temp_model = torch.nn.Sequential(self.encoder.embedding.ff,
                                                    torch.nn.Linear(self.encoder.embedding_dim, self.alphabet_size),
                                                    torch.nn.Softmax()
                                             )
    
    batch_start_time = time.time()
    pretrain_opt = torch.optim.SGD(pretrain_temp_model.parameters(), lr=0.01, momentum=0.9)
    
    # For each batch of data ...
    for batch_idx, (input_batch, output_batch) in enumerate(dataloader):
        
        # Zero out the previous gradient information.
        pretrain_opt.zero_grad()

        # Split the batch into semantic parts.
        family = input_batch.family
        language = input_batch.language
        tags = input_batch.tags
        lemma = input_batch.lemma
        form = output_batch.form
        tags_str = output_batch.tags_str
        lemma_str = output_batch.lemma_str
        form_str = output_batch.form_str
        
        #TODO: Unless the dataloader is sending the data to the appropriate device, maybe handle it here.
        
        # Run the model on this batch of data.
        probabilities = [ pretrain_temp_model(lemma[i]) for i in range(len( lemma )) ]
        outputs = [torch.argmax(probability, dim=1) for probability in probabilities]
        
        # Compute the batch loss.
        batch_loss = 0.0
        batch_size = len(tags)
        for i in range(batch_size):
            batch_loss += pretrain_loss_function(probabilities[i], lemma[i])
            
        
        # Update model parameter.
        batch_loss.backward()
        pretrain_opt.step()
        
        #Logging to console
        for i in range(batch_size):
            output_str = "".join([dataloader.dataset.alphabet_input[int(integral)]
                                  for integral in outputs[i]])
            language_family =\
                dataloader.dataset.language_collection[int(family[i][0])]
            language_object = language_family[int(language[i][0])]
            logging.getLogger(consts.MAIN).debug(
                "PRETRAIN stem: {},"
                "\ttarget: {},"
                "\ttags: {}"
                "\tlanguage: {}/{}"
                "\toutput: '{}'".format(lemma_str[i], form_str[i], tags_str[i], language_family.name,
                                        language_object.name, output_str))
        
        
        
        #benchmark stuff
        batch_end_time = time.time()
        batches_per_second = 1.0/(batch_end_time-batch_start_time)
        batch_start_time = batch_end_time
        #benchmark Log the benchmark to wandb
        items_per_sec = int(len(output_batch.lemma_str))*batches_per_second

        # Log the outcome of this batch.
        logging.getLogger(consts.MAIN
            ).info('PRETRAIN Loss: {:.6f}\tItems/s: {:.2f} '.format(
                            batch_loss.item() / batch_size,
                            items_per_sec
                        )
                    )
        
        

"""
