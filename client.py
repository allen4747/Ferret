from optimizers.ferret_optimizer import *
from tqdm import tqdm
from copy import deepcopy

class Client(object):
    def __init__(self, idx, args, candidate_seeds, train_loader):
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.train_iterator = iter(self.train_loader)
        self.model = None

        self.device = torch.device(f'cuda:{args.device}')
        self.candidate_seeds = candidate_seeds

    def local_train_with_seed_pool(self, pulled_model, cur_round):
        self.model = pulled_model
        old_params = [(name, deepcopy(param.data)) for name, param in self.model.named_parameters() if param.requires_grad]
        
        self.model.to(self.device)
        
        # initialize a seed pool
        self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

        lr = self.args.lr
        
        if self.args.batch_or_epoch == 'epoch':
            iter_steps = self.args.local_step * len(self.train_loader)
            print("iter_steps:", iter_steps)
        else:
            iter_steps = self.args.local_step
           
        # Ferret Framework
        framework = FerretFramework(self.model, args=self.args, lr=lr, candidate_seeds=self.candidate_seeds)
        self.model.train()
        self.model.zero_grad()
        
        # with torch.inference_mode():
        if self.args.batch_or_epoch == 'batch':
                loss_total_train = 0.0
                num_trained = 0
                progress_bar = tqdm(range(iter_steps))
                
        for cur_step in range(iter_steps):
            # init epoch progress bar
            if self.args.batch_or_epoch == 'epoch':
                if cur_step % len(self.train_loader) == 0:
                    loss_total_train = 0.0
                    num_trained = 0
                    progress_bar = tqdm(range(len(self.train_loader)))
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                batch = next(self.train_iterator)
            batch = {
                'input_ids': batch['input_ids'].to(self.device),
                'labels': batch['labels'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device) 
            }
            
            if cur_step % self.args.n_accum == self.args.n_accum - 1:
                apply_optim_step = True
            else:
                apply_optim_step = False
            
            logits, loss = framework.step(batch, apply_optim_step=apply_optim_step)
            
            progress_bar.update(1)
            if (not torch.isnan(loss)) and (self.args.grad_clip <= 0 or loss != 0.0):
                loss_total_train += loss
                num_trained += len(batch['input_ids'])
            if self.args.batch_or_epoch == 'epoch':
                progress_bar.set_description(f'client {self.idx} train at epoch {int(cur_step / len(self.train_loader)) + 1}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')
            else:
                progress_bar.set_description(f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')

        self.local_seed_pool = framework.project_update(dict(old_params))
            
        # save both CPU and GPU memory
        del old_params, framework
        self.model = None

    def clear_model(self):
        # clear model to same memory
        self.model = None

    def migrate(self, device):
        """
        migrate a client to a new device
        """
        self.device = device

    def pull(self, forked_global_model):
        """
        pull model from the server
        """
        self.model = forked_global_model