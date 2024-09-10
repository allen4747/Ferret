import torch
import random
import numpy as np

class FerretFramework(object):
    def __init__(self, model, args, lr, candidate_seeds):
        self.args = args
        self.lr = lr
        self.model = model
        self.named_parameters_to_optim = [
            (name, param) for name, param in self.model.named_parameters() if param.requires_grad
        ]
        self.candidate_seeds = candidate_seeds
        self.optim = torch.optim.SGD(
            [p for n, p in self.named_parameters_to_optim], 
            lr=args.lr, momentum=0.0, weight_decay=args.weight_decay
        )
        self.param_groups = self._group_parameters()
        
    def _group_parameters(self):
        # Group parameters with similar dimensions
        groups = []
        current_group = []
        current_dim = 0
        target_dim = int(0.9 * np.median([p.numel() for _, p in self.named_parameters_to_optim]))  # Adjust this value to change group sizes
        for name, param in self.named_parameters_to_optim:
            param_dim = param.numel()
            if current_dim + param_dim > target_dim and current_group:
                groups.append(current_group)
                current_group = []
                current_dim = 0
            current_group.append((name, param))
            current_dim += param_dim

        if current_group:
            groups.append(current_group)

        return groups
        
    def step(self, batch, apply_optim_step=False):
        """
        Perform a training step using a first-order optimizer.
        """
        logits, loss = self.forward(batch)
        (loss / self.args.n_accum).backward()
        
        if self.args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        
        if apply_optim_step:
            self.optim.step()
            self.optim.zero_grad()
        
        return logits.detach(), loss.detach()

    def project_update(self, old_params):
        with torch.no_grad():
            coordinate_of_seeds = []
            total_coordinates = 0

            max_delta_norm = 1e-8
            for group in self.param_groups:
                group_delta = torch.cat([old_params[name].to(param.data.device).flatten() - param.data.flatten() for name, param in group])
                delta_norm = group_delta.norm()
                if (not torch.isnan(delta_norm).any().item()) and (delta_norm.item() > max_delta_norm):
                    max_delta_norm = delta_norm.item()

            for group in self.param_groups:
                group_delta = torch.cat([old_params[name].to(param.data.device).flatten() - param.data.flatten() for name, param in group])
                group_size = group_delta.numel()
                
                coordinate = torch.zeros(self.args.K, device=group_delta.device, dtype=group_delta.dtype)
                if torch.isnan(group_delta.norm()).any().item():
                    max_n_seeds = 2
                else:
                    max_n_seeds = max(int(group_delta.norm().item() / max_delta_norm * self.args.K), 2)
                
                total_coordinates += max_n_seeds
                
                seed_idxs = random.sample(range(self.args.K), max_n_seeds)
                for idx in seed_idxs:
                    seed = self.candidate_seeds[idx]
                    sqrt_d = 1 / group_size ** 0.5
                    torch.manual_seed(seed)
                    torch.cuda.random.manual_seed(seed)
                    base = torch.empty(group_size, device=group_delta.device, dtype=group_delta.dtype)
                    base = torch.nn.init.trunc_normal_(base, mean=0., std=1., a=-sqrt_d, b=sqrt_d)
                    coordinate[idx] = torch.sum(group_delta * base)
                
                coordinate *= group_size / max_n_seeds
                coordinate_of_seeds.append(coordinate)
            
            print("total coordinates to send:", total_coordinates)
            self.local_seed_pool = {}
            coordinate_of_seeds = torch.stack(coordinate_of_seeds, dim=1)
            for i, seed in enumerate(self.candidate_seeds):
                self.local_seed_pool[seed] = coordinate_of_seeds[i]
        return self.local_seed_pool

    def forward(self, batch):
        """
        Forward pass to compute the loss.
        """
        outputs = self.model(**batch)
        logits = outputs.logits
        loss = outputs.loss
        return logits, loss

    def update(self, seed=None, grad=None, max_norm=10):
        """
        Update the parameters using the true/estimated gradients.
        """
        with torch.no_grad():
            grad_idx = 0
            for group in self.param_groups:
                group_size = sum(param.numel() for _, param in group)
                sqrt_d = group_size ** -0.5
                torch.manual_seed(seed)
                torch.cuda.random.manual_seed(seed)
                base = torch.empty(group_size, device=grad[grad_idx].device, dtype=grad[grad_idx].dtype)
                base = torch.nn.init.trunc_normal_(base, mean=0., std=1., a=-sqrt_d, b=sqrt_d)
                base.mul_(grad[grad_idx])
                if torch.isfinite(base).all():
                    total_norm = torch.linalg.norm(base)
                    if torch.isfinite(total_norm):
                        clip_coef = max_norm / (total_norm + 1e-8)
                        if clip_coef < 1:
                            base.mul_(clip_coef)
                        start = 0
                        for name, param in group:
                            end = start + param.numel()
                            param_update = base[start:end].reshape(param.shape)
                            param.data.sub_(self.args.slr * param_update)
                            start = end
                grad_idx += 1

