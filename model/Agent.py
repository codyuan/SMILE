import torch
import torch.nn as nn
import numpy as np
from functools import partial
from util.functions import exists,cosine_beta_schedule,cosine_distance,extract,default,normal_log_density

class SMILE(nn.Module):
    def __init__(
        self,
        denoiser,
        policy,
        args,
        betas = None,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.policy = policy
        self.diffusion_steps = args.diffusion_steps

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(self.diffusion_steps)

        sigmas = np.sqrt(np.cumsum(betas**2))
        # sigmas=betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.denoiser_loss_type = args.denoiser_loss_type
        self.policy_loss_type = args.policy_loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('sigmas', to_torch(sigmas))

    def diffuse(self,  a, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(a))
        mean = a
        std = extract(self.sigmas, t, a.shape)

        a_noisy = mean + std * noise

        return (
            a_noisy,   # a_t
            mean,
            std,
            noise
        )

    @torch.no_grad()
    def compute_Q(self,s, a, a_noisy, t_prime):

        pred_noise = self.denoiser(s, a_noisy, t_prime)
        a, *_ = self.diffuse(a, t_prime, pred_noise)

        return  -1.0 * torch.sum((a-a_noisy) ** 2, dim=1)

    def policy_losses(self, s, a, t):

        a_t,*_ = self.diffuse(a=a, t=t)
        a_prime=self.policy(s)

        with torch.no_grad():
            pred_noise = self.denoiser(s, a_t, t)
        a_t_prime,mean_prime,std_prime,_=self.diffuse(a=a_prime, t= t, noise=pred_noise)

        if self.policy_loss_type == 'l1':
            loss = (a_t - a_t_prime).abs().mean()
        elif self.policy_loss_type == 'l2':
            loss = torch.sum((a_t_prime-a_t) ** 2, dim=1).mean()
        elif self.policy_loss_type == 'cosine':
            loss = cosine_distance(a_t,a_t_prime)
        elif self.policy_loss_type=='mle':
            loss = -1.0 * normal_log_density(a_t, mean_prime, std_prime.log(), std_prime).mean()
        else:
            raise NotImplementedError()

        return loss
    
    def diffusion_trust_region_losses(self, s, t):
        
        a=self.policy(s)
        a_t,*_ = self.diffuse(a=a, t=t)
        
        with torch.no_grad():
            pred_noise = self.denoiser(s, a_t, t)
        a_t_prime,mean_prime,std_prime,_=self.diffuse(a=a, t= t, noise=pred_noise)
        
        loss = torch.sum((a_t_prime-a_t) ** 2, dim=1).mean()

        return loss


    def denoiser_losses(self, s, a, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(a))
        a_noisy,*_ = self.diffuse(a=a, t=t, noise=noise)
        pred_noise = self.denoiser(s,a_noisy,t)

        if self.denoiser_loss_type == 'l1':
            loss = (noise - pred_noise).abs().mean()            
        elif self.denoiser_loss_type == 'l2':
            loss = torch.sum((noise-pred_noise) ** 2, dim=1).mean()
        elif self.denoiser_loss_type == 'cosine':
            loss = cosine_distance(noise, pred_noise)
        elif self.denoiser_loss_type == 'kl':
            raise NotImplementedError()
        else:
            raise NotImplementedError()


        return loss

    def denoiser_loss(self, s,a):  
        b, device, = s.shape[0], s.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()   

        return self.denoiser_losses(s, a ,t )

    def policy_loss(self, s ,a ):   
        b, device, = s.shape[0], s.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()  
        
        if self.policy_loss_type == 'dtr':
            return self.diffusion_trust_region_losses(s,t) 
        
        return self.policy_losses(s, a ,t )


    def sample_action(self,state):
        if len(state.shape)==1:
            a_policy = torch.randn((self.action_dim)).to(state.device)
        else:
            a_policy = torch.randn((state.shape[0],self.action_dim)).to(state.device)

        t = torch.ones((1,), dtype=torch.int64).to(state.device)*(self.num_timesteps-1)
        for _ in range(self.num_timesteps-1,-1,-1):
            betas=extract(self.betas, t ,a_policy.shape)
            epsilon=self.forward(state,a_policy,t)
            a_policy= (a_policy-betas*epsilon).squeeze(0)
            t-=1

        return a_policy.cpu().numpy()
