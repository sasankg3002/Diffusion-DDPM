import torch
import numpy as np
import pytorch_lightning as pl

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2,hidden_dim=64,num_layers=10):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1,hidden_dim),
            torch.nn.ReLU(),
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()
                )
                for _ in range(num_layers - 1)
            ],
            torch.nn.Linear(hidden_dim,hidden_dim)
        )


        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_dim + hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()
                )
                for _ in range(num_layers - 1)
            ],
            torch.nn.Linear(hidden_dim, n_dim),
        )

        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.n_steps = n_steps
        self.n_dim = n_dim

        """
        Sets up variables for noise schedule
        """
        self.alphas=None
        self.betas=None
        self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
    
        if not torch.is_tensor(t):
            t = torch.reshape(torch.FloatTensor([t]).expand(x.size(0)), (x.size(0), 1))
        t_embed = self.time_embed(t.float())
        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        step=(ubeta-lbeta)/(self.n_steps)
        self.betas=torch.arange(lbeta,ubeta,step)
        # a = np.power((ubeta/lbeta),(1/self.n_steps))
        # self.betas = torch.pow(a,torch.arange(0,self.n_steps))* lbeta

        # self.betas = torch.sin(((torch.arange(0,self.n_steps))*torch.pi/(2*self.n_steps)))* (ubeta-lbeta) + lbeta
        self.alphas=torch.cumprod(1-self.betas,dim=0)


    def q_sample(self, x, t,epsilon):
        """
        Sample from q given x_t.
        """
        # We have n_samples= Number of samples in one batch
        alpha=torch.reshape(self.alphas[torch.flatten((t-1).type(torch.int64))],(x.shape[0],1))
        x_t=torch.sqrt(alpha)*x+ torch.sqrt(1-alpha)*epsilon
        return x_t

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        """
        x_t=x
        print(x.shape)
        n_samples = x.shape[0]
        Mid_Values = torch.zeros(self.n_steps, n_samples, self.n_dim)
        output = torch.zeros(n_samples, self.n_dim)
        Mid_Values[0] = x_t
        for j in range(t):
            alpha1 = (1/torch.sqrt(1-self.betas[t - j - 1]))
            alpha1_ = (self.betas[t - j - 1])/(torch.sqrt(1 - self.alphas[t - j - 1]))

            x_t = alpha1 * (x_t - alpha1_*self.forward(x_t, float(t - j - 1))) + self.betas[t-j-1]*torch.randn(n_samples, self.n_dim)
            if (j < t - 1):
                Mid_Values[j + 1] = x_t
            else :
                output = x_t
        return output, Mid_Values
        

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        eps=torch.randn(batch.shape)
        t=torch.randint(1,self.n_steps,(batch.shape[0],1))
        q_samples=self.q_sample(batch,t,eps)
        loss=torch.norm(eps-self.forward(q_samples,t))**2
        return loss/batch.shape[0]

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        output=torch.zeros((n_samples,self.n_steps))
        x_t=torch.randn(n_samples,self.n_dim)
        output,intermediate=self.p_sample(x_t,self.n_steps)

        if(return_intermediate):
            return output,intermediate
        else:
            return output

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)
