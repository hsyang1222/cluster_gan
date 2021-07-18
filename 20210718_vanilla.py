import generative_model_score
score_model = generative_model_score.GenerativeModelScore()
score_model.lazy_mode(True)
import dataset
import TorchKmeans
train_loader = dataset.get_cifar1_dataset(2048, 32, shuffle=False)
data_cl_loader, numdata_in_cl = TorchKmeans.clusterset_with_gm_load_or_make(score_model, train_loader)
#score_model.load_or_make(train_loader)

import torch

device = 'cuda:2'
n_epoch = 1e+6

import torch.nn as nn
ndf = 64
ngf = 64
nz = 100
nc = 3
ngpu = 1

class Assigner(nn.Module):
    def __init__(self, ngpu, K):
        super(Assigner, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            #  -- original code --
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            # state size. 1x1x1
            #nn.Sigmoid()
            nn.Linear(2048, K)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

assign = Assigner(ngpu, 100).to(device)
assign.load_state_dict(torch.load('cifar10_32_assin.model'))
'''
criterion = torch.nn.CrossEntropyLoss()
assign_optim = torch.optim.Adam(assign.parameters(), lr=1e-4)

epoch = 0
while epoch < n_epoch : 
    
    
    total_hit = 0
    for i, (data, cluster) in enumerate(data_cl_loader) :         
        batch_size = data.size(0)
        image_cuda = data.to(device)
        cluster_cuda = cluster.to(device)
        predict_cluster = assign(image_cuda).view(batch_size, -1)
        loss = criterion(predict_cluster, cluster_cuda)
        hit = torch.argmax(predict_cluster, dim=1) == cluster_cuda
        total_hit += torch.sum(hit).item()
        
        #if i >= 20 : break
        
        assign_optim.zero_grad()
        loss.backward()
        assign_optim.step()
        
    #acc = torch.sum(hit).float()/len(hit)
    acc = total_hit / len(data_cl_loader.dataset)
    print(epoch, acc, loss.item())
    epoch += 1
    
    if acc > 0.99 : break
 '''

class ClusterDiscriminator(nn.Module):
    def __init__(self, n_class, n_hidden) : 
        self.n_class = n_class
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_class, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,1),
            nn.Sigmoid()
        )
    
    def forward(self, x) :
        x = self.net(x)
        return x
    
cd_net = ClusterDiscriminator(2048, 64).to(device)
criterion_bce= torch.nn.BCELoss()
cd_optim = torch.optim.Adam(cd_net.parameters(), lr=1e-4)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            #nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.ConvTranspose2d(     nz, ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
g_optim = torch.optim.Adam(netG.parameters(), lr=1e-4)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            # state size. 1x1x1
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

    
netD = Discriminator(ngpu).to(device)
d_optim = torch.optim.Adam(netD.parameters(), lr=1e-4)

fixed_z = torch.randn(8*8, 100, 1, 1, device=device)
import torchvision.utils as vutils
def get_fixed_z_image_np(netG) : 
    netG.eval()
    fake_img = netG(fixed_z)
    fake_np = vutils.make_grid(fake_img.detach().cpu(), nrow=8).permute(1,2,0).numpy()
    netG.train()
    return fake_np

epoch = 0

import wandb
wandb.init(project='cluster_gan', name='vanilla')

assign.eval()
while epoch < 1e+6 : 
    real_assign_list = []
    fake_assign_list = []
    for i, (data, cluster) in enumerate(data_cl_loader) :   
        batch_size = data.size(0)
        
        if batch_size != 2048 : continue
        
        image_cuda = data.to(device)
        
        ### train D by image
        # real image
        d_predict_real_image = netD(image_cuda)
        label_one = torch.ones(batch_size, 1, device=device)
        loss_d_real = criterion_bce(d_predict_real_image, label_one)
        
        # fake image
        fake_latent = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_image = netG(fake_latent)
        d_predict_fake_image = netD(fake_image)
        label_zero = torch.zeros(batch_size, 1, device=device)
        loss_d_fake = criterion_bce(d_predict_fake_image, label_zero)
        
        # backward and step
        loss_d = loss_d_real + loss_d_fake
        netD.zero_grad()
        loss_d.backward()
        d_optim.step()
        
        
        ### train G by image
        # fake image
        fake_image = netG(fake_latent)
        d_predict_fake_image = netD(fake_image)
        loss_g_fake = criterion_bce(d_predict_fake_image, label_one)
        
        # backward and step
        loss_g = loss_g_fake
        netG.zero_grad()
        loss_g.backward()
        g_optim.step()
        
        
        ### train CD by assign cluster
        # real cluster
        real_assign = assign(image_cuda).view(batch_size, -1).detach()
        cd_predict_real_assign = cd_net(real_assign.T)
        label_one = torch.ones(100, 1, device=device)
        loss_cd_real = criterion_bce(cd_predict_real_assign, label_one)
        
        # fake cluster
        fake_assign = assign(fake_image).view(batch_size, -1).detach()
        cd_predict_fake_assign = cd_net(fake_assign.T)
        label_zero = torch.zeros(100, 1, device=device)
        loss_cd_fake = criterion_bce(cd_predict_fake_assign, label_zero)
        
        # backward and step
        loss_cd = loss_cd_real + loss_cd_fake
        cd_net.zero_grad()
        loss_cd.backward()
        #cd_optim.step()
        
        
        ### train G by assign cluster
        # fake image
        fake_image = netG(fake_latent)
        fake_assign = assign(fake_image).view(batch_size, -1)
        cd_predict_fake_assign = cd_net(fake_assign.T)
        loss_gcd_fake = criterion_bce(cd_predict_fake_assign, label_one)
        
        #backward and step
        loss_gcd = loss_gcd_fake
        netG.zero_grad()
        loss_gcd.backward()
        #g_optim.step()
       
        real_assign_list.append(real_assign.detach().cpu())
        fake_assign_list.append(fake_assign.detach().cpu())
        
   
    epoch += 1
    print(epoch, loss_g.item(), loss_d.item(), loss_cd.item(), loss_gcd.item())
    #plt_cluster(real_assign, fake_assign)
    real_argmax = torch.argmax(torch.cat(real_assign_list), dim=1).float()
    fake_argmax = torch.argmax(torch.cat(fake_assign_list), dim=1).float()
    wandb.log({
            'step' : epoch,
            'real_assign' : real_argmax,
            'real_assign_mean' : torch.mean(real_argmax),
            'real_assign_std' : torch.std(real_argmax),
            'fake_assign' : fake_argmax,
            'fake_assign_mean' : torch.mean(fake_argmax),
            'fake_assign_std' : torch.std(fake_argmax),  
            'fake_img' : [wandb.Image(get_fixed_z_image_np(netG), caption='fixed z image')]
          })