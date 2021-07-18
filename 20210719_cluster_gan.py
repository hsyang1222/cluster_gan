import generative_model_score
score_model = generative_model_score.GenerativeModelScore()
score_model.lazy_mode(True)
import dataset
import TorchKmeans
train_loader = dataset.get_cifar1_dataset(2048, 32, shuffle=False)
data_cl_loader, numdata_in_cl = TorchKmeans.clusterset_with_gm_load_or_make(score_model, train_loader)
#score_model.load_or_make(train_loader)

import torch



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
  
     
def inference_matric(score_model, train_model_list, device) :
    for train_model in train_model_list : 
        train_model.to('cpu')
    score_model.model_to(device)
    score_model.lazy_forward(batch_size=64, device=device, fake_forward=True)
    score_model.calculate_fake_image_statistics()
    matrics = score_model.calculate_generative_score()
    score_model.model_to('cpu')
    for train_model in train_model_list : 
        train_model.to(device)
    score_model.clear_fake()
    return matrics

fixed_z = None
import torchvision.utils as vutils
def get_fixed_z_image_np(netG) : 
    global fixed_z
    netG.eval()
    fake_img = netG(fixed_z)
    fake_np = vutils.make_grid(fake_img.detach().cpu(), nrow=8).permute(1,2,0).numpy()
    netG.train()
    return fake_np  
    
def main(args) : 
    device = args.device
    model_name = args.model_name
    inference_interval = args.inference_interval
    n_epoch = 1e+6

    
    assign = Assigner(ngpu, 100).to(device)
    assign.load_state_dict(torch.load('cifar10_32_assin.model'))
    
    netG = Generator(ngpu).to(device)
    g_optim = torch.optim.Adam(netG.parameters(), lr=1e-4)
    
    netD = Discriminator(ngpu).to(device)
    d_optim = torch.optim.Adam(netD.parameters(), lr=1e-4)
    
    cd_net = ClusterDiscriminator(2048, 64).to(device)
    criterion_bce= torch.nn.BCELoss()
    cd_optim = torch.optim.Adam(cd_net.parameters(), lr=1e-4)

    epoch = 0
    
    global fixed_z
    fixed_z = torch.randn(8*8, 100, 1, 1, device=device)




    import wandb
    wandb.init(project='cluster_gan', name=model_name)

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
            label_one = torch.ones(nz, 1, device=device)
            loss_cd_real = criterion_bce(cd_predict_real_assign, label_one)

            # fake cluster
            fake_assign = assign(fake_image).view(batch_size, -1).detach()
            cd_predict_fake_assign = cd_net(fake_assign.T)
            label_zero = torch.zeros(nz, 1, device=device)
            loss_cd_fake = criterion_bce(cd_predict_fake_assign, label_zero)

            # backward and step
            loss_cd = loss_cd_real + loss_cd_fake
            cd_net.zero_grad()
            if model_name == 'cluster_gan' : 
                loss_cd.backward()
                cd_optim.step()


            ### train G by assign cluster
            # fake image
            fake_image = netG(fake_latent)
            fake_assign = assign(fake_image).view(batch_size, -1)
            cd_predict_fake_assign = cd_net(fake_assign.T)
            loss_gcd_fake = criterion_bce(cd_predict_fake_assign, label_one)

            #backward and step
            loss_gcd = loss_gcd_fake
            netG.zero_grad()
            if model_name == 'cluster_gan' : 
                loss_gcd.backward()
                g_optim.step()

            real_assign_list.append(real_assign.detach().cpu())
            fake_assign_list.append(fake_assign.detach().cpu())
            
            
            if epoch % inference_interval == 0 : 
                score_model.put_fake(fake_image.detach().cpu())
            
        if epoch % inference_interval == 0 :   
            matric = inference_matric(score_model, [netG, netD, cd_net], device)

            #plt_cluster(real_assign, fake_assign)
            real_argmax = torch.argmax(torch.cat(real_assign_list), dim=1).float()
            fake_argmax = torch.argmax(torch.cat(fake_assign_list), dim=1).float()
            add_log = {
                    'step' : epoch,
                    'real_assign' : real_argmax,
                    'real_assign_mean' : torch.mean(real_argmax),
                    'real_assign_std' : torch.std(real_argmax),
                    'fake_assign' : fake_argmax,
                    'fake_assign_mean' : torch.mean(fake_argmax),
                    'fake_assign_std' : torch.std(fake_argmax),  
                    'fake_img' : [wandb.Image(get_fixed_z_image_np(netG), caption='fixed z image')]
                  }
            matric.update(add_log)
            wandb.log(matric)
    
        epoch += 1
        print(epoch, loss_g.item(), loss_d.item(), loss_cd.item(), loss_gcd.item())

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    #parser.add_argument('--batch_size', type=int, default=64)
    #parser.add_argument('--img_size', type=int, default=32)
    #parser.add_argument('--save_image_interval', type=int, default=5)
    #parser.add_argument('--loss_calculation_interval', type=int, default=5)
    #parser.add_argument('--latent_dim', type=int, default=10)
    #parser.add_argument('--n_iter', type=int, default=3)
    #parser.add_argument('--project_name', type=str, default='AAE_exact')
    #parser.add_argument('--dataset', type=str, default='', choices=['LSUN_dining_room', 'LSUN_classroom', 'LSUN_conference', 'LSUN_churches','FFHQ', 'CelebA', 'cifar10', 'mnist', 'mnist_fashion', 'emnist'])

    parser.add_argument('--model_name', type=str, default='', choices=['vanilla', 'cluster_gan'])
    parser.add_argument('--inference_interval', type=int, default=10)
    #parser.add_argument('--lr', type=float, default=1e-4)
    #parser.add_argument('--run_test', type=bool, default=False)
    #parser.add_argument('--latent_layer', type=int, default=0)

    args = parser.parse_args()

    main(args)