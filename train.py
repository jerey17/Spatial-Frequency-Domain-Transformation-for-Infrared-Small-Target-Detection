import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.ffnet import *
from dataset import *
import matplotlib.pyplot as plt
from metric import *
import numpy as np
import os
from loss import *
#from freqLoss import *
from utils import *
# from model.FTNet import *
# from model.ffnet2 import FFNet2
# from model.FDTNet import *
# from model.ffnet_fuse import *
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['SFDTNet'], nargs='+',
                     )
parser.add_argument("--dataset_names", default=['NUDT-SIRST'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--every_test", default=1, type=int)
parser.add_argument("--every_print", default=10, type=int)
parser.add_argument("--every_save_pth", default=100, type=int)
parser.add_argument("--dataset_dir", default='./dataset', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--baseSize", type=int, default=256, help="Test size")
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, nargs='+', help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--pretrained", default=None, nargs='+', help="Load pretrained checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=800, help="Number of epochs")
parser.add_argument("--begin_test", default=500, type=int)
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict, help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")

global opt
opt = parser.parse_args()

if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

seed_pytorch(opt.seed)


def train():
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    
    net = Net(model_name=opt.model_name, mode='train').cuda()

    net.train()
    
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    best_mIOU = (0, 0)

    if opt.resume:
        for resume_pth in opt.resume:
            if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                for i in range(len(opt.scheduler_settings['step'])):
                    opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']
    if opt.pretrained:
        for pretrained_pth in opt.pretrained:
            if opt.dataset_name in pretrained_pth and opt.model_name in pretrained_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
    
    ### Default settings                
    # if opt.optimizer_name == 'Adam':
    #     opt.optimizer_settings = {'lr': 50e-4}
    #     opt.scheduler_name = 'MultiStepLR'
    #     opt.scheduler_settings = {'epochs':400, 'step': [100, 300], 'gamma': 0.5}
    #     opt.scheduler_settings['epochs'] = opt.nEpochs
    
    ### Default settings of DNANet                
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs':1500, 'min_lr':1e-5}
        opt.scheduler_settings['epochs'] = opt.nEpochs
        
    opt.nEpochs = opt.scheduler_settings['epochs']

    
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings, opt.scheduler_settings)
    
    for idx_epoch in range(epoch_state, opt.nEpochs):
        results1 = (0, 0)
        results2 = (0, 0)
        with tqdm(total=len(train_loader), desc=f'Epoch {idx_epoch + 1}/{opt.nEpochs}') as pbar:
            for idx_iter, (img, gt_mask) in enumerate(train_loader):
                img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()

                if img.shape[0] == 1:
                    continue
                pred = net.forward(img)
                loss = net.loss(pred, gt_mask)
                total_loss_epoch.append(loss.detach().cpu())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

            scheduler.step()
            if (idx_epoch + 1) % opt.every_print == 0:
                total_loss_list.append(float(np.array(total_loss_epoch).mean()))
                print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,'
                      % (idx_epoch + 1, total_loss_list[-1]))
                opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                      % (idx_epoch + 1, total_loss_list[-1]))
                total_loss_epoch = []

      

            if (idx_epoch + 1) >= opt.begin_test and (idx_epoch + 1) % opt.every_test == 0:
                test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name,base_size=opt.baseSize,
                                         img_norm_cfg=opt.img_norm_cfg)
                test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
                net.eval()
                with torch.no_grad():
                    eval_mIoU = mIoU()
                    eval_PD_FA = PD_FA()
                    test_loss = []
                    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
                        img = Variable(img).cuda()
                        pred = net.forward(img)
                        if isinstance(pred, tuple):
                            pred = pred[-1]
                        elif isinstance(pred, list):
                            pred = pred[-1]
                        else:
                            pred = pred
                        pred = pred[:, :, :size[0], :size[1]]
                        gt_mask = gt_mask[:, :, :size[0], :size[1]]
                        # if pred.size() != gt_mask.size():
                        #     print('1111')
                        loss = net.loss(pred, gt_mask.cuda())
                        test_loss.append(loss.detach().cpu())
                        eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
                        eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)
                    test_loss.append(float(np.array(test_loss).mean()))
                    results1 = eval_mIoU.get()
                    results2 = eval_PD_FA.get()

                    print("pixAcc, mIoU:\t" + str(results1))
                    print("PD, FA:\t" + str(results2))
                    
                    
            if (idx_epoch + 1) % opt.every_save_pth == 0:
                save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                    }, save_pth)
                test(save_pth)

            if idx_epoch == 0:
                best_mIOU = results1
                best_Pd = results2

            if results1[1] > best_mIOU[1]:
                best_mIOU = results1
                best_Pd = results2
                print('------save the best model epoch', opt.model_name, '_%d ------' % (idx_epoch + 1))
                opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
                print("pixAcc, mIoU:\t" + str(best_mIOU))
                print("testloss:\t" + str(test_loss[-1]))
                print("PD, FA:\t" + str(best_Pd))

                opt.f.write("pixAcc, mIoU:\t" + str(best_mIOU) + '\n')
                opt.f.write("PD, FA:\t" + str(best_Pd) + '\n')
                save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(
                    idx_epoch + 1) + '_' + 'best' + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, save_pth)

            if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 50 != 0:
                save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                    }, save_pth)
                test(save_pth)

def test(save_pth):
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, base_size=opt.baseSize,img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    with torch.no_grad():
        eval_mIoU = mIoU() 
        eval_PD_FA = PD_FA()
        for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
            img = Variable(img).cuda()
            pred = net.forward(img)
            pred = pred[:,:,:size[0],:size[1]]
            gt_mask = gt_mask[:,:,:size[0],:size[1]]
            eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask.cpu())
            eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)     
        
        results1 = eval_mIoU.get()
        results2 = eval_PD_FA.get()
        print("pixAcc, mIoU:\t" + str(results1))
        print("PD, FA:\t" + str(results2))
        opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
        opt.f.write("PD, FA:\t" + str(results2) + '\n')
        
def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        # ************************************************loss*************************************************#
        #self.cal_loss = nn.BCELoss(size_average=True)
        # self.soft_iou_loss = SoftIoULoss()
        self.cal_loss = SoftIoULoss()
        # self.cal_loss = AdaptiveCombinedLoss()

                
        if model_name == 'FFNet':
            if mode == 'train':
                self.model = SFDTNet(in_channels=1,deep_supervision=True,mode='train')
                
            else:
                self.model  = SFDTNet(in_channels=1,deep_supervision=True,mode='test')

        # if model_name == 'FTNet':
        #     if mode == 'train':
        #         self.model = FTNet(in_channels=1, deep_supervision=True, mode='train')

        #     else:
        #         self.model = FTNet(in_channels=1, deep_supervision=True, mode='test')

        # if model_name == 'FFNet2':
        #     if mode == 'train':
        #         self.model = FFNet2(in_channels=1, deep_supervision=True, mode='train')

        #     else:
        #         self.model = FFNet2(in_channels=1, deep_supervision=True, mode='test')
        # if model_name == 'FDTNet':
        #     if mode == 'train':
        #         self.model = FDTNet(in_channels=1, deep_supervision=True, mode='train')
        #     else:
        #         self.model = FDTNet(in_channels=1, deep_supervision=True, mode='test')
        # if model_name == 'FFNet(fuse)':
        #     if mode == 'train':
        #         self.model = FFNet_fuse(in_channels=1,deep_supervision=True,mode='train')
        #     else:
        #         self.model = FFNet_fuse(in_channels=1,deep_supervision=True,mode='test')

    def forward(self, img):
        return self.model(img)

    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)

        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss


    # def loss(self, preds, gt_masks):
    #     if isinstance(preds, list):
    #         total_loss = 0
    #         for i in range(len(preds)):
    #             pred = preds[i]
    #             gt_mask = gt_masks[i]
    #             #bce_loss = self.cal_loss(pred, gt_mask)
    #             iou_loss = self.soft_iou_loss(pred, gt_mask)
    #             # 合并损失，可以根据需求调整权重
    #             total_loss +=  iou_loss
    #
    #         return total_loss / len(preds)
    #
    #     elif isinstance(preds, tuple):
    #         total_loss = 0
    #         for i in range(len(preds)):
    #             pred = preds[i]
    #             #bce_loss = self.cal_loss(pred, gt_masks)
    #             iou_loss = self.soft_iou_loss(pred, gt_masks)
    #             total_loss +=  iou_loss
    #
    #         return total_loss
    #
    #     else:
    #         #bce_loss = self.cal_loss(preds, gt_masks)
    #         iou_loss = self.soft_iou_loss(preds, gt_masks)
    #         return  iou_loss

if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
