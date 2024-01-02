from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from util.utils import *
from util.metric import *
from model.loss import *
from util.load_param_data import  load_dataset, load_param
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from model.net import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def seed_pytorch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# seed_pytorch(3407)

class Trainer(object):
    def __init__(self, args):
        # Initial 
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.PD_FA = PD_FA(1, 1)
        self.mIoU = MIoU()
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone) #[16,32,64,128,256] [2,2,2,2]

        model       = MViTLD(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks,nb_filter=nb_filter)
        model           = model.cuda()
        model.apply(weights_init_xavier)

        #####多卡并行训练
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model, device_ids=[0,1])
        # else:
        #     model = model.cuda()  bestmIoU__MTU_Net_mdfa_epoch.pth.tar   bestmIoU__MTU_Net_NUAA-SIRST_epoch.pth.tar bestmIoU__MTU_Net_sirst_aug_epoch.pth.tar

        #####checkpoint加载模型
        # checkpoint = torch.load('path of model')  
        # model.load_state_dict(checkpoint['state_dict'],strict=True)

        print("Model Initializing")
        self.model      = model
    
        # state = self.model.state_dict()
        # dict_name = list(state)
        # for i, p in enumerate(dict_name):
        #     print(i, p)

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=64, eta_min=args.min_lr)
        elif args.scheduler == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200,500,800,1200], gamma=0.1)

    
        self.best_iou       = 0
        self.best_recall    = 0
        self.best_f = 0
        self.best_auc = 0
        self.best_niou = 0
        self.best_pixAcc = 0
        self.best_pre = 0

        ###plot
        self.train_losses = []
        self.test_losses = [] 
        self.tpr = 0
        self.fpr = 0

    def Data_Augmentation(self,args):
        input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        dataset_dir = args.root + '/' + args.dataset
        train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True,pin_memory=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False,pin_memory=True)
        
    # Training
    def training(self,epoch):
        tbar = tqdm(self.train_data)
        self.model.train()
        
        losses = AverageMeter()
        for i, ( data, labels) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()
            preds= self.model(data)
            loss = 0
            for pred in preds:
                loss += SoftIoULoss(pred, labels)
            loss /= len(preds)
            pred =preds[0]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, lr %.8f, training loss %.4f' % (epoch, self.optimizer.param_groups[0]['lr'] ,losses.avg))
        self.train_loss = losses.avg
        
        if (epoch+1) % args.save_iter == 0:
            self.train_losses.append(self.train_loss)

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        self.ROC.reset()
        self.metric.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                
                #level-wise decoding
                preds= self.model(data)
                loss = 0
                for pred in preds:
                  loss += SoftIoULoss(pred, labels)
                loss /= len(preds)
                pred =preds[0]
                
                losses.update(loss.item(), pred.size(0))
                self.metric.update(labels=labels, preds=pred)
                self.ROC.update(pred,labels)
                self.mIoU.update(pred, labels)
                self.PD_FA.update(pred, labels)

                pixAcc, mean_IOU = self.mIoU.get()
                
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU))

            test_loss = losses.avg

            Final_FA, Final_PD,Final_nIoU = self.PD_FA.get(214) #545 for sirst_aug, 847 for NUDT-SIRST-Sea 100 for mdfa 214 for NUAA 201 for irst1k
            tp_rates, fp_rates = self.ROC.get()

            _, precision, recall, F1 = self.metric.get()
            AUC = auc(fp_rates,tp_rates)
            pixAcc, mean_IOU = self.mIoU.get()

            save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                            self.train_loss, test_loss, Final_PD, Final_FA,  Final_nIoU, 
                            epoch,self.model.state_dict(),
                            pixAcc, F1, AUC,recall,precision)
            if mean_IOU >self.best_iou:
                self.best_iou = mean_IOU
                self.tpr = tp_rates
                self.fpr = fp_rates
            if recall > self.best_recall:
                self.best_recall = recall
            if F1 > self.best_f:
                self.best_f = F1
            if AUC > self.best_auc:
                self.best_auc = AUC
            if Final_nIoU.max() > self.best_niou:
                self.best_niou = Final_nIoU.max()
            if pixAcc > self.best_pixAcc:
                self.best_pixAcc = pixAcc
            if precision >self.best_pre:
                self.best_pre = precision
            
            self.PD_FA.reset()
            self.ROC.reset()
            self.mIoU.reset()
            self.metric.reset()
        
        if (epoch+1) % args.save_iter == 0:
            self.test_losses.append(test_loss)
            

    #print the best
    def print_best(self):
        print('best_iou: {:04f}:  best_recall: {:04f}:  best_prec: {}  best_f1: {:04f}  best_auc: {:04f} best_niou: {:04f}  best_pixAcc: {:04f}\n '.format(self.best_iou, self.best_recall, self.best_pre, self.best_f, self.best_auc, self.best_niou,self.best_pixAcc))
        file = open('./res.txt','w')
        file.write('tpr'+'\n')
        for i in range(len(self.tpr)):
            file.write('   ')
            file.write(str(round(self.tpr[i], 6)))
            file.write('   ')
        file.write('\n')
        file.write('fpr'+'\n')
        for i in range(len(self.fpr)):
            file.write('   ')
            file.write(str(round(self.fpr[i], 6)))
            file.write('   ')
        file.write('\n')
        file.close()
    
    def plot_loss(self,epoch):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(range(int(epoch/args.save_iter)), self.train_losses)
        # ax.plot(range(int(epoch/args.save_iter)), self.test_losses)
        ax.plot(range(int(epoch)), self.train_losses)
        ax.plot(range(int(epoch)), self.test_losses)
        # ax.plot(range(epoch+1), self.ious)
        # ax.plot(range(epoch+1), self.nious)
        ax.legend(["Train loss","Val loss"])
        ax.set_xlabel("Epoch")
        fig.savefig("./losses.png")
        plt.close('all')


def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.Data_Augmentation(args)
        trainer.training(epoch)
        if (epoch+1) % args.save_iter == 0:
            trainer.testing(epoch)
    trainer.print_best()
    trainer.plot_loss(args.epochs)

def parse_args():

    parser = argparse.ArgumentParser(description='MViT')

    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--dataset', type=str, default='IRSTD-1k',
                        help='dataset name: IRSTD-1K, NUAA-SIRST')
    parser.add_argument('--root', type=str, default='/home/dww/OD/dataset')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='1_1',
                        help='idx_4961+847 for NUDT-SIRST-Sea')
    parser.add_argument('--workers', type=int, default=6,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')
    
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 110)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 32)') 
    parser.add_argument('--min_lr', default=5e-6,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help=' Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR'])
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    
    parser.add_argument('--save_iter', type=int, default='1',
                        help='save_iter.')

    args = parser.parse_args()

    args.save_dir = make_dir(args.dataset, args.model)

    save_train_log(args, args.save_dir)

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)





