from tqdm             import tqdm
from torchvision      import transforms
from torch.utils.data import DataLoader
import time
from util.utils import *
from model.loss import *
from util.load_param_data import  load_dataset, load_param
from util.metric import *
from model.net import  *
######获取模型中间特征的钩子，便于生成heatmap
# layer_outputs = []
# def for_hook(module, input, output):
#             layer_outputs.append(output)
# def register_hook(model, func, hook_layers):
#             for name, layer in model.named_modules():
#                 # print(f"name: {name}")
#                 if name in hook_layers:
#             # print(f"register_forward_hook successfully: {name}")
#                     layer.register_forward_hook(func)

class Trainer(object):
    def __init__(self, args):
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.mIoU = MIoU()

        # Initial
        self.args  = args
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        dataset_dir = args.root + '/' + args.dataset
        train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)
        
        model       = MViTLD(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks,nb_filter=nb_filter)
        
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        checkpoint             = torch.load(args.model_dir)
       
        self.model.load_state_dict(checkpoint['state_dict'])

        ####钩子层
        # hook_layers = ["stem","conv1_0","conv2_0","conv3_0"]
        # register_hook(self.model, for_hook, hook_layers)

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        with torch.no_grad():
            num = 0
            t_all=[]
            self.metric.reset()
            self.mIoU.reset()
            for i, ( data, labels) in enumerate(tbar):
                start = time.time()
                data = data.cuda()
                labels = labels.cuda()
                preds = self.model(data)
                pred =preds[0]

                ####每一级解码的可视化
                # for pred in preds:
                #     num = num + 1
                #     predsss = np.array((pred > 0).cpu()).astype('int64') * 255
                #     predsss = np.uint8(predsss)
                #     img = Image.fromarray(predsss.reshape(256, 256))
                #     img.save('XDU45-{}_Pred.png'.format(num))
                       
        #####特征热力图    
            # i = 0
            # for fea in layer_outputs:
            #     i = i+1
            #     features = fea
            #     heatmap = torch.sum(features, dim=1)  # 尺度大小， 如torch.Size([1,45,45])
            #     max_value = torch.max(heatmap)
            #     min_value = torch.min(heatmap)
            #     heatmap = (heatmap-min_value)/(max_value-min_value)*255
            #     heatmap = heatmap.cpu().detach().numpy().astype(np.uint8).transpose(1,2,0)  # 尺寸大小，如：(45, 45, 1)
            #     src_size = (256,256)  # 原图尺寸大小
            #     heatmap = cv2.resize(heatmap, src_size,interpolation=cv2.INTER_LINEAR)  # 重整图片到原尺寸
            #     heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
            #     cv2.imwrite('Misc_53heat-{}.jpg'.format(i), heatmap)
        ###metric指标
                self.metric.update(labels=labels, preds=pred)
                # self.ROC.update(pred,labels)
                self.mIoU.update(pred, labels)
                # self.PD_FA.update(pred, labels)
                
                # end = time.time()
                # t_all.append(end-start)

            _, precision, recall, F1 = self.metric.get()
            # AUC = auc(fp_rates,tp_rates)
            pixAcc, mean_IOU = self.mIoU.get()
            print("'iou: {:04f}:  best_recall: {:04f}:  best_prec: {}  best_f1: {:04f}".format(mean_IOU,recall,precision,F1))

        #######fps
            # print('average time:', np.mean(t_all) / 1)
            # print('average fps:',1 / np.mean(t_all))
            # print('fastest time:', min(t_all) / 1)
            # print('fastest fps:',1 / min(t_all))
            # print('slowest time:', max(t_all) / 1)
            # print('slowest fps:',1 / max(t_all))

def parse_args():

    parser = argparse.ArgumentParser(description='MViT-LD')
   
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--dataset', type=str, default='IRSTD',
                        help='dataset name: NUAA-SIRST,IRSTD-1K')
    parser.add_argument('--model_dir', type=str,
                        default = '/home/dww/OD/ICME/bestmIoU__MTU_Net_IRSTD-1k_0.6776mtu11.pth.tar')
    parser.add_argument('--root', type=str, default='/home/dww/OD/dataset')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='img_idx',
                        help='the split of dataset')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPU.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)





