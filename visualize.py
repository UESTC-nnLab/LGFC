from tqdm             import tqdm
from torchvision      import transforms
from torch.utils.data import DataLoader
import time
from util.utils import *
from model.loss import *
from util.load_param_data import  load_dataset, load_param
from model.net import  *


class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
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

        # Checkpoint
        checkpoint             = torch.load(args.model_dir)
        visulization_path      = dataset_dir + '/test' 
        visulization_fuse_path = dataset_dir + '/test' 

        make_visulization_dir(visulization_path, visulization_fuse_path)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        with torch.no_grad():
            num = 0
            t_all=[]
            for i, ( data, labels) in enumerate(tbar):
                start = time.time()
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    pred =preds[5]
                else:
                    pred = self.model(data)
                end = time.time()
                t_all.append(end-start)
            
                save_Pred_GT(pred, labels,visulization_path, val_img_ids, num, args.suffix)
                num += 1
            total_visulization_generation(dataset_dir, args.mode, test_txt, args.suffix, visulization_path, visulization_fuse_path)


def main(args):
    trainer = Trainer(args)

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
    main(args)





