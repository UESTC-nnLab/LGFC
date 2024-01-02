from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
import matplotlib
matplotlib.use('agg')
from  matplotlib import pyplot as plt
import cv2

class TrainSetLoader(Dataset):

    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png'):
        super(TrainSetLoader, self).__init__()

        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        # self.T_masks = dataset_dir+'/'+'Target_mask'
        # self.T_images = dataset_dir+'/'+'Target_image'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _copy_paste_transform(self,img, mask, CP_num):

        img_path = self.T_images + '/'  # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.T_masks + '/'
        w, h = mask.size
        img_dir = os.listdir(img_path)
        label_dir = os.listdir(label_path)
        range_k = len(img_dir) #range_k表示图像数量
        dice = random.randint(0, 1) #生成一个随机数 dice，用于决定是否执行复制粘贴操作。

        if dice==0:
            img=img
            mask=mask
        else:
            for i in range(CP_num):
                k = random.randint(0,range_k-1 ) #随机选择一个图像
                x = random.randint(0, w-1)   #随机生成x,y位置
                y = random.randint(0, h-1)
                T_I_path = img_path + img_dir[k]
                T_M_path = label_path + label_dir[k]
                T_img = Image.open(T_I_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
                T_mask = Image.open(T_M_path)
                img.paste(T_img, (x, y))
                mask.paste(T_mask, (x, y))

        return img, mask

    
    def _sync_transform(self, img, mask):
        # random mirror
        # img  = img.resize ((self.base_size, self.base_size), Image.BILINEAR)
        # mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        # img,mask = self._copy_paste_transform(img,mask,random.randint(1,5))

        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask


    def __getitem__(self, idx):

        img_id     = self._items[idx]                        # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix   # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix

        img = Image.open(img_path).convert('RGB')         ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        # img = Image.open(img_path)
        mask = Image.open(label_path)

        # synchronized transform
        img, mask = self._sync_transform(img, mask)
        # img, mask = self._copy_paste_transform(img, mask)
        # general resize, normalize and toTensor
        if self.transform is not None: 
            img = self.transform(img)
       
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0


        return img, torch.from_numpy(mask) #img_id[-1]

    def __len__(self):
        return len(self._items)



class TestSetLoader(Dataset):
    NUM_CLASS = 1
    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        # img  = img.resize ((base_size, base_size), Image.BILINEAR)
        # mask = mask.resize((base_size, base_size), Image.NEAREST)
        img  = img.resize ((self.crop_size, self.crop_size), Image.BILINEAR)
        mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)
        ##将图像和掩码转换为NumPy数组
        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix
        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        # img  = Image.open(img_path)
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0


        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)

class DemoLoader (Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(DemoLoader, self).__init__()
        self.transform = transform
        self.images    = dataset_dir
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _demo_sync_transform(self, img):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)

        # final transform
        img = np.array(img)
        return img

    def img_preprocess(self):
        img_path   =  self.images
        img  = Image.open(img_path).convert('RGB')

        # synchronized transform
        img  = self._demo_sync_transform(img)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img

def get_img_norm_cfg(dataset_name, dataset_dir):
    if  dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=62.10432052612305, std=23.96998405456543)
    elif dataset_name == 'IRDST-real':   
        img_norm_cfg = {'mean': 101.54053497314453, 'std': 56.49856185913086}
    else:
        with open(dataset_dir + '/' + dataset_name +'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir + '/' + dataset_name +'/img_idx/test_' + dataset_name + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    return img_norm_cfg

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and hasattr(m, 'weight'):
        # init.kaiming_normal_(m.weight.data,a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.xavier_normal(m.weight.data)

def Normalized(img, img_norm_cfg):
    return (img-img_norm_cfg['mean'])/img_norm_cfg['std']

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))

def save_train_log(args, save_dir):
    dict_args=vars(args)
    args_key=list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('ICME/result/%s/train_log.txt'%save_dir ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return

def save_model_and_result(dt_string, epoch,train_loss, test_loss, best_iou, recall, False_alarm,nIoU, save_mIoU_dir, save_other_metric_dir,pixAcc,F1,AUC,recall2,precision):

    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}:  - train_loss: {:04f}: - test_loss: {:04f}:  mIoU {:.4f}- pixACC: {:04f}: AUC: {:04f}  F1: {:04f}:  recall: {:04f}:  precision: {:04f}:\n '.format(dt_string, epoch,train_loss, test_loss, best_iou,pixAcc,AUC,F1,recall2,precision))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('False_alarm:')
        for i in range(len(False_alarm)):
            f.write('   ')
            f.write(str(round(False_alarm[i], 8)))
            f.write('   ')
        f.write('\n')
        f.write('nIoU:')
        for i in range(len(nIoU)):
            f.write('   ')
            f.write(str(round(nIoU[i], 8)))
            f.write('   ')
        f.write('\n')

def save_model(mean_IOU, best_iou, save_dir, save_prefix, train_loss, test_loss, recall,False_alarm, Final_nIoU, epoch, net,pixAcc,F1,AUC, recall2, precision):

        save_mIoU_dir = 'ICME/result/' + save_dir + '/' + save_prefix + '_best_IoU_IoU.log'
        save_other_metric_dir = 'ICME/result/' + save_dir + '/' + save_prefix + '_best_IoU_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        if mean_IOU > best_iou:
            best_iou = mean_IOU

        # save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou,
        #                       recall, precision, save_mIoU_dir, save_other_metric_dir)
            save_ckpt({
                'epoch': epoch,
                'state_dict': net,
                'loss': test_loss,
                'mean_IOU': best_iou,
                'n_IOU':Final_nIoU,
                'pixAcc':pixAcc
                }, save_path='ICME/result/' + save_dir,
                filename='bestmIoU_' + '_' + save_prefix + '_epoch' + '.pth.tar')

        save_model_and_result(dt_string, epoch, train_loss, test_loss, mean_IOU,
                              recall, False_alarm,Final_nIoU , save_mIoU_dir, save_other_metric_dir,pixAcc,F1,AUC, recall2, precision)

        # save_ckpt({
        #     'epoch': epoch,
        #     'state_dict': net,
        #     'loss': test_loss,
        #     'mean_IOU': mean_IOU,
        #     }, save_path='ICME/result/' + save_dir,
        #     filename='mIoU_' + '_' + save_prefix + '_epoch' + '.pth.tar')

def save_result_for_test(dataset_dir, st_model, epochs, best_iou, recall, precision ):
    with open(dataset_dir + '/' + 'value_result'+'/' + st_model +'_best_IoU.log', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epochs, best_iou))

    with open(dataset_dir + '/' +'value_result'+'/'+ st_model + '_best_other_metric.log', 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
    return

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def make_dir(dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    
    save_dir = "%s_%s_%s_wDS" % (dataset, model, dt_string)
    
    # save_dir = "%s_%s_%s_woDS" % (dataset, model, dt_string)
    os.makedirs('ICME/result/%s' % save_dir, exist_ok=True)
    return save_dir

def total_visulization_generation(dataset_dir, mode, test_txt, suffix, target_image_path, target_dir):
    source_image_path = dataset_dir + '/images'

    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')




def make_visulization_dir(target_image_path, target_dir):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_dir)

def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

def save_loss_Input(map, source_image_path,target_image_path, val_img_ids, suffix):

    predss=map/2
    predsss = np.array(predss.cpu() * 255).astype('int64')
    predsss = np.uint8(predsss)
    predsss = predsss[0,0,:,:]


    input_img = Image.open(source_image_path + '\\' + val_img_ids + suffix)
    input_img = input_img.resize((256, 256), Image.ANTIALIAS)
    predsss = Image.fromarray(predsss.reshape(256, 256))
    pred_img = predsss
    fig = plt.figure(figsize=(6, 6), dpi=300)
    fig1=plt.subplot(1,1,1)
    # fig1.imshow(input_img, alpha=1)
    # fig1.axis('off')
    fig1.imshow(pred_img, alpha=1, interpolation='nearest', cmap="jet")
    fig1.axis('off')

    fig.savefig(target_image_path + '/' + '%s_Pred_input' % (val_img_ids) +suffix, facecolor='none', edgecolor='none')
    fig.clf()
    plt.close()


##用于保存和可视化预测结果和原始图像。
def save_Pred_GT_visulize(pred, img_demo_dir, img_demo_index, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(img_demo_dir + '/testresults/' + '%s_Pred' % (img_demo_index) +suffix)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    img = Image.open(img_demo_dir + '/images/' + img_demo_index + suffix)
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = plt.imread(img_demo_dir + '/images/' + img_demo_index + suffix)
    
    plt.imshow(img, cmap='gray')
    plt.xlabel("Raw Imamge", size=11)

    plt.subplot(1, 3, 2)
    img = plt.imread(img_demo_dir +'/testresults/' + '%s_Pred' % (img_demo_index) +suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Predicts", size=11)

    plt.subplot(1, 3, 3)
    img = Image.open(img_demo_dir + '/masks/' + img_demo_index + suffix)
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = plt.imread(img_demo_dir + '/masks/' + img_demo_index + suffix)
    
    plt.imshow(img, cmap='gray')
    plt.xlabel("Mask", size=11)

    plt.savefig(img_demo_dir + '/testresults/' + img_demo_index + "_fuse" + suffix, facecolor='w', edgecolor='red')
    plt.show()


##用于保存和可视化预测结果和标签。
def save_and_visulize_demo(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

    return

##kaiming初始化
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params 
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
