import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_image_batches 

from utils import BASE_DIR, evaluate, set_seed
import pdb
import torch.utils.data as data
from torchvision import transforms
import scipy.io as sio

from PIL import Image

from network import Minmaxcam_VGG, Minmaxcam_mobilenet, Minmaxcam_resnet

# Some constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def default_loader(path):
    return Image.open(path).convert('RGB')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class ImageFilelist(data.Dataset):
    def __init__(self, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        # self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join('./', impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


def changeIntensity(x):

	# im = Image.open('1_tree.jpg')
	im = x.convert('RGB')
	r, g, b = im.split()
	factor_ = np.random.uniform(0.25, 1)
	r = r.point(lambda i: i * factor_)
	g = g.point(lambda i: i * factor_)
	b = b.point(lambda i: i * factor_)

	out = Image.merge('RGB', (r, g, b))

	return out


def get_data_loader_list(input_folder, batch_size, train=True):
    if train is True:
        transform_list = [
                transforms.Resize([256,256]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomCrop(size=[224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ]
    else:
        transform_list = [
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]


    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=16)
    return loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MinMaxCam model")
    parser.add_argument(
        "--logname", type=str, default="vgg", help="Base network to use"
    )

    parser.add_argument(
        "--mode", type=str, default="VGG_MCAM", help="Base network to use"
    )
    parser.add_argument(
        "--name", type=str, default="", help="Base network to use"
    )


    parser.add_argument("--seed", type=int, default=610, help="Seed to use")

    parser.add_argument("--ss", type=int, default=5, help="Seed to use")

    parser.add_argument("--ms_1", type=int, default=50, help="Seed to use")
    parser.add_argument("--ms_2", type=int, default=100, help="Seed to use")

    parser.add_argument("--bs", type=int, default=5, help="Seed to use")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=200, help="Epoch count")
    parser.add_argument("--offset", type=int, default=0, help="Offset count")

    parser.add_argument("--w_loss_com", type=float, default=1., help="Learning rate")
    parser.add_argument("--w_loss_ori", type=float, default=1., help="Learning rate")
    parser.add_argument("--w_loss_img", type=float, default=1., help="Learning rate")

    parser.add_argument("--train", type=bool, default=False, help="train or test")
    parser.add_argument("--test", type=bool, default=False, help="train or test")


    args = parser.parse_args()

    # Set the seed
    set_seed(args.seed)

    # Set the hyperparameters
    LR = args.lr
    WD = args.wd

    EPOCHS = args.epochs
    OFFSET = args.offset



    # pdb.set_trace()
    
    dataset_train = get_data_loader_list('./data/CUB_2011_train.txt',1, train=True)
    dataset_test = get_data_loader_list('./data/CUB_2011_test.txt',1, train=False)
    dataset_val = get_data_loader_list('./data/CUB_val_list_monkey.txt',1, train=False)

    label = sio.loadmat('./data/cub_2011_train_labels.mat')
    label_train = label['train_labels'][0]

    label = sio.loadmat('./data/cub_2011_test_labels.mat')
    label_test = label['test_labels'][0]

    label = sio.loadmat('./data/cub_2011_val_labels.mat')
    label_val = label['val_labels'][0]



    bbox_test = sio.loadmat('./data/bbox_test.mat')
    bbox_test = bbox_test['bbox_test']

    bbox_val = sio.loadmat('./data/bbox_val.mat')
    bbox_val = bbox_val['bbox_val']

    ori_shape = sio.loadmat('./data/ori_shape.mat')
    ori_shape = ori_shape['ori_shape']
 
    ori_shape_val = sio.loadmat('./data/ori_shape_val.mat')
    ori_shape_val = ori_shape_val['ori_shape_val']



    if args.mode == 'Minmaxcam_VGG':
        net = Minmaxcam_VGG(base_net='vgg', set_size=args.ss)

    elif args.mode == 'Minmaxcam_mobilenet':
        net = Minmaxcam_mobilenet(base_net='vgg', set_size=args.ss)
    elif args.mode == 'Minmaxcam_resnet':
        net = Minmaxcam_resnet(base_net='vgg', set_size=args.ss)


    
    net = net.cuda()



    if args.mode == 'Minmaxcam_resnet':

        optimizer = optim.SGD([
            {'params': net.base.parameters(),'lr':LR},
            {'params': net.pred.parameters(),'lr':LR*10},
            ], lr=LR, weight_decay=WD, momentum=0.9, nesterov=True)




    elif args.mode == 'Minmaxcam_mobilenet':
        optimizer = optim.SGD([
            {'params': net.features.parameters(),'lr':LR*1},
            {'params': net.pred.parameters(),'lr':LR*10},
            ], lr=LR, weight_decay=WD, momentum=0.9, nesterov=True)

    elif args.mode == 'Minmaxcam_VGG':

        optimizer = optim.SGD([
            {'params': net.features.parameters(),'lr':LR},
            {'params': net.extra_conv.parameters(), 'lr':LR*10},
            {'params': net.pred.parameters(),'lr':LR*10}
            ], lr=LR, weight_decay=WD, momentum=0.9, nesterov=True)





    scheduler = MultiStepLR(optimizer, milestones=[args.ms_1, args.ms_2], gamma=0.1)
    scheduler.last_epoch = OFFSET




    if OFFSET != 0:
        state_path = os.path.join('./checkpoint', 'minmaxcam_{}_{}_epoch_best.pt'.format(args.mode, args.name, OFFSET))
    

        print(state_path)

        net.load_state_dict(torch.load(state_path))

        tqdm.write(f"Loaded epoch {OFFSET}'s state.")


        scheduler.step()



    # # Train the model

    val_list = np.array([])
 




    if args.train:


        for epoch in tqdm(range(OFFSET + 1, EPOCHS + 1), "Total"):

            for param_group in optimizer.param_groups:
                tqdm.write('{}---LR is {}'.format(epoch, param_group['lr']))

            epoch_loss = 0.0

            # pdb.set_trace()
            loss_common_ = 0
            loss_ori_ = 0
            loss_set_ = 0
            loss_img_ = 0
            loss_all_ = 0



            for it in tqdm(range(int(len(dataset_train)/args.ss/args.bs))):


                it += 1

                optimizer.zero_grad()

                # pdb.set_trace()
                batches, c_label, selected_idx = get_image_batches(dataset_train, label_train, args.ss, args.bs, 200, epoch*1000 + it)
                batches = batches.cuda()
                c_label = torch.tensor([c_label]).cuda()


                net.train()
                loss_img = net.update_classification(batches, c_label,  args.ss, args.bs)

                loss_all_p1 = args.w_loss_img * loss_img 
                loss_all_p1.backward()
                optimizer.step()

                #######################################################################################
                

                optimizer.zero_grad()
                net.eval()
                    
                loss_common, loss_ori = net.update_pwnn(batches, c_label,  args.ss, args.bs)

                net.train()


                loss_all_p2 = args.w_loss_com *loss_common + \
                            args.w_loss_ori *loss_ori 

                loss_all_p2.backward()
                optimizer.step()



                loss_common_ += args.w_loss_com * loss_common.item()
                loss_ori_ += args.w_loss_ori * loss_ori.item()
                

                loss_all_ += (loss_all_p1.item())
                loss_img_ += args.w_loss_img * loss_img.item() 




                if it % 10 == 0:
            

                    tqdm.write('Loss: {} | Loss_com: {}, Loss_ori: {} Loss_img: {}'.format(loss_all_/it, loss_common_/it, loss_ori_/it, loss_img_/it))
                    
                    with open("./log/{}.txt".format(args.logname), "a") as text_file:
                        text_file.write('{}---Loss: {} | Loss_com: {}, Loss_ori: {}, Loss_img: {}\n'.format(epoch, loss_all_/it, loss_common_/it, loss_ori_/it, loss_img_/it))
                    # tqdm.write('Loss: {} '.format(loss_all_/10./(it/10)))


        # # # # # ###############################################################################
            if epoch >= 10:
            # if (epoch > 0) and (epoch % 10 == 0) :

                tqdm.write(f"Evaluation started at {datetime.now()}")

                net.eval()
                counter_03 = 0
                counter_05 = 0
                counter_07 = 0

                for c in tqdm(range(200)):
                    index = np.where((c+1)==label_val)[0]


                    # pdb.set_trace()
                    sss = 1

                    for iit in range(int(np.floor(1.*len(index)/sss))):

                        if (iit+1) != int(np.floor(1.*len(index)/sss)):  

                            select_idx = index[iit*sss : (iit+1)*sss]
                        else:
                            select_idx = index[iit*sss :]

                        batches = torch.zeros([len(select_idx), 3, 224, 224])

                        for ii in range(len(select_idx)):
                            batches[ii] =  dataset_val.dataset[select_idx[ii]]

            
                        select_label = label_val[select_idx]
                        select_bbox = bbox_val[select_idx,:]
                        select_size = ori_shape_val[select_idx,:]
                        # pdb.set_trace()


                        batches = batches.cuda()
                        select_label = torch.tensor([select_label]).cuda()

                        with torch.no_grad():
                            counter_03_, counter_05_, counter_07_ = net.top1_loc(batches, select_bbox, select_label, select_size)
                            # if iouscore > 0.5:

                            counter_03+=counter_03_
                            counter_05+=counter_05_
                            counter_07+=counter_07_

                print(counter_03.max()/len(label_val))
                print(counter_05.max()/len(label_val))

                print(counter_07.max()/len(label_val))


                print((counter_03.max()/len(label_val)+counter_05.max()/len(label_val)+counter_07.max()/len(label_val))/3.)

                val_list = np.append(val_list, (counter_03.max()/len(label_val)+counter_05.max()/len(label_val)+counter_07.max()/len(label_val))/3.)

                tqdm.write('Top1_ori_acc: {}, Top1_masked_acc: {}, Top1_masked_set_acc: {}'.format(counter_03.max()/len(label_val), counter_05.max()/len(label_val), counter_07.max()/len(label_val)))
                with open("./log/{}.txt".format(args.logname), "a") as text_file:
                    text_file.write('---{}---\n'.format((counter_03.max()/len(label_val)+counter_05.max()/len(label_val)+counter_07.max()/len(label_val))/3.))

                    text_file.write('Top1_ori_acc: {}, Top1_masked_acc: {}, Top1_masked_set_acc: {}\n'.format(counter_03.max()/len(label_val), counter_05.max()/len(label_val), counter_07.max()/len(label_val)))
            else:
                val_list = np.append(val_list, 0)

            # pdb.set_trace()

            if len(val_list) == 1:

                path = os.path.join('./checkpoint', '{}_{}{}_epoch_best.pt'.format(args.mode, net.base_net, args.name))
                path2 = os.path.join('./checkpoint', 'OPT_{}_{}{}_epoch_best.pt'.format(args.mode, net.base_net, args.name))

                torch.save(net.state_dict(), path)
                torch.save(optimizer.state_dict(), path2)
                tqdm.write(f"State saved to {path}")

            else:
                if val_list[-1] > np.max(val_list[0:-1]):

                    path = os.path.join('./checkpoint', '{}_{}{}_epoch_best.pt'.format(args.mode, net.base_net, args.name))
                    path2 = os.path.join('./checkpoint', 'OPT_{}_{}{}_epoch_best.pt'.format(args.mode, net.base_net, args.name))

                    torch.save(net.state_dict(), path)
                    torch.save(optimizer.state_dict(), path2)
                    tqdm.write(f"State saved to {path}")

            path = os.path.join('./checkpoint', '{}_{}{}_epoch_last.pt'.format(args.mode, net.base_net, args.name))
            path2 = os.path.join('./checkpoint', 'OPT_{}_{}{}_epoch_last.pt'.format(args.mode, net.base_net, args.name))

            torch.save(net.state_dict(), path)
            torch.save(optimizer.state_dict(), path2)
            tqdm.write(f"State saved to {path}")


            scheduler.step()


    # # # # ###############################################################################


    if args.test:

        # state_path = os.path.join('/esat/qayd/kwang/Works/WSOD/WSOL', '{}_{}{}_epoch_best.pt'.format(args.mode, net.base_net, args.name))
        # net.load_state_dict(torch.load(state_path))

        tqdm.write(f"Evaluation started at {datetime.now()}")

        net.eval()
        counter_03 = 0
        counter_05 = 0
        counter_07 = 0

        for c in tqdm(range(200)):
            index = np.where((c+1)==label_test)[0]


            pdb.set_trace()
            args.ss = 5

            for iit in range(int(np.floor(1.*len(index)/args.ss))):

                if (iit+1) != int(np.floor(1.*len(index)/args.ss)):  

                    select_idx = index[iit*args.ss : (iit+1)*args.ss]
                else:
                    select_idx = index[iit*args.ss :]

                batches = torch.zeros([len(select_idx), 3, 224, 224])

                for ii in range(len(select_idx)):
                    batches[ii] =  dataset_test.dataset[select_idx[ii]]


                select_label = label_test[select_idx]
                select_bbox = bbox_test[select_idx,:]
                select_size = ori_shape[select_idx,:]
                # pdb.set_trace()


                batches = batches.cuda()
                select_label = torch.tensor([select_label]).cuda()

                with torch.no_grad():
                    counter_03_, counter_05_, counter_07_ = net.top1_loc(batches, select_bbox, select_label, select_size)
                    # if iouscore > 0.5:

                    counter_03+=counter_03_
                    counter_05+=counter_05_
                    counter_07+=counter_07_

        print(counter_03.max()/len(label_test))
        print(counter_05.max()/len(label_test))

        print(counter_07.max()/len(label_test))




        tqdm.write('Top1_ori_acc: {}, Top1_masked_acc: {}, Top1_masked_set_acc: {}'.format(counter_03.max()/len(label_test), counter_05.max()/len(label_test), counter_07.max()/len(label_test)))
        with open("./log/{}.txt".format(args.logname), "a") as text_file:
            text_file.write('Top1_ori_acc: {}, Top1_masked_acc: {}, Top1_masked_set_acc: {}\n'.format(counter_03.max()/len(label_test), counter_05.max()/len(label_test), counter_07.max()/len(label_test)))
            text_file.write('{}--{}--{}'.format(np.argmax(counter_03), np.argmax(counter_05), np.argmax(counter_07)))
            text_file.write('\n---{}---\n'.format((np.argmax(counter_03) + np.argmax(counter_05) + np.argmax(counter_07))/3.))

        print('{}--{}--{}'.format(np.argmax(counter_03), np.argmax(counter_05), np.argmax(counter_07)))




  




