import argparse
import random

from torchvision import transforms
from torch.utils import data
import os
import torch
import json
from PIL import Image
import glob
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import timm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

ROOT_PATH = '/tmp/my_file/pycharm_projects/demo/clean_resized_images'
name_to_class_ids_file = os.path.join(ROOT_PATH, 'transformer/image_name_to_class_id_and_name.json')


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def get_params():
    parser = argparse.ArgumentParser(description='PIM Params')

    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--nn', default='vit_base_patch16_224', type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--eps', default=16 / 255, type=float)
    parser.add_argument('--alpha', default=2 / 255, type=float)
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--m1', default=5, type=int)
    parser.add_argument('--m2', default=3, type=int)
    parser.add_argument('--resize_rate', default=0.9, type=float)
    parser.add_argument('--diversity_prob', default=0.5, type=float)
    parser.add_argument('--decay', default=1, type=float)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    parser.add_argument('--seed', default=1029, type=int)
    parser.add_argument('--log_dir', default='pi_log', type=str)
    parser.add_argument('--log_name', default='pi_result', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    subfolder = os.path.join(args.log_dir, args.nn)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    return args


class AdvDataset(data.Dataset):
    def __init__(self, adv_path=os.path.join(ROOT_PATH, 'clean_resized_images')):
        self.transform = transforms.Compose([transforms.ToTensor()])
        paths = glob.glob(os.path.join(adv_path, '*.png'))
        paths = [i.split(os.sep)[-1] for i in paths]
        print('Using ', len(paths))
        paths = [i.strip() for i in paths]
        self.query_paths = [i.split('.')[0] + '.JPEG' for i in paths]
        self.paths = [os.path.join(adv_path, i) for i in paths]

        with open(name_to_class_ids_file, 'r') as ipt:
            self.json_info = json.load(ipt)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        query_path = self.query_paths[index]
        class_id = self.json_info[query_path]['class_id']
        # deal with image
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, class_id


class Base_Attack:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.patch_sizes = [2, 4, 8, 16, 32, 28, 56]
        self.nums = np.array([256, 64, 32, 8, 4, 4, 2]) * 8
        self.nums = self.nums.tolist()

    def __call__(self, images, labels):
        images, labels = images.clone().detach(), labels.clone().detach()
        return self.forward(images, labels)

    def forward(self, images, labels):
        NotImplementedError

    def shuffle(self, adv_images, img_size, patch_size, num):
        adv_images_temp = rearrange(adv_images,
                                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                    p1=patch_size, p2=patch_size)
        row = torch.arange((img_size // patch_size) ** 2)
        x1 = torch.randint(0, (img_size // patch_size) ** 2, (num,))
        x2 = torch.randint(0, (img_size // patch_size) ** 2, (num,))
        for i in range(num):
            temp1 = row[x1[i]].clone()
            temp2 = row[x2[i]].clone()
            row[x1[i]] = temp2
            row[x2[i]] = temp1
        adv_images_temp = adv_images_temp[:, row, :]  # images have been shuffled already
        adv_images_temp = rearrange(adv_images_temp, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                    h=img_size // patch_size, w=img_size // patch_size,
                                    p1=patch_size, p2=patch_size)

        return adv_images_temp


class PIM(Base_Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, m1=5, m2=3, resize_rate=0.9, diversity_prob=0.5,
                 decay=1):
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.m1 = m1  # SI
        self.m2 = m2  # Admix
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.decay = decay

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def forward(self, inputs, labels, admix_images=None):
        inputs = inputs.clone().detach()
        labels = labels.clone().detach()
        if admix_images is not None:
            admix_images = admix_images.clone().detach()
        else:
            admix_images = inputs.clone().detach()

        adv_images = inputs.clone().detach()

        momentum = torch.zeros_like(inputs)
        for _ in range(self.steps):
            adv_images.requires_grad = True
            grad = torch.zeros_like(inputs)
            for i in range(len(self.patch_sizes)):
                patch_size = self.patch_sizes[i]
                num = self.nums[i] // 2
                augmented_inputs = adv_images
                if self.m2 > 0:
                    # Admix
                    augmented_inputs = torch.cat(
                        [adv_images + 0.2 * admix_images[torch.randperm(admix_images.size(0))[0]].unsqueeze(dim=0)
                         for _ in range(self.m2)], dim=0)
                if self.m1 > 0:
                    # SI
                    augmented_inputs = torch.cat([augmented_inputs / (2 ** i) for i in range(self.m1)], dim=0)
                if self.diversity_prob > 0:
                    # DI
                    augmented_inputs = self.input_diversity(augmented_inputs)
                # RI
                augmented_inputs = self.shuffle(augmented_inputs, augmented_inputs.size(-1), patch_size, num)
                loss_label = torch.cat([labels] * (augmented_inputs.shape[0] // inputs.shape[0]), dim=0)
                logits = self.model(augmented_inputs)
                loss = torch.nn.CrossEntropyLoss()(logits, loss_label)
                grad += torch.autograd.grad(loss, adv_images)[0]

            if self.m2 > 0 and self.m1 > 0:
                grad /= self.m2 * len(self.patch_sizes) * self.m1
            elif self.m2 > 0:
                grad /= self.m2 * len(self.patch_sizes)
            elif self.m2 > 0:
                grad /= len(self.patch_sizes) * self.m1
            else:
                grad /= len(self.patch_sizes)
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = self.decay * momentum + grad
            grad = momentum

            adv_images = adv_images + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - inputs, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(inputs + delta, min=0, max=1).detach()

        return adv_images


def main():
    args = get_params()
    seed_torch(args.seed)
    set = AdvDataset()
    val_loader = torch.utils.data.DataLoader(set, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.workers,
                                             pin_memory=True)
    device = torch.device(args.gpu)
    device_ids = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))))

    soruce_model_name = args.nn
    source_model = timm.create_model(soruce_model_name)
    source_model.load_state_dict(torch.load(f'./run_attack/weights/{soruce_model_name}.pth'))
    source_model = torch.nn.DataParallel(source_model.to(device), device_ids=device_ids).eval()

    black_models_name = ['vit_base_patch16_224', 'deit_base_patch16_224', 'swin_base_patch4_window7_224',
                         'pit_b_224', 'cait_s24_224', 'visformer_small']
    black_models = []
    for item in black_models_name:
        black_model = timm.create_model(item)
        black_model.load_state_dict(torch.load(f'./run_attack/weights/{item}.pth'))
        black_models.append(torch.nn.DataParallel(black_model.to(device), device_ids=device_ids).eval())

    atk = PIM(source_model, eps=args.eps, alpha=args.alpha, steps=args.steps, m1=args.m1, m2=args.m2,
              resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, decay=args.decay)

    total = 0
    transfer_correct = {item: 0 for item in black_models_name}
    transfer_clean = {item: 0 for item in black_models_name}

    for idx, (imgaes, labels) in enumerate(val_loader):
        imgaes, labels = imgaes.to(device), labels.to(device)
        adv_images = atk(imgaes, labels)

        total += imgaes.size(0)

        with torch.no_grad():
            for model_name, model in zip(black_models_name, black_models):
                pred = model(adv_images)
                transfer_correct[model_name] += (pred.argmax(dim=-1) == labels).sum().item()
                pred = model(imgaes)
                transfer_clean[model_name] += (pred.argmax(dim=-1) == labels).sum().item()

                with open(f'{args.log_dir}/{args.nn}/{args.log_name}.txt', 'w') as f:
                    f.write('Iter'.ljust(10, ' ') + 'model_name'.ljust(40, ' ') +
                            'ASR'.ljust(20, ' ') + 'Acc'.ljust(15, ' ') + 'Correct' + '\n')
                    for model_name in black_models_name:
                        s = f'[{idx}/{len(val_loader)}]'.ljust(10, ' ') + model_name.ljust(40, ' ') + \
                            str(round(100*(1 - transfer_correct[model_name] / total), 2)).ljust(20, ' ') + \
                            str(round(100*(transfer_clean[model_name] / total), 2)).ljust(15, ' ') + \
                            str(transfer_correct[model_name]) + '-' + str(transfer_clean[model_name]) + '\n'
                        f.write(s)


if __name__ == '__main__':
    main()
