import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
import albumentations.pytorch
import numpy as np
import pandas as pd
import random
import json

CLS_NAME_TO_ID = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Consolidation': 2, 'Edema': 3, 'Lung Opacity': 4, 'Pneumothorax': 5, 'pneumonia': 6}

class DataUnet(Dataset):
    def __init__(
            self, 
            root_dir, 
            image_set, 
            image_size, 
            multi_channel_mask = False, 
            return_yolo_labels = False,
            return_text_labels = True,
            data_path = None, # to replace the original parent dir
            return_organ_mask = False, # for organ segmentation training
            no_aug=False,
            ):
        super().__init__()
        self.image_size = image_size
        self.image_set = image_set
        self.return_yolo_labels = return_yolo_labels # used for end-to-end training only
        self.multi_channel_mask = multi_channel_mask
        self.return_text_labels = return_text_labels
        self.class_name_to_id = CLS_NAME_TO_ID
        self._labels_path = f"{root_dir}/{image_set}.csv"
        self.data_path = data_path
        ann_df = pd.read_csv(self._labels_path)
        ann_df['class_name'].replace({"No Finding":""}, inplace=True) # replace no finding with empty string
        # ann_df['class_name'].replace({"No finding":""}, inplace=True) # replace no finding with empty string
        ann_df['class_name'].replace({"Pleural effusion":"Pleural Effusion"}, inplace=True) 
        self.ann_df = ann_df
        self.img_ids = self.ann_df['image_id'].unique().tolist()
        
        self.return_organ_mask = return_organ_mask

        if return_yolo_labels: # when returning img, mask, bbox
            self.transform = A.Compose([
                A.RandomCrop(width=image_size, height=image_size),
                # A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.pytorch.transforms.ToTensorV2(),
                A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
                ),
            ], bbox_params=A.BboxParams(format='yolo',  label_fields=['class_id']))

        if no_aug: # when returning img and mask
            self.transform = A.Compose([
                A.augmentations.geometric.resize.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR),
                A.pytorch.transforms.ToTensorV2(),
            ])
        else: # when returning img and mask
            self.transform = A.Compose([
                A.augmentations.geometric.resize.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR),
                # A.RandomCrop(width=self.image_size, height=self.image_size),
                # A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.pytorch.transforms.ToTensorV2(),
                
            ])
            self.norm_transform = A.Compose([
                A.Normalize(
                    mean=(0.5),
                    std=(0.5)
                ),])

        self.resize_transform =A.Compose( # size of label should be equal to the latent size
            [A.augmentations.geometric.resize.Resize(height = image_size//4, width = image_size//4, interpolation=cv2.INTER_LINEAR),
            A.pytorch.transforms.ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_ids)

    def get_mask(self, img_data):
        mask = torch.zeros((1, 512, 512)) # keeping it 512 because the annotations are for 512x512 images        
        for idx, img_data_i in img_data.iterrows():
            obj_class = img_data_i["class_id"]
            class_name = img_data_i["class_name"]
            if obj_class == 14:
                # obj_class = 6
                x_min_all = -1
                y_min_all = -1
                y_max_all = -1
                x_max_all = -1
            else:
                x_min_all = img_data_i["x_min"]  # original annotations are for 512x512 images
                y_min_all = img_data_i["y_min"] 
                y_max_all = img_data_i["y_max"] 
                x_max_all = img_data_i["x_max"] 

            if y_min_all > 0:
                mask[0, int(y_min_all):int(y_max_all), int(x_min_all):int(x_max_all)] = self.class_name_to_id[class_name]
        return mask # 3 channel mask

    def get_labels_volume(self, img_id):
        img_data = self.ann_df[self.ann_df['image_id'] == img_id].copy().round(3)
        obj_classes = img_data["class_id"].to_list()
        x_min_all = img_data["x_min"].fillna(-1).to_list()
        y_min_all = img_data["y_min"].fillna(-1).to_list()
        y_max_all = img_data["y_max"].fillna(-1).to_list()
        x_max_all = img_data["x_max"].fillna(-1).to_list()


        if self.multi_channel_mask:
            labels = torch.zeros((self.num_class-1, self.image_size, self.image_size)) # single channel mask for all classes
            for i, cls_id in enumerate(obj_classes):
                mask_idx = self.class_id_mapping[cls_id] - 1 # class_id_mapping starts from 1
                if y_min_all[i] > 0:
                    labels[mask_idx, int(y_min_all[i]):int(y_max_all[i]), int(x_min_all[i]):int(x_max_all[i])] = 1

        else: # single channel mask
            labels = torch.zeros((1, self.image_size, self.image_size)) # single channel mask for all classes
            for i, cls_id in enumerate(obj_classes):
                if y_min_all[i] > 0:
                    labels[:, int(y_min_all[i]):int(y_max_all[i]), int(x_min_all[i]):int(x_max_all[i])] = self.class_id_mapping[cls_id]
        return labels

    def get_circle(self, size = 256, r = None):
        r = r if r is not None else torch.randint(50, 400, (1,)).item()
        y = torch.randint(80, 450, (1,)).item()
        x = torch.randint(80, 450, (1,)).item()
        xx, yy = np.mgrid[:size, :size]
        circle = ((xx - y) ** 2 + (yy - x) ** 2) < r**2
        return torch.tensor(circle).unsqueeze(0).float()

    def get_text_label(self):
        pass

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_data = self.ann_df[self.ann_df['image_id'] == img_id]
        if self.data_path is not None: # not used by default
            img_path = img_data['path'].tolist()[0].replace('REPLACE THIS ACCORDINGLY', self.data_path)
        else:
            img_path = img_data['path'].tolist()[0]
        img = torch.load(img_path) # image has shape [512, 512]
        if img.shape[0] == 1: # quick fix for RSNA images
            img = img[0]

        if self.return_organ_mask: # get the organ mask
            masks = torch.load(img_path[:-3]+"_seg.pt").unsqueeze(0)
        else: # get lesion mask
            masks = self.get_mask(img_data)

        try:
            transformed = self.transform(image=img.numpy(), mask=masks[0,:,:].numpy())
        except:
            print('isss')
        masks = transformed['mask'] # original size masks for controlnet
        img = transformed['image']

        masks = torch.stack([masks, masks, masks]) # conver to 3 channel mask

        img = img.expand(3,*img.shape[1:]) # convert to 3 channel
        img = (img - img.min())/ (img.max() - img.min())

        if self.return_text_labels:
            text_label = f"{' '.join(img_data['class_name'].tolist())}"
            return {"image":img, "mask": masks,  "text_label":text_label, "image_info":img_id}
        else:
            return {"image":img, "mask": masks.unsqueeze(0)}

# range 0-1
class DataVAE(Dataset):
    def __init__(
            self, 
            csv_path, 
            image_size=256, 
            is_val=False, 
            pretraining=True, 
            reports_path = None,
            use_nih_data = False, 
            use_rsna_data=False,
            # return_labels=False,
            return_text_labels=False,
            return_image_info=False,
            return_report=True,
            return_image_only=False,
            data_path=None,
            no_aug=False,
            finetuning=False,
            controlnet=False,
            mask_prob=0.5,
            ):
        super().__init__()
        self.root_dir = csv_path
        if reports_path is not None:
            self.reports_dict = json.load(open(reports_path, 'r'))
        self.mask_prob = mask_prob
        annotations_df = pd.read_csv(csv_path)

        if is_val:
            annotations_df = annotations_df[annotations_df['split'] == 'test']
        else:
            annotations_df = annotations_df[annotations_df['split'].isin(['train', 'validate']) ]

        self.label_cols = annotations_df.columns[13:-2] # used for text labels
        self.finetuning = finetuning
        self.controlnet = controlnet
        self.return_text_labels = return_text_labels
        # self.return_labels = return_labels 
        self.return_image_info = return_image_info
        self.return_image_only = return_image_only # for VAE training, to speed up training
        self.data_path = data_path
        # if return_text_labels: # return labels must be true if return_text_labels is true
            # self.return_labels = True
        
        annotations_df = annotations_df[annotations_df['ViewPosition']== "AP"]
        self.no_aug = no_aug
        self.df = annotations_df
        
        if use_nih_data:
            print('loading nih data')
            annotations_df_nih = pd.read_csv("../dataset/nih_cxr.csv")
            annotations_df_nih = annotations_df_nih[annotations_df_nih['View Position']== "AP"]
            annotations_df_nih.rename(columns={"Image Index": "dicom_id"}, inplace=True)
            label_cols = annotations_df.columns[13:-2]
            # mimic_nih_overlapping_cls = [i for i in label_cols if i in annotations_df_nih.columns]
            # mimic_nih_overlapping_cls = mimic_nih_overlapping_cls + ['No Finding']
            mimic_nih_nonoverlapping_cls = [i for i in label_cols if i not in annotations_df_nih.columns]
            annotations_df_nih[mimic_nih_nonoverlapping_cls] = 0
            annotations_df_nih = annotations_df_nih[['dicom_id', 'path'] + list(label_cols)]
            self.df = pd.concat([self.df, annotations_df_nih], ignore_index=True, join='outer')

        if no_aug:
            self.transform1 = A.Compose([
                A.augmentations.geometric.resize.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            ])

            self.transform2 = A.Compose([
                A.pytorch.transforms.ToTensorV2(),                
            ])

            self.transform2_mask = A.Compose([
                A.pytorch.transforms.ToTensorV2(),                
            ])
        else:
            self.transform1 = A.Compose([
                A.augmentations.geometric.resize.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
                # A.RandomCrop(width=self.image_size, height=self.image_size),
                # A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),        
            ])

            self.transform2 = A.Compose([
                A.RandomBrightnessContrast(p=0.2),
                A.pytorch.transforms.ToTensorV2(),                
            ])

            self.transform2_mask = A.Compose([
                A.pytorch.transforms.ToTensorV2(),                
            ])

    def binary_list_to_int(self, input_list):
        # Check if the input list contains only 1s and 0s
        if all(bit == 0 or bit == 1 for bit in input_list):
            # Convert the binary list to a binary string and then to an integer
            binary_string = ''.join(map(str, input_list))
            decimal_integer = int(binary_string, 2)
            return decimal_integer
        else:
            raise ValueError("Input list should contain only 1s and 0s")

    def __len__(self):
        return len(self.df)

    def get_text_labels(self, labels_ids):
        label_classes = [self.label_cols[i] for i in range(len(labels_ids)) if labels_ids[i]==1]
        text_labels = ''
        for idx, cls in enumerate(label_classes):
            if len(label_classes) == 1:
                text_labels +=cls 
            elif idx == len(label_classes)-1:
                text_labels += f'{cls}'
            else:
                text_labels += cls + ' '
        return text_labels

    def vae_mask_to_image_getitem(self, img_data):
        if self.data_path is not None:
            img_path = img_data['path'].replace('/share/nvmedata/', self.data_path)
        else:
            img_path = img_data['path']
        
        mask_path = img_path.replace('mimic_cxr_pt', 'mimic_cxr_pt_mask')

        mask = torch.load(mask_path).unsqueeze(0) # mask shape = [512, 512]         
        img = torch.load(img_path) # image has shape [1, 512, 512]
        
        if self.no_aug:
            transformed = self.transform(image=img[0].numpy(), mask=mask[0].numpy())
            img = transformed['image']
            mask = transformed['mask']
        # transformed = self.transform({'image':img, 'mask':mask})# bug
        else:
            transformed = self.transform1(image = img[0].numpy(),mask=mask[0].numpy())
            img = self.transform2(image = transformed['image'])['image']
            mask = self.transform2_mask(image = transformed['mask'])['image']
        
        if mask.max()> 0: # loss nan bug fix, some images have all 0 masks
            mask = mask/ 4 # or this

        img = img.expand(3,*img.shape[1:])
        mask = mask.expand(3,*mask.shape[1:]) #TODO: check if we need to normalize mask

        if random.random() < self.mask_prob:
            return {'image':mask.float(), 'target':img}
        else:
            return {'image':img, 'target':img}

    def diffusion_ft(self, img_data):
        vae_data = self.vae_mask_to_image_getitem(img_data) # this has image and target keys
        labels_ids = [int(i) for i in img_data[self.label_cols].tolist()]
        
        text_labels = self.get_text_labels(labels_ids)
        vae_data['text_labels'] = text_labels # this is used for diffusion forward
        image_info = f"{img_data['subject_id']}_{img_data['study_id']}_{img_data['dicom_id']}"
        vae_data['image_info'] = image_info
        return vae_data

    def diffusion_text_labels_to_image_getitem(self, img_data, return_report=False):

        if self.data_path is not None:
            img_path = img_data['path'].replace('/share/nvmedata/', self.data_path)
        else:
            img_path = img_data['path']

        # image loading
        img = torch.load(img_path)
        img = img.expand(3,*img.shape[1:]) # convert to 3 channel
        if return_report:
            report = self.reports_dict[img_data['dicom_id']]['report']
        else:
            report = ''
        
        labels_ids = [int(i) for i in img_data[self.label_cols].tolist()]
        label = self.binary_list_to_int(labels_ids)

        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img[0]

        transformed = self.transform1(image = img.permute(1,2,0).numpy())['image']
        img = self.transform2(image = transformed)['image']
        
        if self.return_text_labels:
            text_labels = self.get_text_labels(labels_ids)

            image_info = f"{img_data['subject_id']}_{img_data['study_id']}_{img_data['dicom_id']}"
            return {'image':img, 'label': label, "report": report, "image_info": image_info, "text_labels": text_labels}

        return {'image':img, 'label': label, "report": report}

    def __getitem__(self, idx):
        img_data = self.df.iloc[idx]
        
        # for mask/image to image VAE training
        # this can return organ mask and lesion masks both
        # this returns image, target
        if self.return_image_only: 
            return self.vae_mask_to_image_getitem(img_data)

        # for ddim inversion i.e. vae + diffusion model training
        # this returns image, target, text_labels
        elif self.finetuning or self.controlnet:
            return self.diffusion_ft(img_data)
        
        # for diffusion model training (text labels to image)
        # this returns image, label, report, image_info, text_labels
        else:
            return self.diffusion_text_labels_to_image_getitem(img_data)


def get_bbox_dataloaders(
        name, 
        batch_size, 
        dataset_path, 
        im_set, 
        im_size, 
        num_workers = 8, 
        return_yolo_labels = False,
        return_text_labels = False,
        ):
    dataset = globals()[name](
        dataset_path, 
        im_set, 
        im_size, 
        return_yolo_labels=return_yolo_labels, 
        return_text_labels=return_text_labels
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    return dataloader

# rename to get_pretaining_dataset
def get_dataset_distributed_VAE(
        name, 
        world_size, 
        rank, 
        batch_size, 
        num_workers = 4, 
        use_rank=True, 
        custom_sampler = None,
        **kwargs
        ):
    train_dataset = globals()[name](
        **kwargs
        )
    val_dataset = globals()[name](
        is_val=True, 
        **kwargs
        )

    if use_rank:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        return train_dataloader, val_dataloader

    else:
        if custom_sampler is not None:
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                num_workers=num_workers,
                custom_sampler = custom_sampler,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            )



        return train_dataloader, val_dataloader

def get_vindr_dataloader(dataset_path, batch_size, img_set, img_size, num_cls, num_workers = 8, return_yolo_labels=False, return_text_labels=True, data_path=None, return_organ_mask = False, no_aug=False):



    dataset = DataUnet(
        dataset_path, 
        img_set, 
        img_size, 
        num_cls, 
        return_yolo_labels=return_yolo_labels, 
        return_text_labels = return_text_labels, 
        data_path = data_path,
        return_organ_mask=return_organ_mask,
        no_aug=no_aug,
        )

    if return_yolo_labels:
        def collate_fn(batch):
            return tuple(zip(*batch))
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            # num_workers=num_workers, # cant use num_workers > 0 with custom collate_fn
            collate_fn=collate_fn, # for yolo labels
        )
        
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
        )

    return dataloader

