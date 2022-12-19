from dataset.dataset import CUB, STANFORD_CAR, FGVC_aircraft, read_dataset, Stanford_Dogs
import numpy as np


def Read_Dataset(cfg, n_select_train):
    if cfg.DATASETS.NAMES == "CUB":
        train_data = CUB(cfg.DATASETS.ROOT_DIR, is_train=True)
        train_img = train_data.train_img
        train_label = train_data.train_label
        num_train = len(train_label)
        perm_index = np.random.permutation(num_train)
        select_samples_index = perm_index[0:n_select_train]
        select_train_img = list(np.array(train_img)[select_samples_index])
        select_train_label = list(np.array(train_label)[select_samples_index])
        train_dataloader = read_dataset([select_train_img, select_train_label], cfg.INPUT.SIZE_TRAIN,
                                        cfg.DATALOADER.NUM_INSTANCE, cfg.DATASETS.NAMES, cfg.DATALOADER.NUM_WORKERS, is_train=True,
                                        is_shuffle=True)
        test_data = CUB(cfg.DATASETS.ROOT_DIR, is_train=False)
        test_img = test_data.test_img
        test_label = test_data.test_label
        num_test = len(test_label)
        test_dataloader = read_dataset([test_img, test_label], cfg.INPUT.SIZE_TRAIN, cfg.TEST.IMS_PER_BATCH,
                                       cfg.DATASETS.NAMES, cfg.DATALOADER.NUM_WORKERS, is_train=False, is_shuffle=False)
        # return train_dataloader,num_train,test_dataloader,num_test
    elif cfg.DATASETS.NAMES == "AIR":
        train_data = FGVC_aircraft(cfg.DATASETS.ROOT_DIR, is_train=True)
        train_img_label = train_data.train_img_label
        num_train = len(train_img_label)
        perm_index = np.random.permutation(num_train)
        select_samples_index = perm_index[0:n_select_train]
        select_img_label = list(np.array(train_img_label)[select_samples_index])
        train_dataloader = read_dataset(select_img_label, cfg.INPUT.SIZE_TRAIN, cfg.DATALOADER.NUM_INSTANCE,
                                        cfg.DATASETS.NAMES, cfg.DATALOADER.NUM_WORKERS, is_train=True, is_shuffle=True)
        test_data = FGVC_aircraft(cfg.DATASETS.ROOT_DIR, is_train=False)
        test_img_label = test_data.test_img_label
        num_test = len(test_img_label)
        test_dataloader = read_dataset(test_img_label, cfg.INPUT.SIZE_TRAIN, cfg.TEST.IMS_PER_BATCH, cfg.TEST.IMS_PER_BATCH,
                                       cfg.DATALOADER.NUM_WORKERS, is_train=False, is_shuffle=False)
        # return train_dataloader,num_train,test_dataloader,num_test
    elif cfg.DATASETS.NAMES == "CAR":
        train_data = STANFORD_CAR(cfg.DATASETS.ROOT_DIR, is_train=True)
        train_img_label = train_data.train_img_label
        num_train = len(train_img_label)
        perm_index = np.random.permutation(num_train)
        select_samples_index = perm_index[0:n_select_train]
        select_img_label = list(np.array(train_img_label)[select_samples_index])
        train_dataloader = read_dataset(select_img_label, cfg.INPUT.SIZE_TRAIN, cfg.DATALOADER.NUM_INSTANCE,
                                        cfg.DATASETS.NAMES, cfg.DATALOADER.NUM_WORKERS, is_train=True, is_shuffle=True)
        test_data = STANFORD_CAR(cfg.DATASETS.ROOT_DIR, is_train=False)
        test_img_label = test_data.test_img_label
        num_test = len(test_img_label)
        test_dataloader = read_dataset(test_img_label, cfg.INPUT.SIZE_TRAIN, cfg.TEST.IMS_PER_BATCH, cfg.DATASETS.NAMES,
                                       cfg.DATALOADER.NUM_WORKERS, is_train=False, is_shuffle=False)
        # return train_dataloader,num_train, test_dataloader,num_test
    else:
        train_data = Stanford_Dogs(cfg.DATASETS.ROOT_DIR, is_train=True)
        train_img_label = train_data.train_img_label
        num_train = len(train_img_label)
        perm_index = np.random.permutation(num_train)
        select_samples_index = perm_index[0:n_select_train]
        select_img_label = list(np.array(train_img_label)[select_samples_index])
        train_dataloader = read_dataset(select_img_label, cfg.INPUT.SIZE_TRAIN, cfg.DATALOADER.NUM_INSTANCE,
                                        cfg.DATASETS.NAMES, cfg.DATALOADER.NUM_WORKERS, is_train=True, is_shuffle=True)
        test_data = Stanford_Dogs(cfg.DATASETS.ROOT_DIR, is_train=False)
        test_img_label = test_data.test_img_label
        num_test = len(test_img_label)
        test_dataloader = read_dataset(test_img_label, cfg.INPUT.SIZE_TRAIN, cfg.TEST.IMS_PER_BATCH, cfg.DATASETS.NAMES,
                                       cfg.DATALOADER.NUM_WORKERS, is_train=False, is_shuffle=False)
        # return train_dataloader,num_train, test_dataloader,num_test
    return train_dataloader, num_train, test_dataloader, num_test
