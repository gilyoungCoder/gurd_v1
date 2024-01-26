# pylint: disable=E1101
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from physionet import PhysioNet, get_data_min_max
from sklearn import model_selection
from sklearn import metrics

import pandas as pd

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, args, device):
    observed_data, observed_mask \
        = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

    noise_std = args.std  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if args.norm:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl


def evaluate_classifier(model, test_loader, dec=None, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if args.classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x)
                else:
                    out = classifier(z0)
            if args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    true = np.concatenate(true, 0)
    acc = np.mean(pred.argmax(1) == true)
    auc = metrics.roc_auc_score(
        true, pred[:, 1]) if not args.classify_pertp else 0.
    return test_loss/pred.shape[0], acc, auc


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def get_physionet_data(args, device, q, flag=1):
    train_dataset_obj = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=min(10000, args.n),
                                  device=device)
    # Use custom collate_fn to combine samples with arbitrary time observations.
    # Returns the dataset along with mask and time steps
    test_dataset_obj = PhysioNet('data/physionet', train=False,
                                 quantization=q,
                                 download=True, n_samples=min(10000, args.n),
                                 device=device)

    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]

    if not args.classif:
        # Concatenate samples from original Train and Test sets
        # Only 'training' physionet samples are have labels.
        # Therefore, if we do classifiction task, we don't need physionet 'test' samples.
        total_dataset = total_dataset + \
            test_dataset_obj[:len(test_dataset_obj)]
    print(len(total_dataset)) 
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                             random_state=42, shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]

    # n_samples = len(total_dataset) 특성 숫자/텐서의 마지막 차원
    input_dim = vals.size(-1)
    data_min, data_max = get_data_min_max(total_dataset, device)
    batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
    if flag:
        test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
                                                      data_min=data_min, data_max=data_max)

        if args.classif:
            train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                                    random_state=11, shuffle=True)
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            val_data_combined = variable_time_collate_fn(
                val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            print(train_data_combined[1].sum(
            ), val_data_combined[1].sum(), test_data_combined[1].sum())
            print(train_data_combined[0].size(), train_data_combined[1].size(),
                  val_data_combined[0].size(), val_data_combined[1].size(),
                  test_data_combined[0].size(), test_data_combined[1].size())

            train_data_combined = TensorDataset(
                train_data_combined[0], train_data_combined[1].long().squeeze())
            val_data_combined = TensorDataset(
                val_data_combined[0], val_data_combined[1].long().squeeze())
            test_data_combined = TensorDataset(
                test_data_combined[0], test_data_combined[1].long().squeeze())
        else:
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            print(train_data_combined.size(), test_data_combined.size())

        train_dataloader = DataLoader(
            train_data_combined, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=batch_size, shuffle=False)

    else:
        # train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
        #                               collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device, data_type="train",
        #                                                                                  data_min=data_min, data_max=data_max))
        # test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
        #                              collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device, data_type="test",
        #                                                                                 data_min=data_min, data_max=data_max))
        print("improper case -- no fn2")
    attr_names = train_dataset_obj.params
    data_objects = {"dataset_obj": train_dataset_obj,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    "attr": attr_names,  # optional
                    "classif_per_tp": False,  # optional
                    "n_labels": 1}  # optional
    if args.classif:
        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
        data_objects["val_dataloader"] = val_dataloader
    return data_objects

# 시간, 값들을 모두 0~1로 정규화
def variable_time_collate_fn(batch, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    # number of labels
    N = batch[0][-1].shape[1] if activity else 1
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if classify:
        if activity:
            combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(batch), N]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        ## max 길이를 둬서 앞에서부터 채우고, 나머지는 0이 되게 하는 방식
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
        if classify:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                combined_labels[b] = labels.to(device)

    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    if classify:
        return combined_data, combined_labels
    else:
        return combined_data
    
    
###------------for grud
# functions to process the time in the data
def timeparser(time):
    return pd.to_timedelta(time + ':00')

def timedelta_to_day_figure(timedelta):
    return timedelta.days + (timedelta.seconds/86400) #(24*60*60)


def df_to_x_m_d(df, inputdict, size, id_posistion, split):
    grouped_data = df.groupby('Time')
    
    #generate input vectors
    x = np.zeros((len(inputdict)-2, grouped_data.ngroups))
    masking = np.zeros((len(inputdict)-2, grouped_data.ngroups))
    delta = np.zeros((split, size))
    timetable = np.zeros(grouped_data.ngroups)
    id = 0
    
    all_x = np.zeros((split,1))
    
    s_dataset = np.zeros((3, split, size))
   
    if grouped_data.ngroups > size:
        
        # fill the x and masking vectors
        pre_time = pd.to_timedelta(0)
        t = 0
        for row_index, value in df.iterrows():
            '''
            t = colum, time frame
            agg_no = row, variable
            '''
            #print(value)
            agg_no = inputdict[value.Parameter]

            # same timeline check.        
            if pre_time != value.Time:
                pre_time = value.Time
                t += 1
                timetable[t] = timedelta_to_day_figure(value.Time)

            #print('agg_no : {}\t t : {}\t value : {}'.format(agg_no, t, value.Value))
            x[agg_no, t] = value.Value    
            masking[agg_no, t] = 1
        
        '''
        # generate random index array 
        ran_index = np.random.choice(grouped_data.ngroups, size=size, replace=False)
        ran_index.sort()
        ran_index[0] = 0
        ran_index[size-1] = grouped_data.ngroups-1
        '''
        
        # generate index that has most parameters and first/last one.
        ran_index = grouped_data.count()
        ran_index = ran_index.reset_index()
        ran_index = ran_index.sort_values('Value', ascending=False)
        ran_index = ran_index[:size]
        ran_index = ran_index.sort_index()
        ran_index = np.asarray(ran_index.index.values)
        ran_index[0] = 0
        ran_index[size-1] = grouped_data.ngroups-1
        
        #print(ran_index)
        
        # take id for outcome comparing
        id = x[id_posistion, 0]
        
        # remove unnesserly parts(rows)
        x = x[:split, :]
        masking = masking[:split, :]
        
        # coulme(time) sampling
        x_sample = np.zeros((split, size))
        m_sample = np.zeros((split, size))
        time_sample = np.zeros(size)

        t_x_sample = x_sample.T
        t_marsking = m_sample.T
        #t_time = t_sample.T
        
        t_x = x.T
        t_m = masking.T
        #t_t = t.T

        it = np.nditer(ran_index, flags=['f_index'])
        while not it.finished:
            #print('it.index = {}, it[0] = {}, ran_index = {}'.format(it.index, it[0], ran_index[it.index]))
            t_x_sample[it.index] = t_x[it[0]]
            t_marsking[it.index] = t_m[it[0]]
            time_sample[it.index] = timetable[it[0]]
            it.iternext()
        
        x = x_sample
        masking = m_sample
        timetable = time_sample
        '''
        # normalize the X
        nor_x = x/max_input[:, np.newaxis]
        '''
        # fill the delta vectors
        for index, value in np.ndenumerate(masking):
            '''
            index[0] = row, agg
            index[1] = col, time
            '''
            if index[1] == 0:
                delta[index[0], index[1]] = 0
            elif masking[index[0], index[1]-1] == 0:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1] + delta[index[0], index[1]-1]
            else:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1]
    
    else:
                
        # fill the x and masking vectors
        pre_time = pd.to_timedelta(0)
        t = 0
        for row_index, value in df.iterrows():
            '''
            t = colum, time frame
            agg_no = row, variable
            '''
            #print(value)
            agg_no = inputdict[value.Parameter]

            # same timeline check.        
            if pre_time != value.Time:
                pre_time = value.Time
                t += 1
                timetable[t] = timedelta_to_day_figure(value.Time)

            #print('agg_no : {}\t t : {}\t value : {}'.format(agg_no, t, value.Value))
            x[agg_no, t] = value.Value    
            masking[agg_no, t] = 1
        
        # take id for outcome comparing
        id = x[id_posistion, 0]
        
        # remove unnesserly parts(rows)
        x = x[:split, :]
        masking = masking[:split, :]
        
        x = np.pad(x, ((0,0), (size-grouped_data.ngroups, 0)), 'constant')
        masking = np.pad(masking, ((0,0), (size-grouped_data.ngroups, 0)), 'constant')
        timetable = np.pad(timetable, (size-grouped_data.ngroups, 0), 'constant')
        '''
        # normalize the X
        nor_x = x/max_input[:, np.newaxis]
        '''
        # fill the delta vectors
        for index, value in np.ndenumerate(masking):
            '''
            index[0] = row, agg
            index[1] = col, time
            '''
            if index[1] == 0:
                delta[index[0], index[1]] = 0
            elif masking[index[0], index[1]-1] == 0:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1] + delta[index[0], index[1]-1]
            else:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1]
    

    all_x = np.concatenate((all_x, x), axis=1)
    all_x = all_x[:,1:]
    
    s_dataset[0] = x
    s_dataset[1] = masking
    s_dataset[2] = delta
    
    return s_dataset, all_x, id


def get_mean(x):
    x_mean = []
    for i in range(x.shape[0]):
        mean = np.mean(x[i])
        x_mean.append(mean)
    return x_mean

def get_median(x):
    x_median = []
    for i in range(x.shape[0]):
        median = np.median(x[i])
        x_median.append(median)
    return x_median

def get_std(x):
    x_std = []
    for i in range(x.shape[0]):
        std = np.std(x[i])
        x_std.append(std)
    return x_std

def get_var(x):
    x_var = []
    for i in range(x.shape[0]):
        var = np.var(x[i])
        x_var.append(var)
    return x_var
    
# dataset shape : (4000, 3, 33, 49)
def dataset_normalize(dataset, mean, std):
    for i in range(dataset.shape[0]):        
        dataset[i][0] = (dataset[i][0] - mean[:, None])
        dataset[i][0] = dataset[i][0]/std[:, None]
        
    return dataset

def normalize_chk(dataset):
    all_x_add = np.zeros((dataset[0][0].shape[0],1))
    for i in range(dataset.shape[0]):
        all_x_add = np.concatenate((all_x_add, dataset[i][0]), axis=1)
    
    mean = get_mean(all_x_add)
    median = get_median(all_x_add)
    std = get_std(all_x_add)
    var = get_var(all_x_add)
    
    return mean, median, std, var

# only check In-hospital_death
def df_to_y1(df):
    output = df.values
    output = output[:,5:]

    return output

def data_dataloader(dataset, outcomes,\
                    train_proportion = 0.8, dev_proportion = 0.2, test_proportion = 0.2):
    
    train_index = int(np.floor(dataset.shape[0] * train_proportion))
    dev_index = int(np.floor(dataset.shape[0] * (train_proportion - dev_proportion)))
    
    # split dataset to tarin/dev/test set
    train_data, train_label = dataset[:train_index, :,:,:], outcomes[:train_index, :]
    dev_data, dev_label = dataset[dev_index:train_index, :,:,:], outcomes[dev_index:train_index, :]
    test_data, test_label = dataset[train_index:, :,:,:], outcomes[train_index:, :]   
    
    # ndarray to tensor
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    dev_data, dev_label = torch.Tensor(dev_data), torch.Tensor(dev_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    # tensor to dataset
    train_dataset = TensorDataset(train_data, train_label)
    dev_dataset = TensorDataset(dev_data, dev_label)
    test_dataset = TensorDataset(test_data, test_label)
    
    # dataset to dataloader 
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    # train_dataloader = DataLoader(train_dataset, batch_size=2)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=2)
    # test_dataloader = DataLoader(test_dataset, batch_size=2)
    
    # # dataset to dataloader 
    # train_dataloader = DataLoader(train_dataset)
    # dev_dataloader = DataLoader(dev_dataset)
    # test_dataloader = DataLoader(test_dataset)
    
    print("train_data.shape : {}\t train_label.shape : {}".format(train_data.shape, train_label.shape))
    print("dev_data.shape : {}\t dev_label.shape : {}".format(dev_data.shape, dev_label.shape))
    print("test_data.shape : {}\t test_label.shape : {}".format(test_data.shape, test_label.shape))
    
    return train_dataloader, dev_dataloader, test_dataloader


