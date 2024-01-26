import os
import tarfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
import matplotlib.pyplot as plt
import utils


## returns max, min val feature list among records
def get_data_min_max(records, device):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)

    ## loop for each record_id == patient
    for b, (record_id, tt, vals, mask, lbaels) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        ## extract max min val of each feature
        for i in range(n_features):
            existing_vals = vals[:, i][mask[:, i] == 1]
            if len(existing_vals) != 0:
                batch_max.append(torch.max(existing_vals))
                batch_min.append(torch.min(existing_vals))
            else:
                batch_max.append(-inf)
                batch_min.append(inf)
        batch_max = torch.stack(batch_max)
        batch_min = torch.stack(batch_min)

        if (data_min is None) and (data_max is None):
            data_max = batch_max
            data_min = batch_min
        else:
            data_max = torch.max(data_max, batch_max)
            data_min = torch.min(data_min, batch_min)

    return data_min, data_max


class PhysioNet(object):
    ## time series data
    urls = [
        "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download",
        "https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download",
    ]

    ## outcomes for each pateint
    outcome_urls = ["https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt"]

    params = [
        "Age",
        "Gender",
        "Height",
        "ICUType",
        "Weight",
        "Albumin",
        "ALP",
        "ALT",
        "AST",
        "Bilirubin",
        "BUN",
        "Cholesterol",
        "Creatinine",
        "DiasABP",
        "FiO2",
        "GCS",
        "Glucose",
        "HCO3",
        "HCT",
        "HR",
        "K",
        "Lactate",
        "Mg",
        "MAP",
        "MechVent",
        "Na",
        "NIDiasABP",
        "NIMAP",
        "NISysABP",
        "PaCO2",
        "PaO2",
        "pH",
        "Platelets",
        "RespRate",
        "SaO2",
        "SysABP",
        "Temp",
        "TroponinI",
        "TroponinT",
        "Urine",
        "WBC",
    ]

    # key : params, value: index of parmams
    params_dict = {k: i for i, k in enumerate(params)}

    labels = ["SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"]

    # key : labels, value: index of labels
    labels_dict = {k: i for i, k in enumerate(labels)}

    # initializing
    def __init__(
        self,
        root,
        train=True,
        download=False,
        quantization=0.1,
        n_samples=None,
        device=torch.device("cpu"),
    ):
        self.root = root  # data path
        self.train = train  # training set use or not
        self.device = device  # device to load data
        self.reduce = "average"  #같은 시간대의 같은 특성은 평균 값 사용
        self.quantization = quantization  # time units

        # download if there is no data
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("No data set. Set download = True")

        # set appropriate file as data_file
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        # load data and labels on appropriate device
        if self.device == "cpu":
            self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location="cpu")
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file), map_location="cpu")
        else:
            self.data = torch.load(os.path.join(self.processed_folder, data_file))
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))

        # truncate samples up to n_samples
        if n_samples is not None:
            self.data = self.data[:n_samples]
            # self.labels = self.labels[:n_samples]

    def download(self):
        if self._check_exists():
            return

        # if folder exists pass else make dir
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download outcome data
        for url in self.outcome_urls:
            # 뒤에서부터 찾아서 / 기준으로 3개의 구역으로 분리
            filename = url.rpartition("/")[2]
            # Download a file from a url and place it in root.
            download_url(url, self.raw_folder, filename, None)

            txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                # 데이터 시작 지점부터 읽음
                for l in lines[1:]:
                    l = l.rstrip().split(",")
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels).to(self.device)

                torch.save(
                    outcomes,  ##labels
                    os.path.join(self.processed_folder, filename.split(".")[0] + ".pt"),
                )

        for url in self.urls:
            filename = url.rpartition("/")[2]
            download_url(url, self.raw_folder, filename, None)
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            # raw 폴더에 압축 해제
            tar.extractall(self.raw_folder)
            tar.close()

            print(f"Processing {filename}")

            dirname = os.path.join(self.raw_folder, filename.split(".")[0])  # 그냥 이름이 set-a이렇게 생김
            patients = []
            total = 0
            for txtfile in os.listdir(dirname):
                record_id = txtfile.split(".")[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.0]  # time trace
                    vals = [torch.zeros(len(self.params)).to(self.device)]  # 관측값
                    mask = [torch.zeros(len(self.params)).to(self.device)]  # 존재여부
                    nobs = [torch.zeros(len(self.params))]  # 관측 횟수
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(",")
                        # hours
                        time = float(time.split(":")[0]) + float(time.split(":")[1]) / 60.0
                        # 계산 속도 향상을 위한 시간 단위 부여
                        time = round(time / self.quantization) * self.quantization

                        # 매 시간 마다 각 변수에 새로운 값 할당
                        if time != prev_time:
                            tt.append(time)
                            vals.append(torch.zeros(len(self.params)).to(self.device))
                            mask.append(torch.zeros(len(self.params)).to(self.device))
                            nobs.append(torch.zeros(len(self.params)).to(self.device))
                            prev_time = time

                        if param in self.params_dict:
                            n_observations = nobs[-1][self.params_dict[param]]
                            # 같은 시간에 한개 이상의 관측값이 있는 경우, 평균값을 사용
                            if self.reduce == "average" and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)

                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        else:
                            assert param == "RecordID", f"Unexpected param {param}"

                tt = torch.tensor(tt).to(self.device)
                vals = torch.stack(vals)
                mask = torch.stack(mask)

                labels = None
                if record_id in outcomes:
                    # for training set take mortality data
                    labels = outcomes[record_id]
                    labels = labels[4]

                # 각 환자 데이터에 대한 정보 저장
                patients.append((record_id, tt, vals, mask, labels))

            torch.save(
                patients,
                os.path.join(
                    self.processed_folder,
                    filename.split(".")[0] + "_" + str(self.quantization) + ".pt",
                ),
            )

        print("Done !")

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition("/")[2]

            if not os.path.exists(
                os.path.join(
                    self.processed_folder,
                    filename.split(".")[0] + "_" + str(self.quantization) + ".pt")
            ):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def training_file(self):
        return "set-a_{}.pt".format(self.quantization)

    @property
    def test_file(self):
        return "set-b_{}.pt".format(self.quantization)

    @property
    def label_file(self):
        return "Outcomes-a.pt"
    # 대괄호를 사용한 인덱싱, 슬라이싱 정의
    def __getitem__(self, index):
        return self.data[index]
    # 객체 길이 정의
    def __len__(self):
        return len(self.data)

    def visualize(self, timesteps, data, mask, plot_name):
        width = 15
        height = 15

        ## 3번이상 입력이 있는 특성 선별, 3이상이 되면 true 아니면 false
        non_zero_attributes = (torch.sum(mask, 0) > 2).numpy()

        non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i]]
        n_non_zero = sum(non_zero_attributes)

        # non_zero_idx를 통해서 3번 이상 등장하는 feature에 대해서 mask와 data(vals 각 feature의 시간에 따른 값들) 생성
        mask = mask[:, non_zero_idx]
        data = data[:, non_zero_idx]

        params_non_zero = [self.params[i] for i in non_zero_idx]
        params_dict = {k: i for i, k in enumerate(params_non_zero)}

        n_col = 3
        n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
        fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecoloer="white")

        for i in range(n_non_zero):
            param = params_non_zero[i]
            param_id = params_dict[param]

            tp_mask = mask[:, param_id].long()

            tp_cur_param = timesteps[tp_mask == 1.0]
            data_cur_param = data[tp_mask == 1.0, param_id]

            ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(), marker="o")
            ax_list[i // n_col, i % n_col].set_title(param)

        fig.tight_layout()
        fig.savefig(plot_name)
        plt.close(fig)


def variable_time_collate_fn(
    batch,
    args,
    device=torch.device("cpu"),
    data_type="train",
    data_min=None,
    data_max=None,
):
    D = batch[0][2].shape[1]            # 변수 개수(특성)
    # @inverse_indices 
    combined_tt, inverse_indices = torch.unique(torch.cat(ex[1] for ex in batch), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)
    
    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    
    combined_labels = None
    N_labels = 1            # label 목록 개수
    
    # nan array
    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        
        if labels is not None:
            labels = labels.to(device)
            
        indices = inverse_indices[offset : offset+len(tt)]
        offset = offset + len(tt)
        
        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask
        
        if labels is not None:
            combined_labels[b] = labels
    
    combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, 
                                                      att_min = data_min, att_max = data_max)
    
    if torch.max(combined_tt) != 0:
        combined_tt = combined_tt / torch.max(combined_tt)
    
    data_dict = {
        "data" : combined_vals,
        "time_steps" : combined_tt,
        "mask" : combined_mask,
        "labels" : combined_labels
    }
    
    # data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
    return data_dict

if __name__ == '__main__':
    torch.manual_seed(1991) 

    dataset = PhysioNet('data/physionet', train=False, download=True)
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn)
    # print(dataloader.__iter__().next())
    # print(len(dataset))