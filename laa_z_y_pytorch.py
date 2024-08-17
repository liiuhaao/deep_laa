import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import deep_laa_support as dls
import scipy.sparse
import matplotlib.pyplot as plt

# 加载数据
filename = "flower_data"
if filename != "millionaire_non_empty_sparse":
    data_all = np.load(filename+'.npz')
    user_labels = data_all['user_labels']
    label_mask = data_all['label_mask']
    true_labels = data_all['true_labels']
    category_size = data_all['category_num']
    source_num = data_all['source_num']
    n_samples, _ = np.shape(true_labels)
else:
    data_all = scipy.io.loadmat(filename+'.mat')
    user_labels = data_all['user_labels']
    label_mask = data_all['label_mask']
    true_labels = data_all['true_labels']
    category_size = data_all['category_num'][0,0]
    source_num = data_all['source_num'][0,0]
    n_samples, _ = np.shape(true_labels)

mv_y = dls.get_majority_y(user_labels, source_num, category_size)

input_size = source_num * category_size
batch_size = n_samples

n_z = 2  # number of latent aspects
flag_deep_z = False

# 添加的模型
class Encoder(nn.Module):
    def __init__(self, input_size, n_z):
        super(Encoder, self).__init__()
        if not flag_deep_z:
            self.fc = nn.Linear(input_size, n_z)
        else:
            n_hz = 10
            self.fc1 = nn.Linear(input_size, n_hz)
            self.fc2 = nn.Linear(n_hz, n_z)

    def forward(self, x):
        if not flag_deep_z:
            z = torch.nn.functional.softplus(self.fc(x))
        else:
            hz = torch.nn.functional.softplus(self.fc1(x))
            z = torch.nn.functional.softplus(self.fc2(hz))
        return z

class Classifier(nn.Module):
    def __init__(self, input_size, n_z, category_size):
        super(Classifier, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(input_size, category_size) for _ in range(n_z)])

    def forward(self, x, z):
        tmp_y = [self.fc[i](x) for i in range(n_z)]
        tmp_y_2 = sum(tmp_y[i] * z[:, i].unsqueeze(1) for i in range(n_z))
        y = torch.nn.functional.softmax(tmp_y_2, dim=1)
        return y

class Decoder(nn.Module):
    def __init__(self, category_size, input_size, n_z):
        super(Decoder, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(category_size, input_size) for _ in range(n_z)])

    def forward(self, y, z):
        tmp_x = [self.fc[i](y) for i in range(n_z)]
        tmp_x_2 = sum(tmp_x[i] * z[:, i].unsqueeze(1) for i in range(n_z))
        x_reconstr = torch.exp(tmp_x_2) / torch.matmul(torch.exp(tmp_x_2), source_wise_template)
        return x_reconstr

# 模型初始化
encoder = Encoder(input_size, n_z)
classifier = Classifier(input_size, n_z, category_size)
decoder = Decoder(category_size, input_size, n_z)

criterion_classifier_x_y = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()

optimizer_classifier_x_y = optim.Adam(classifier.parameters(), lr=0.001)
optimizer_classifier = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()) + list(decoder.parameters()), lr=0.001)

# 训练分类器
epochs = 50
for epoch in range(epochs):
    total_hit = 0
    for batch in range(n_samples // batch_size):
        batch_x = torch.tensor(user_labels, dtype=torch.float32)
        batch_mask = torch.tensor(label_mask, dtype=torch.float32)
        batch_y_label = torch.tensor(true_labels, dtype=torch.long).squeeze()
        batch_majority_y = torch.tensor(mv_y, dtype=torch.float32)

        # 前向
        z = encoder(batch_x)
        y = classifier(batch_x, z)
        loss_classifier_x_y = criterion_classifier_x_y(y, batch_y_label)
        
        # 后向以及优化
        optimizer_classifier_x_y.zero_grad()
        loss_classifier_x_y.backward()
        optimizer_classifier_x_y.step()

        # 准确率计算
        _, predicted = torch.max(y, 1)
        total_hit += (predicted == batch_y_label).sum().item()

    accuracy = total_hit / n_samples
    print(f"epoch: {epoch} accuracy: {accuracy}")

# 网络计算
epochs = 500
for epoch in range(epochs):
    total_hit = 0
    for batch in range(n_samples // batch_size):
        batch_x = torch.tensor(user_labels, dtype=torch.float32)
        batch_mask = torch.tensor(label_mask, dtype=torch.float32)
        batch_y_label = torch.tensor(true_labels, dtype=torch.long).squeeze()
        batch_majority_y = torch.tensor(mv_y, dtype=torch.float32)

        # 前向
        z = encoder(batch_x)
        y = classifier(batch_x, z)
        y_prior = batch_majority_y

        # 计算损失
        reconstr_x = decoder(y, z)
        loss_classifier_y_x = criterion_mse(y, reconstr_x)
        loss_y_kl = torch.mean(torch.sum(y * (torch.log(y + 1e-10) - torch.log(y_prior + 1e-10)), dim=1))
        loss_classifier = loss_classifier_y_x + 0.0001 * loss_y_kl

        # 后向以及优化
        optimizer_classifier.zero_grad()
        loss_classifier.backward()
        optimizer_classifier.step()

        # 计算准确率
        _, predicted = torch.max(y, 1)
        total_hit += (predicted == batch_y_label).sum().item()

    accuracy = total_hit / n_samples
    print(f"epoch: {epoch} accuracy: {accuracy}")

print("Done!")
