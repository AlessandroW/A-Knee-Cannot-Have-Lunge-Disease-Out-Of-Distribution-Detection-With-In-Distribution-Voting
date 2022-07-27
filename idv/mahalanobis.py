import torch
import numpy as np
from tqdm  import tqdm
from torch.autograd import Variable

def sample_estimator(model, num_classes, num_activations, train_loader, device):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    model.to(device)
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    num_output = 1
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = [[0] * num_classes for i in range(num_output)]
    class_indices = torch.arange(num_classes)

    with torch.no_grad():
        for data, target, idx in tqdm(train_loader):
            # No ensembling here
            data = data[:,0,:,:]
            data = data.to(device)
            out_features = [model.features(data)]

            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

            # construct the sample matrix
            for i in range(data.size(0)):
                # multi-label problem
                for label in class_indices[target[i] == 1]:
                    if num_sample_per_class[label] == 0:
                        out_count = 0
                        for out in out_features:
                            list_features[out_count][label] = out[i].view(1, -1)
                            out_count += 1
                    else:
                        out_count = 0
                        for out in out_features:
                            list_features[out_count][label] \
                            = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                            out_count += 1
                    num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in [num_activations]:
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().to(device)
        precision.append(temp_precision)

    return sample_class_mean, precision


def get_Mahalanobis_score(model, test_loader, num_classes, sample_mean, precision,
                          layer_index, magnitude, device):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []

    for data, target, idx in tqdm(test_loader):
        # No ensembling here
        data = data[:,0,:,:]
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, requires_grad = True), Variable(target)

        out_features = model.features(data)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = torch.sub(out_features.data, batch_sample_mean)
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).to(device),
                     gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.229))
        gradient.index_copy_(1, torch.LongTensor([1]).to(device),
                             gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.224))
        gradient.index_copy_(1, torch.LongTensor([2]).to(device),
                             gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.225))
        tempInputs = torch.add(data.data, -magnitude, gradient)

        noise_out_features = model.features(Variable(tempInputs))
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())


    return Mahalanobis
