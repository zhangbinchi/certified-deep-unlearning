import argparse
import numpy as np
import torch
from utils import load_data, load_dataset, load_unlearn_data
from model import MLP, AllCNN, ResNet18
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


class DT:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class RF:
    def __init__(self, min_samples_leaf=30):
        self.model = RandomForestClassifier(random_state=0, n_estimators=500, min_samples_leaf=min_samples_leaf)

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class MLP_INF:
    def __init__(self):
        self.model = MLPClassifier(early_stopping=True, learning_rate_init=0.01)

    def scaler_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class LR:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=400, multi_class='ovr', n_jobs=1)

    def train_model(self, train_x, train_y):
        self.scaler = preprocessing.StandardScaler().fit(train_x)
        # temperature = 1
        # train_x /= temperature
        self.model.fit(self.scaler.transform(train_x), train_y)

    def predict_proba(self, test_x):
        self.scaler = preprocessing.StandardScaler().fit(test_x)
        return self.model.predict_proba(self.scaler.transform(test_x))

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(self.scaler.transform(test_x))

        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(self.scaler.transform(test_x))
        return roc_auc_score(test_y, pred_y[:, 1])  # binary class classification AUC
        # return roc_auc_score(test_y, pred_y[:, 1], multi_class="ovr", average=None)  # multi-class AUC


def posterior(dataloader, model, device):
    posterior_list = []
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            post = torch.softmax(outputs, 1)
            posterior_list.append(post)
    return torch.cat(posterior_list, 0)


def construct_feature(post_ori, post_unl, method):
    if method == "direct_diff":
        return post_ori - post_unl

    elif method == "sorted_diff":
        for index, posterior in enumerate(post_ori):
            sort_indices = np.argsort(posterior)
            post_ori[index] = posterior[sort_indices]
            post_unl[index] = post_unl[index][sort_indices]
        return post_ori - post_unl

    elif method == "l2_distance":
        feat = torch.ones(post_ori.shape[0])
        for index in range(post_ori.shape[0]):
            euclidean = distance.euclidean(post_ori[index], post_unl[index])
            feat[index] = euclidean
        return feat.unsqueeze(1)

    elif method == "direct_concat":
        return torch.cat([post_ori, post_unl], 1)

    elif method == "sorted_concat":
        for index, posterior in enumerate(post_ori):
            sort_indices = np.argsort(posterior)
            post_ori[index] = posterior[sort_indices]
            post_unl[index] = post_unl[index][sort_indices]
        return torch.cat([post_ori, post_unl], 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--epochs', type=int, default=31, metavar='N',
                        help='number of epochs to train (default: 31)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--C', type=float, default=20,
                        help='Norm constraint of parameters')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--mlp-layer', type=int, default=3, metavar='N',
                        help='number of layers of MLP (default: 3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--num-unlearn', default=1000, type=int, help='number of unlearned samples')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--method', type=str, default='sorted_concat', help='Method of Feature Construction')
    parser.add_argument('--attack-model', type=str, default='LR', help='Choice of Attack Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # Set the random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    unl_set, res_set = load_unlearn_data(args.dataset, args.num_unlearn)
    train_set, test_set = load_dataset(args.dataset)
    # num_classes = len(train_set.classes) if args.num_classes is None else args.num_classes
    num_classes = 10
    args.num_classes = num_classes

    if args.model == 'mlp':
        model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        unlearn_model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'cnn':
        model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        unlearn_model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        unlearn_model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)

    PATH_unl = ''
    PATH = './model/'+f"clip_{str(args.C).replace('.','_')}_"+args.model+'_'+args.dataset+f"_lr_{str(args.lr).replace('.','_')}"+f"_bs_{str(args.batch_size)}"+f"_wd_{str(args.weight_decay).replace('.','_')}"+f"_epoch_{str(args.epochs)}"+f"_seed_{str(args.seed)}"+'.pth'
    
    model.load_state_dict(torch.load(PATH))
    unlearn_model.load_state_dict(torch.load(PATH_unl))

    unl_loader = torch.utils.data.DataLoader(unl_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    res_loader = torch.utils.data.DataLoader(res_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    unl_pos_post = posterior(unl_loader, unlearn_model, args.device).detach().cpu()
    unl_neg_post = posterior(res_loader, unlearn_model, args.device).detach().cpu()
    ori_pos_post = posterior(unl_loader, model, args.device).detach().cpu()
    ori_neg_post = posterior(res_loader, model, args.device).detach().cpu()

    feat_pos = construct_feature(ori_pos_post, unl_pos_post, args.method)
    feat_neg = construct_feature(ori_neg_post, unl_neg_post, args.method)

    feat = torch.cat([feat_pos, feat_neg], 0).numpy()
    label = torch.cat([torch.ones(feat_pos.shape[0]), torch.zeros(feat_neg.shape[0])], 0).numpy().astype('int')

    if args.attack_model == 'LR':
        attack_model = LR()
    elif args.attack_model == 'DT':
        attack_model = DT()
    elif args.attack_model == 'MLP':
        attack_model = MLP_INF()
    elif args.attack_model == 'RF':
        attack_model = RF()
    else:
        raise Exception("invalid attack name")
    
    attack_model.train_model(feat, label)

    train_acc = attack_model.test_model_acc(feat, label)
    train_auc = attack_model.test_model_auc(feat, label)
    print(f"Attack Accuracy: {100 * train_acc:.2f}%, Attack AUC: {100 * train_auc:.2f}%")
