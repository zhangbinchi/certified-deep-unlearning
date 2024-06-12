import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from sklearn.metrics import f1_score
from utils import load_data, load_dataset, load_unlearn_data
from model import MLP, AllCNN, ResNet18
from matplotlib import pyplot as plt
import os


def test(dataloader, model, name, device):
    criterion = nn.CrossEntropyLoss()
    loss = 0
    correct = 0
    total = 0
    pred_test = []
    label_test = []
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_test.append(predicted)
            label_test.append(labels)
    pred_test = torch.cat(pred_test, 0)
    label_test = torch.cat(label_test, 0)
    f1 = f1_score(label_test.detach().cpu().numpy(), pred_test.detach().cpu().numpy(), average='micro')
    print(f"{name} Loss: {loss / len(dataloader):.4f}, {name} Accuracy: {100.0 * correct / total:.2f}%, {name} Micro F1: {100.0 * f1:.2f}%")


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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # Set the random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    if not os.path.exists('./model'):
        os.mkdir('./model')

    unl_set, res_set = load_unlearn_data(args.dataset, args.num_unlearn)
    train_set, test_set = load_dataset(args.dataset)
    num_classes = 10
    args.num_classes = num_classes

    if args.model == 'mlp':
        model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        unlearn_model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        retrain_model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'cnn':
        model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        unlearn_model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        retrain_model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        unlearn_model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        retrain_model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)

    PATH_unl = ''
    PATH_retrain = './model/'+f"clip_{str(args.C).replace('.','_')}_"+args.model+'_'+args.dataset+f"_lr_{str(args.lr).replace('.','_')}"+f"_bs_{str(args.batch_size)}"+f"_wd_{str(args.weight_decay).replace('.','_')}"+f"_epoch_{str(args.epochs)}"+f"_unlearn_{str(args.num_unlearn)}"+f"_seed_{str(args.seed)}"+'.pth'
    PATH = './model/'+f"clip_{str(args.C).replace('.','_')}_"+args.model+'_'+args.dataset+f"_lr_{str(args.lr).replace('.','_')}"+f"_bs_{str(args.batch_size)}"+f"_wd_{str(args.weight_decay).replace('.','_')}"+f"_epoch_{str(args.epochs)}"+f"_seed_{str(args.seed)}"+'.pth'
    
    model.load_state_dict(torch.load(PATH))
    unlearn_model.load_state_dict(torch.load(PATH_unl))
    retrain_model.load_state_dict(torch.load(PATH_retrain))

    norm_dist = torch.linalg.vector_norm(nn.utils.parameters_to_vector(model.parameters()) - nn.utils.parameters_to_vector(unlearn_model.parameters()))
    norm_dist_rt = torch.linalg.vector_norm(nn.utils.parameters_to_vector(retrain_model.parameters()) - nn.utils.parameters_to_vector(unlearn_model.parameters()))
    print(f"original distance: {norm_dist}, retrain distance: {norm_dist_rt}")

    unl_loader = torch.utils.data.DataLoader(unl_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    res_loader = torch.utils.data.DataLoader(res_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print('Original Model===')
    test(unl_loader, model, 'Unlearn', args.device)
    test(res_loader, model, 'Residual', args.device)
    test(test_loader, model, 'Test', args.device)

    print('Unlearn Model===')
    test(unl_loader, unlearn_model, 'Unlearn', args.device)
    test(res_loader, unlearn_model, 'Residual', args.device)
    test(test_loader, unlearn_model, 'Test', args.device)
