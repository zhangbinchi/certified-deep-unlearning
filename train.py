import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from utils import load_data, load_dataset
from model import MLP, AllCNN, ResNet18, ResNet50
import os


def train(train_loader, val_loader, model, lr, wd, num_epochs, max_norm, path, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    min_val_loss = 1e10
    for epoch in range(num_epochs):
        # Training phase
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(data)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Perform backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)

            param_norm = nn.utils.parameters_to_vector(model.parameters()).norm()
            if param_norm > max_norm:
                scale_factor = max_norm / param_norm
                for param in model.parameters():
                    param.data *= scale_factor

        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()  # Set the model in evaluation mode
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
            
        val_loss /= len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), path)
        
        # Print the validation loss and accuracy for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    print('Finished Training')


def test(testloader, model, device):
    criterion = nn.CrossEntropyLoss()
    loss = 0
    correct = 0
    total = 0
    pred_test = []
    label_test = []
    with torch.no_grad():
        for data, labels in testloader:
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
    print(f"Test Loss: {loss / len(testloader):.4f}, Test Accuracy: {100.0 * correct / total:.2f}%, Test Micro F1: {100.0 * f1:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--mlp-layer', type=int, default=3, metavar='N',
                        help='number of layers of MLP (default: 3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--C', type=float, default=20,
                        help='Norm constraint of parameters')
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

    PATH = './model/'+f"clip_{str(args.C).replace('.','_')}_"+args.model+'_'+args.dataset+f"_lr_{str(args.lr).replace('.','_')}"+f"_bs_{str(args.batch_size)}"+f"_wd_{str(args.weight_decay).replace('.','_')}"+f"_epoch_{str(args.epochs)}"+f"_seed_{str(args.seed)}"+'.pth'

    trainloader, valloader, testloader = load_data(args.dataset, args.batch_size, args.seed)
    train_set, _ = load_dataset(args.dataset)
    num_classes = 10
    args.num_classes = num_classes

    if args.model == 'mlp':
        model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'cnn':
        model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet50':
        model = ResNet50(num_classes=args.num_classes).to(args.device)

    train(trainloader, valloader, model, args.lr, args.weight_decay, args.epochs, args.C, PATH, args.device)
    model.load_state_dict(torch.load(PATH))
    test(testloader, model, args.device)
    