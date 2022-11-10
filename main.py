import torch
from tensorboardX import SummaryWriter
from data_pro import generate_dataloader
from matplotlib import pyplot as plt
from RES_model import ComplexConvNeuralNetwork
from model_evaluate import accuracy,precision
def train(epoch):
    loss_total = 0
    for i,(inputs, labels) in enumerate(train_data, 0):
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    writer.add_scalar('loss', loss_total / batch_size, epoch + 1)
    return loss_total/batch_size


def test(epoch):
    total_number = 0
    correct_number = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data, 0):
            y_pred = model(inputs)
            _, prediction = torch.max(y_pred, dim=1)
            total_number+=labels.size(0)
            correct_number+=sum((1 * (prediction == labels))).item()
    accuracy = correct_number / total_number * 100
    writer.add_scalar('accuracy', accuracy, epoch + 1)

    if epoch == epochs - 1:
        mat = confusion_matrix()
        precision_value = precision(mat)
        show_result(precision_value)
    return accuracy

def confusion_matrix():
    con_mat = {'T0': {'T0': 0, 'T1': 0, 'T2': 0},
               'T1': {'T0': 0, 'T1': 0, 'T2': 0},
               'T2': {'T0': 0, 'T1': 0, 'T2': 0}}
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data, 0):
            y_pred = model(inputs)
            _, prediction = torch.max(y_pred, dim=1)
            for j in range (0, batch_size):
                actual = classes[labels[j].item()]
                predict = classes[prediction[j].item()]
                con_mat[actual][predict] = con_mat[actual][predict] + 1
    return con_mat


def show_result(precision):
    print('\t\t', 'precision')
    for item in classes:
        print(item, '\t', precision[item])


if __name__ == '__main__':
    file_path_train = './data/train.csv'
    file_path_test = './data/test.csv'
    batch_size = 32
    epochs = 30
    classes = ['T0', 'T1', 'T2']

    model = ComplexConvNeuralNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    writer = SummaryWriter('./data_record/model')
    dummy_input = torch.rand(batch_size, 64)
    writer.add_graph(model, (dummy_input,))

    train_data, test_data = generate_dataloader(file_path_train, file_path_test, size=batch_size)

    loss_list = []
    acc_list = []
    model.train()
    for i in range(epochs):
        model.train()
        loss_list.append(train(i))
        model.eval()
        acc_list.append(test(i))

    plt.figure()
    ax1 = plt.subplot(211)
    ax1.plot(loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    ax2 = plt.subplot(212)
    ax2.plot(acc_list)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()