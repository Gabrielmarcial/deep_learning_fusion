import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score

def Test(model, patch_test):
    result = model.predict(patch_test)
    predicted_classes = np.argmax(result, axis=-1)
    return predicted_classes

def compute_metrics(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    f1score = 100*f1_score(true_labels, predicted_labels, average=None)
    recall = 100*recall_score(true_labels, predicted_labels, average=None)
    precision = 100*precision_score(true_labels, predicted_labels, average=None)
    return accuracy, f1score, recall, precision

def show_graph_loss_accuracy(history,accuracy_position,filename):
    plt.rcParams['axes.facecolor']='white'
    plt.figure(figsize=(14,6))
    config = [ { 'title': 'model accuracy', 'ylabel': 'accuracy', 'legend_position': 'upper left', 'index_position': accuracy_position },
               { 'title': 'model loss', 'ylabel': 'loss', 'legend_position': 'upper right', 'index_position': 0 } ]
    for i in range(len(config)):
        plot_number = 120 + (i+1)
        plt.subplot(plot_number)
        plt.plot(history[0,:,0,config[i]['index_position']])
        plt.plot(history[1,:,0,config[i]['index_position']])
        plt.title(config[i]['title'])
        plt.ylabel(config[i]['ylabel'])
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc=config[i]['legend_position'])
        plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_pred(reference, y, h, filename):
    if type(reference) == str:
        gt = np.load(reference)
    else:
        gt = reference
    fig1, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(14, 10))
    ax1.imshow( y, cmap='gray', vmin=0, vmax=2 )
    ax1.set_title('Prediction (y)', fontsize=20)
    ax1.axis('off')
    ax2.imshow( gt, cmap='gray', vmin=0, vmax=2 )
    ax2.set_title('Ground truth (gt)', fontsize=20)
    ax2.axis('off')
    ax3.imshow( gt-y, cmap='gray', vmin=-1, vmax=1 )
    ax3.set_title('Diff', fontsize=20)
    ax3.axis('off')
    ax4.imshow( np.concatenate((h,(np.expand_dims(y, axis=-1)).astype(h.dtype)),axis=-1))
    ax4.set_title(' y selection', fontsize=20)
    ax4.axis('off')
    ax5.imshow( np.concatenate((h,(np.expand_dims(gt,axis=-1)).astype(h.dtype)),axis=-1))
    ax5.set_title('gt selection', fontsize=20)
    ax5.axis('off')
    ax6.imshow( h )
    ax6.set_title('Net input', fontsize=20)
    ax6.axis('off')
    fig1.savefig(filename)
    plt.close()

def plot_cm(cm, filename):
    # Configurando o gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    # Adicionando a barra de cores
    fig.colorbar(cax)
    # Adicionando rótulos
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground truth')
    ax.set_title('Confusion Matrix')
    # Adicionando anotações
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center', color='black')
    # Adicionando os rótulos dos eixos
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    plt.savefig(filename)
    plt.close()

def load_dataset(img_pattern_path, grdt_pattern_path):


if __name__ == '__main__':
    # Defining data path
    root_path = "../"

    # Test and validation proportion
    vald_prop = .2

    # Load dataset
    x, y = [], []
    for file in glob.glob(root_path+'data/imgs/*.npy'):
        img = np.load(file)
        norm = img / 255
        x.append(norm)
    for file in glob.glob(root_path+'data/label/*.npy'):
        label = np.load(file)
        y.append(label)
    dataset = []
    for i in range(len(x)):
        if x[i].shape == (128,128,4) and y[i].shape == (128,128):
            dataset.append([x[i],y[i]])
    del x, y
    valid_start_idx = int((1-vald_prop)*len(dataset))
    train = dataset[:valid_start_idx]
    valid = dataset[valid_start_idx:]
    del dataset
    def get_tensors(dataset):
        t1, t2 = [], []
        for data in dataset:
            t1.append(data[0])
            t2.append(data[1])
        return np.array(t1), np.stack([1-np.array(t2),np.array(t2)],axis=-1)
    x_train, y_train = get_tensors(train)
    x_valid, y_valid = get_tensors(valid)
    del train, valid
    print(f'Dataset is loaded:\nx_train={x_train.shape}\ny_train={y_train.shape}\nx_valid={x_valid.shape}\ny_valid={y_valid.shape}')
    
    # Load the model
    model = tf.keras.models.load_model('../trained/best_model_gm3.keras')

    # Test the model over training and validation data
    pred_train = Test(model, x_train)
    pred_valid = Test(model, x_valid)
    #for i in range(pred_train.shape[0]):
    #    plot_pred(y_train[i,:,:,1], pred_train[i,:,:], x_train[i,:,:,:3], 
    #              f"../data/predict_vs_gt_learned/prediction_{i}.png")
    #for i in range(pred_valid.shape[0]):
    #    plot_pred(y_valid[i,:,:,1], pred_valid[i,:,:], x_valid[i,:,:,:3], 
    #              f"../data/predict_vs_gt_model_validation/prediction_{valid_start_idx+i}.png")

    # Showing the confusion matrix and accuracy metrics for the training data
    print('========================')
    print('Training data evaluation')
    print('========================')
    true_labels = np.reshape(y_train[:,:,:,1], (y_train.shape[0]* y_train.shape[1]*y_train.shape[2]))
    predicted_labels = np.reshape(pred_train, (pred_train.shape[0]* pred_train.shape[1]*pred_train.shape[2]))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print('Confusion  matrix')
    print('=================')
    print(cm)
    plot_cm(cm,'../train_cm.png')

    # Metrics
    metrics = compute_metrics(true_labels, predicted_labels)
    print('\nMetrics')
    print('=======')
    print('Accuracy: ', metrics[0])
    print('F1score: ', metrics[1][1])
    print('Recall: ', metrics[2][1])
    print('Precision: ', metrics[3][1])
    
    # Showing the confusion matrix and accuracy metrics for the validation data
    print('==========================')
    print('Validation data evaluation')
    print('==========================')
    true_labels = np.reshape(y_valid[:,:,:,1], (y_valid.shape[0]* y_valid.shape[1]*y_valid.shape[2]))
    predicted_labels = np.reshape(pred_valid, (pred_valid.shape[0]* pred_valid.shape[1]*pred_valid.shape[2]))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print('Confusion  matrix')
    print('=================')
    print(cm)
    plot_cm(cm,'../valid_cm.png')

    # Metrics
    metrics = compute_metrics(true_labels, predicted_labels)
    print('\nMetrics')
    print('=======')
    print('Accuracy: ', metrics[0])
    print('F1score: ', metrics[1][1])
    print('Recall: ', metrics[2][1])
    print('Precision: ', metrics[3][1])
    
    # Freeing up memory
    del x_train, y_train, x_valid, y_valid, pred_train, pred_valid, true_labels, predicted_labels
    
    # Load dataset
    x, y = [], []
    for file in glob.glob(root_path+'data/test_imgs/*.npy'):
        img = np.load(file)
        norm = img / 255
        x.append(norm)
    for file in glob.glob(root_path+'data/test_grdt/*.npy'):
        label = np.load(file)
        y.append(label)
    dataset = []
    for i in range(len(x)):
        if x[i].shape == (128,128,4) and y[i].shape == (128,128):
            dataset.append([x[i],y[i]])
    del x, y
    x_test, y_test = get_tensors(dataset)
    del dataset
    print(f'\n\nTestset is loaded:\nx_test={x_test.shape}\ny_test={y_test.shape}')

    # Test the model over test patches
    pred_test = Test(model, x_test)
    
    # Showing the confusion matrix and accuracy metrics for the test data (patches)
    print('=======================')
    print('Testing data evaluation')
    print('=======================')
    true_labels = np.reshape(y_test[:,:,:,1], (y_test.shape[0]* y_test.shape[1]*y_test.shape[2]))
    predicted_labels = np.reshape(pred_test, (pred_test.shape[0]* pred_test.shape[1]*pred_test.shape[2]))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print('Confusion  matrix')
    print('=================')
    print(cm)
    plot_cm(cm,'../test_cm.png')

    # Metrics
    metrics = compute_metrics(true_labels, predicted_labels)
    print('\nMetrics')
    print('=======')
    print('Accuracy: ', metrics[0])
    print('F1score: ', metrics[1][1])
    print('Recall: ', metrics[2][1])
    print('Precision: ', metrics[3][1])
    
    # Freeing up memory
    del x_test, y_test, pred_test, true_labels, predicted_labels
    
    
