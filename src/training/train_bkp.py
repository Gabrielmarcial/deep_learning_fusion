import os, json, gc
import random, glob
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from src.models.resunet import ResUNet
from src.data.preprocess import *
from src.data.split_data import *
from src.evaluation.evaluate import *

def orthogonal_rot(image):
    return np.rot90(image, np.random.choice([-1, 0, 1]))

def get_batch_samples(x, y, batch, batch_size, data_augmentation, number_samples_for_generator, datagen):
    if data_augmentation:
        x_batch = x[batch * number_samples_for_generator : (batch + 1) * number_samples_for_generator, : , : , :]
        y_batch = y[batch * number_samples_for_generator : (batch + 1) * number_samples_for_generator, : , : , :]
        x_iterator = datagen[0].flow(x_batch, seed=batch)
        y_iterator = datagen[1].flow(y_batch, seed=batch)
        x_batch = np.array([next(x_iterator)[0] for _ in range(batch_size)])
        y_batch = np.array([next(y_iterator)[0] for _ in range(batch_size)])
    else:
        x_batch = x[batch * batch_size : (batch + 1) * batch_size, : , : , :]
        y_batch = y[batch * batch_size : (batch + 1) * batch_size, : , : , :]
    return x_batch, y_batch

def set_number_of_batches(qt_train_samples, qt_valid_samples, batch_size, data_augmentation, number_samples_for_generator=6):
    if data_augmentation:
        train_batches_qtd = qt_train_samples // number_samples_for_generator
        valid_batches_qtd = qt_valid_samples // number_samples_for_generator
    else:
        train_batches_qtd = qt_train_samples // batch_size
        valid_batches_qtd = qt_valid_samples // batch_size
    return train_batches_qtd, valid_batches_qtd

def load_history(filename):
    history = {"tloss": [], "tacc": [], "vloss": [], "vacc": []}
    # Load previous history if exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            history = json.load(f)
    return history

def save_history(filename, data, append = False):
    # make dict with history
    history = {"tloss": [], "tacc": [], "vloss": [], "vacc": []}
    for i, arrays in enumerate(data):
        for array in arrays:
            if i == 0:
                history["tloss"].append(array[0,0])
                history["tacc"].append(array[0,1])
            elif i == 1:
                history["vloss"].append(array[0,0])
                history["vacc"].append(array[0,1])
    # Load previous history if exists and concatenate
    if append and os.path.exists(filename):
        with open(filename, 'r') as f:
            previous_history = json.load(f)
        for key in history:
            history[key] = previous_history.get(key, []) + history[key]
    # Save updated history
    with open(filename, 'w') as f:
        json.dump(history, f)
    return history

def decode_hist(data):
    train_list = []
    val_list = []
    tloss = data["tloss"]
    tacc = data["tacc"]
    vloss = data["vloss"]
    vacc = data["vacc"]
    try:
        for i in range(len(tloss)):
            train_list.append(np.array([[tloss[i], tacc[i]]]))
            val_list.append(np.array([[vloss[i], vacc[i]]]))
        return [train_list, val_list]
    except:
        return [[],[]]

def train_net(net, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, filepath, filename, data_augmentation=False, number_samples_for_generator=1, history_file = None):
    print(f'Start the training on Tensorflow {tf.__version__} and Keras {tf.keras.__version__}...')

    # calculating number of batches
    train_batchs_qtd, valid_batchs_qtd = set_number_of_batches(x_train.shape[0], x_valid.shape[0], batch_size, data_augmentation, number_samples_for_generator)

    # restore history if available
    history = [[],[]]
    if history_file:
        history = decode_hist(load_history(history_file))
    history_train, history_valid = history
    valid_loss_best_model = float('inf')
    no_improvement_count = 0

    # Config tranformations for image patches in data augmentation
    datagen_args = dict(
            preprocessing_function=orthogonal_rot,
            horizontal_flip=True, # horizontal flip
            vertical_flip=True, # vertical flip
            fill_mode='constant',
            cval=0
    )
    datagen = (tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args),
               tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args))

    for epoch in range(epochs):
        print('Start epoch ... %d ' %(epoch+1) )
        # shuffle train set
        x_train, y_train = shuffle(x_train, y_train, random_state = 0)
        #x_train = tf.convert_to_tensor(x_train)
        #y_train = tf.convert_to_tensor(y_train)
        #x_valid = tf.convert_to_tensor(x_valid)
        #y_valid = tf.convert_to_tensor(y_valid)

        # TRAINING
        train_loss = np.zeros((1, 2))
        # mini batches strategy
        for  batch in range(train_batchs_qtd):
            print('Start batch ... %d ' %(batch+1) , flush = True)
            x_train_batch, y_train_batch = get_batch_samples(x_train, y_train, batch, batch_size, data_augmentation, number_samples_for_generator, datagen)
            train_loss += net.train_on_batch(x_train_batch, y_train_batch)

        # Estimating the loss in the training set
        train_loss /= train_batchs_qtd

        # VALIDATING
        valid_loss = np.zeros((1, 2))
        # Evaluating the network (model) with the validation set
        for  batch in range(valid_batchs_qtd):
            x_valid_batch, y_valid_batch = get_batch_samples(x_valid, y_valid, batch, batch_size, data_augmentation, number_samples_for_generator, datagen)
            valid_loss += net.test_on_batch(x_valid_batch, y_valid_batch)

        # Estimating the loss in the validation set
        valid_loss /= valid_batchs_qtd

        # Showing the results.
        print("Epoch %d/%d [training loss: %f , Train acc.: %.2f%%][Validation loss: %f , Validation acc.:%.2f%%]" %(epoch+1, epochs, train_loss[0 , 0], 100*train_loss[0 , 1] , valid_loss[0 , 0] , 100 * valid_loss[0 , 1]))
        
        history_train.append( train_loss )
        history_valid.append( valid_loss )
        
        # Early Stopping
        if (1-(valid_loss[0 , 0]/valid_loss_best_model)) < early_stopping_delta:
            if no_improvement_count+1 >= early_stopping_epochs:
                print('Early Stopping reached')
                break
            else:
                no_improvement_count = no_improvement_count+1
        else:
            valid_loss_best_model = valid_loss[0 , 0]
            no_improvement_count = 0

            # Saving best model
            print("Saving the model...")
            net.save(filepath+filename+'.keras')
            del net
            gc.collect()
            net = tf.keras.models.load_model(filepath+filename+'.keras', compile=True)

            # Save and history updated
            history = [ history_train, history_valid ]
            if history_file:
                updated = save_history(history_file, history)
                history = decode_hist(updated)

    return history

if __name__ == '__main__':
    # Defining number of classes and class indices in relation to reference images (RGB)
    n_classes = 2

    # Defining data path
    root_path = "../"

    # Test and validation proportion
    vald_prop = .25

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
    random.shuffle(train)
    random.shuffle(valid)
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

    # shape of the input to the network (training)
    input_shape = (128,128,4)

    # Defining hyperparameters for training
    batch_size = 32
    epochs = 16
    class_weights = [0.30, 1.00]
    early_stopping_epochs = 8
    early_stopping_delta = 0.0001 # delta improvement equivalent to 0.01%
    data_augmentation = True
    number_samples_for_generator = 8

    # optimizer
    adam = tf.keras.optimizers.Adam(learning_rate = 0.0001 , beta_1=0.9)

    # Train the model
    best_model_filename = 'trained/best_model'

    if os.path.exists(root_path+best_model_filename+'.keras'):
        # Load previous model
        model = tf.keras.models.load_model(root_path+best_model_filename+'.keras', compile=False)
    else:
        # Create the model
        model = ResUNet(input_shape, n_classes)
        
    # Summary the model
    #model.summary()

    # Compile the model
    model.compile(loss = "binary_crossentropy", optimizer=adam , metrics=['accuracy'])#, loss_weights=class_weights)
    
    #history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, root_path, best_model_filename, data_augmentation, number_samples_for_generator)
    # Uncomment the line above for training with the whole dataset!!!
    history = train_net(model, x_train[0:512], y_train[0:512], x_valid[0:128], y_valid[0:128], batch_size, epochs, early_stopping_epochs, early_stopping_delta, root_path, best_model_filename, data_augmentation, number_samples_for_generator, "../history.json")

    # Save final history
    show_graph_loss_accuracy(np.asarray(history),1,root_path+'train_history.png')

