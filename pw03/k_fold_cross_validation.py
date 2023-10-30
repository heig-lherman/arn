import numpy as np
import mlp_backprop_momentum as mlp

def split_dataset(dataset, n_parts=5):
    n_rows = dataset.shape[0]
    index_all = np.arange(n_rows)
    np.random.shuffle(index_all)
    parts = []
    current_start = 0
    for p in np.arange(n_parts):
        current_end = current_start + int(np.floor(n_rows / (n_parts-p)))
        parts.append(dataset[index_all[current_start:current_end],:])
        n_rows -= current_end - current_start
        current_start = current_end
    return parts

def compute_confusion_matrix(target, output, threshold):
    """
    This function computes the confusion matrix for a given set of predictions.
    Rows are the actual class and columns are the predicted class
    """
    assert len(target.shape) == 2, "target must be a 2-dimensional array"
    assert len(output.shape) == 2, "output must be a 2-dimensional array"

    if target.shape[1] == 1:
        n_classes = 2
        target_binary = np.concatenate((target > threshold, target <= threshold), axis=1)
    else:
        n_classes = target.shape[1]
        target_binary = target > threshold
        
    if output.shape[1] == 1:
        output_binary = np.concatenate((output > threshold, output <= threshold), axis=1)
    else:
        output_binary = output > threshold

    confusion_matrix = np.zeros((n_classes, n_classes))
    for t in np.arange(n_classes):
        for o in np.arange(n_classes):
            confusion_matrix[t,o] = np.sum(np.logical_and(target_binary[:,t], output_binary[:,o]))
    
    return confusion_matrix

def k_fold_cross_validation(mlp, dataset, k=5, learning_rate=0.01, momentum=0.7, epochs=100, threshold=None):
    MSE_train_mean = 0.0
    MSE_test_mean = 0.0

    parts = split_dataset(dataset, k)
    target_test = []
    output_test = []
    
    for k_i in np.arange(k):
        mlp.init_weights()
        
        training_parts = set(np.arange(k))
        training_parts.remove(k_i)
        dataset_train = np.concatenate([parts[i] for i in list(training_parts)])
        dataset_test = parts[k_i]

        input_data = dataset_train[:,0:mlp.n_inputs]
        output_data = dataset_train[:,mlp.n_inputs:(mlp.n_inputs+mlp.n_outputs)]
        input_data_test = dataset_test[:,0:mlp.n_inputs]
        output_data_test = dataset_test[:,mlp.n_inputs:(mlp.n_inputs+mlp.n_outputs)]
        
        mlp.fit((input_data, output_data),
                learning_rate=learning_rate, momentum=momentum, epochs=epochs)
        MSE_train, _ = mlp.compute_MSE((input_data, output_data))
        MSE_train_mean += MSE_train
        
        MSE_test, temp_out = mlp.compute_MSE((input_data_test, output_data_test))
        MSE_test_mean += MSE_test
        output_test.append(temp_out)
        target_test.append(output_data_test)
    
    target_test = np.concatenate(target_test, axis=0)
    output_test = np.concatenate(output_test, axis=0)
    
    if threshold is None:
        return (MSE_train_mean / k, MSE_test_mean / k)
    else:
        return (MSE_train_mean / k, MSE_test_mean / k, compute_confusion_matrix(target_test, output_test, threshold))
        

def k_fold_cross_validation_per_epoch(mlp, dataset, k=5, learning_rate=0.01, momentum=0.7, epochs=100):
    MSE_train = np.zeros((k, epochs))
    MSE_test = np.zeros((k, epochs))

    parts = split_dataset(dataset, k)
    
    for k_i in np.arange(k):
        mlp.init_weights()
        
        training_parts = set(np.arange(k))
        training_parts.remove(k_i)
        dataset_train = np.concatenate([parts[i] for i in list(training_parts)])
        dataset_test = parts[k_i]

        input_data = dataset_train[:,0:mlp.n_inputs]
        output_data = dataset_train[:,mlp.n_inputs:(mlp.n_inputs+mlp.n_outputs)]
        input_data_test = dataset_test[:,0:mlp.n_inputs]
        output_data_test = dataset_test[:,mlp.n_inputs:(mlp.n_inputs+mlp.n_outputs)]
        
        MSE_train[k_i,:], MSE_test[k_i,:] = mlp.fit((input_data, output_data), (input_data_test, output_data_test),
                                                    learning_rate=learning_rate, momentum=momentum, epochs=epochs)

    return (np.mean(MSE_train, axis=0), np.mean(MSE_test, axis=0))