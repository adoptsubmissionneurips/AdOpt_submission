import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

def accuracy(loader, classifier, generator, device):
    total = 0
    correct = 0
    # Establish the accuracy of the model on the test dataset
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            outputs = classifier(generator(inputs))
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def metrics_adversarial(X, y, classifier, generator, threshold=.5):
    '''
    outputs dictionary of classifier performance metrics:
    'recall'
    'precision'
    'f1'
    'accuracy'
    'predicted_positives'
    'ground_truth'
    '''
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')
    threshold=torch.tensor(.5, dtype=torch.float32, device=device)
    data = TensorDataset(X, y)
    loader = DataLoader(data, batch_size=X.size(dim=0))
    total_pos_c=true_pos_c=false_pos_c=true_neg_c=false_neg_c = 0
    # Establish the accuracy of the model on the test dataset
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = \
                inputs.to(device), labels.to(device)
            outputs = classifier(generator(inputs))
            # outputs = classifier.get_thresholded_predictions(generator(inputs),threshold)
            # predicted=outputs
            # current treshold is 0.5. need to introduce treshold parameter in end code
            predicted = torch.round(outputs).squeeze()
            true_pos, false_pos, true_neg, false_neg = confusion(predicted,labels)
            true_pos_c += true_pos
            false_pos_c += false_pos
            true_neg_c += true_neg
            false_neg_c += false_neg
        # need to look at output in borderline cases more carefully in end code
        if true_pos_c+false_pos_c>0:
            precision=true_pos_c / (true_pos_c+false_pos_c)
        else:
            # classifier is predicting everything to be negative
            precision=None
        if (true_pos_c+false_neg_c)>0:
            recall=true_pos_c/(true_pos_c+false_neg_c)
        else:
            #no positive cases so recall cannot be evaluated
            recall=None
        if (false_pos_c+true_neg_c)>0:
          fpr=false_pos_c/(false_pos_c+true_neg_c)
        else:
          fpr=None
        if precision is not None and recall is not None and (precision+recall)>0:
          f1=2*precision*recall/(precision+recall)
        else:
          f1=None
        acc=(true_pos_c+true_neg_c)/(true_pos_c+false_pos_c+true_neg_c+false_neg_c)
        ground_truth=(true_pos_c+false_neg_c)/(true_pos_c+false_pos_c+true_neg_c+false_neg_c)
        predicted_pos=(true_pos_c+false_pos_c)/(true_pos_c+false_pos_c+true_neg_c+false_neg_c)
        return {'recall':recall, 'precision':precision, 'f1':f1, 'accuracy':acc, 'predicted_pos':predicted_pos, 'ground_truth':ground_truth}

def metrics(X, y, classifier, threshold=.5):
    '''
    outputs dictionary of classifier performance metrics:
    'recall'
    'precision'
    'f1'
    'accuracy'
    'predicted_positives'
    'ground_truth'
    '''
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')
    threshold=torch.tensor(.5, dtype=torch.float32, device=device)
    data = TensorDataset(X, y)
    loader = DataLoader(data, batch_size=X.size(dim=0))
    total_pos_c=true_pos_c=false_pos_c=true_neg_c=false_neg_c = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = \
                inputs.to(device), labels.to(device)
            # outputs = classifier(generator(inputs))
            outputs = classifier.get_thresholded_predictions(inputs,threshold)
            predicted=outputs
            # current treshold is 0.5. need to introduce treshold parameter in end code
            # predicted = torch.round(outputs).squeeze()
            true_pos, false_pos, true_neg, false_neg = confusion(predicted,labels)
            true_pos_c += true_pos
            false_pos_c += false_pos
            true_neg_c += true_neg
            false_neg_c += false_neg
        # need to look at output in borderline cases more carefully in end code
        if true_pos_c+false_pos_c>0:
            precision=true_pos_c / (true_pos_c+false_pos_c)
        else:
            # classifier is predicting everything to be negative
            precision=None
        if (true_pos_c+false_neg_c)>0:
            recall=true_pos_c/(true_pos_c+false_neg_c)
        else:
            #no positive cases so recall cannot be evaluated
            recall=None
        if (false_pos_c+true_neg_c)>0:
          fpr=false_pos_c/(false_pos_c+true_neg_c)
        else:
          fpr=None
        if precision is not None and recall is not None and (precision+recall)>0:
          f1=2*precision*recall/(precision+recall)
        else:
          f1=None
        acc=(true_pos_c+true_neg_c)/(true_pos_c+false_pos_c+true_neg_c+false_neg_c)
        ground_truth=(true_pos_c+false_neg_c)/(true_pos_c+false_pos_c+true_neg_c+false_neg_c)
        predicted_pos=(true_pos_c+false_pos_c)/(true_pos_c+false_pos_c+true_neg_c+false_neg_c)
        return {'recall':recall, 'precision':precision, 'f1':f1, 'accuracy':acc, 'predicted_pos':predicted_pos, 'ground_truth':ground_truth}


def roc_score(loader, classifier, generator):
    # Set device to train on cpu or gpu
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

    classifier.to(device)
    generator.to(device)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = classifier(generator(inputs)).squeeze()
    return roc_auc_score(labels.detach().cpu().numpy(),outputs.detach().cpu().numpy())
