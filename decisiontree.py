

from helperfunctions import train_test_split
import pandas as pd
from collections import Counter
import numpy as np

class DecisionTreeClassifier:
    '''For numeric data only'''

    def __init__(self, min_samples=2, max_depth=5, tree=None):
        
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def fit(self, train_df):

        self.tree = self.decision_tree(train_df, self.min_samples, self.max_depth)
        return self

    def calculate_entropy(self, data):

        # labels for input data
        labels = data[:, -1]

        # instantiate a Counter object on the labels
        counts = Counter(labels)

        # get the label probabilities
        probabilities = counts / counts.sum()

        # calculate the entropy
        entropy = sum(probabilities * -np.log2(probabilities))

        # return the entropy
        return entropy 

    def purity_check(self, data):

        # last column of the df must be the labels!
        labels = data[-1]

        # if data has only one kind of label
        if len(np.unique(labels)) == 1:

            # it is pure
            return True

        # if data has a few different labels still
        else:

            # it isn't pure
            return False

    def make_classification(self, data):

        '''Once the max depth or min samples or purity is 1, 
        we classify the data with whatever the majority of the labels are'''

        # labels for input data
        labels = data[:, -1]

        # instantiate a Counter object on the labels
        counter = Counter(labels)

        # return the most common class/label
        return counter.most_common(1)[0][0]

    def split_data(self, data, split_feature, split_threshold):

        # array of only the split_feature
        feature_values = data[:, split_feature]

        # array where feature values do not exceed threshold
        data_below_threshold = data[feature_values <= split_threshold]

        # array where feature values exceed threshold
        data_above_threshold = data[feature_values > split_threshold]
        
        return data_below_threshold, data_above_threshold

    def overall_entropy(self, data_below_threshold, data_above_threshold):

        '''Overall entropy'''

        p = len(data_below_threshold) / (len(data_below_threshold) + len(data_above_threshold))

        return p * calculate_entropy(data_below_threshold) + p * calculate_entropy(data_above_threshold)

    def potential_splits(data):
        
        # dictionary of splits
        potential_splits = {}

        # store the number of features (not including labels/target)
        n_features = len(data[0]) - 1

        # for each feature in possible features
        for feature in range(n_features):

            # for our dictionary, each feature should be a key
            potential_splits[feature] = []

            # we need to iterate through each feature's unique values
            unique_values_for_feature = np.unique(data[:, feature])

            for index in range(1, len(unique_values_for_feature)):

                # we need to partition the data, we need the midpoint between the unique values
                current = unique_values_for_feature[index]
                prev = unique_values_for_feature[index - 1]
                midpoint = (current + prev) / 2

                # for our dictionary each value should be a midpoint between the 
                # unique values for that feature
                potential_splits[feature].append(midpoint)

        # return dictionary
        return potential_splits

    def find_best_split(data, potential_splits):

        lowest_entropy = 9999

        # for each dictionary key
        for key in potential_splits:
            
            # for each value for that key
            for value in potential_splits[key]:

                # split our data into on that threshold (value)
                data_below_threshold, data_above_threshold = split_data(
                    data=data, 
                    split_feature=key,
                    split_threshold=value)
                
                # calculate entropy at this split
                entropy_for_this_split = overall_entropy(data_below_threshold, data_above_threshold)

                # if entropy at this split is lower than the lowest entropy found so far
                if entropy_for_this_split < lowest_entropy:

                    # the entropy at this split is now the lowest 
                    lowest_entropy = entropy_for_this_split

                    # keep a record of this key, value pair
                    best_split_feature = key
                    best_split_threshold = value

        # return the best potential split
        return best_split_feature, best_split_threshold


    def decision_tree(self, train_df, min_samples, max_depth, counter=0):

        '''only one arg needed (df). fitting this training df will account for
        splitting data into X and y'''

        # if this is our first potential split
        if counter == 0:

            # set this global variable so we can use it each time we call decision_tree()
            global feature_names
            feature_names = train_df.columns

            # get our df values
            data = train_df.values

        # if we have recursively reached this point
        else:

            # our 'train_df' is actually already an array upon recursion
            data = train_df

        # base case: if our impurity for a subtree is 0 or we have reached our 
        # maximum depth or min samples
        
        if (purity_check(data)) or (len(data) < min_samples) or (counter == max_depth):
            
            # at this point we'll have to make a judgment call and classify it based on the majority
            majority_vote = make_classification(data)

            return majority_vote

        # if we haven't reach one of our stopping points
        else:

            # increment counter
            counter += 1

            # get a dictionary of our potetnial splits
            potential_splits = potential_splits(data)

            # find the best split
            split_feature, split_threshold = find_best_split(data, potential_splits)

            # get the data below and above
            data_below_threshold, data_above_threshold = split_data(data, split_feature, split_threshold)

            # store feature name as string
            feature_name_as_string = feature_names[split_feature]

            # feature_name <= threshold value
            question = "{} <= {}".format(feature_name_as_string, split_threshold)

            # create a dictionary with our questions 
            subtree = {question: []}

            # recursion on our true values
            answer_true = decision_tree(data_below_threshold, counter, min_samples, max_depth)

            # recursion on our false values
            answer_false = decision_tree(data_above_threshold, counter, min_samples, max_depth)

            # if both answers are the same class
            if answer_true == answer_false:

                # choose one to be the subtree
                subtree = answer_true
            
            # if answers result in different class
            else:

                # append to dictionary
                subtree[question].append(answer_true)
                subtree[question].append(answer_false)
            
            return subtree








        
        # result must be set to fitted_tree or something

    def classify_observation(self, observation, tree):

        # if the tree is not None
        if tree:

            # store the current question 
            question = list(tree.keys())[0]

            # grab the feature name and value 
            feature_name, _, value = question.split()

            # if the row at that feature column is less than the threshold
            if observation[feature_name] <= float(value):

                # answer yes, it's under the threshold
                answer = tree[question][0]

            # if the row at that feature column has exceeded the threshold
            else:

                # answer no, it has exceeded the threshold
                answer = tree[question][1]

            # if the answer is not a dictionary
            if not isinstance(answer, dict):

                # return answer as it is a class label
                return answer

            # if the answer is a dictionary
            else:

                # recursion with the 'answer' subtree as the tree argument
                return classify_observation(observation, answer)
        
    def predict(self, test_df, tree):

        # if a tree has been fitted
        if tree: 

            # create a new column for our predictions
            test_df['predictions'] = test_df.apply(classify_observation, axis=1, args=(tree,))

            # return True or False where our predictions are equal to our labels
            test_df['correct_predictions'] = test_df['predictions'] == test_df.iloc[:,-1]

            # return how accurate the predictions are
            return test_df['correct_predictions'].mean()

if __name__ == '__main__':
    sample_df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 'blue'], 
        'col2': [3, 4, 5, 6, 'blue'],
        'col3': [3, 4, 5, 6, 'blue'],
    })
    train, test = train_test_split(sample_df, 0.2)
    d = DecisionTreeClassifier()
    # print(d)
    train_data = train.values
    d.fit(train)
    # if d.purity_check(train_data):
    #     print("True positive")
    # else:
    #     print("False negative :0")
# d.purity_check(train_data)
    # print(tree)

    # l = Label(sample_df.values)
    # # print(l)
    # print('Counts:', l.counts(), type(l.counts()))
    # print('Nunique:', l.nunique(), type(l.nunique()))
    # print("MC:", l.most_common(), type(l.most_common()))

