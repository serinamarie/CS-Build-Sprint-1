

from helperfunctions import train_test_split
import pandas as pd
class DecisionTreeClassifier:
    '''For numeric data only'''
    def __init__(self, min_samples=2, max_depth=5):
        
        self.max_depth = max_depth
        self.min_samples = min_samples

    def purity_check(self, data):

        # last column of the df must be the labels!
        labels = data[:,-1]

        # if data has only one kind of label
        if len(np.unique(true_labels)) == 1:

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

    def entropy(self, data):

        # labels for input data
        labels = data[:,-1]

        # instantiate a Counter object on the labels
        counter = Counter(labels)

        # get the label counts
        counts = np.array([c[1] for c in counter.most_common()])

        # get the label probabilities
        probabilities = counts / counts.sum()

        # calculate the entropy
        entropy = sum(probabilities * -np.log2(probabilities))

        # return the entropy
        return entropy 

    def split_data(self, data, split_feature, split_threshold):
        return data_below, data_above

    def overall_entropy(self, data_below, data_above):
        return overall_entropy

    def find_best_split(data, potential_splits):
        return best_split_feature, best_split_threshold

    def fit(self, df, counter=0):
        '''only one arg needed (df). fitting this training df will account for
        splitting data into X and y'''
        if counter == 0:
            global column_headers
            column_headers = df.columns
            data = df.values

        else:
            data = df

        
        # result must be set to fitted_tree or something
        
    def predict(self, test):
        pass


if __name__ == '__main__':
    sample_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5], 'col2': [3, 4, 5, 6, 7]})
    train, test = train_test_split(sample_df, 0.2)
    d = DecisionTreeClassifier()
    tree = d.fit(train)
    print(tree)

