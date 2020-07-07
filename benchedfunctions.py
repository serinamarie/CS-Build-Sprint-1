



class Label:

    '''Attributes of our labels include counts, nunique and most common'''

    '''Eventually place within DecisionTreeClassifier'''

    def __init__(self, data):
        self.data = data

    def counts(self):
            
        '''Returns an array of the label counts'''
        # labels for input data
        labels = self.data[-1]

        # instantiate a Counter object on the labels
        counter = Counter(labels)

        # get the label counts
        return np.array([c[1] for c in counter.most_common()])

    def nunique(self):

        '''Returns the number of unique labels present'''

        # last column of the df must be the labels!
        labels = self.data[-1]

        # if data has only one kind of label
        return len(np.unique(labels))
        
    def most_common(self):

        '''Returns the most common label'''

        labels = self.data[-1]

        counter = Counter(labels)

        # return the most common class/label
        return counter.most_common(1)[0][0]