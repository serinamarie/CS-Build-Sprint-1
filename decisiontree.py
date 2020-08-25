class DecisionTreeClassifier:
    '''For numeric data only'''
    def __init__(self, min_samples=2, max_depth=5):
        self.max_depth = max_depth
        self.min_samples = min_samples

    def _fit(self, df, counter=0):
        '''only one arg needed (df). fitting this training df will account for
        splitting data into X and y'''
        pass
    
    def _predict(self, )


if __name__ == '__main__':
    d = DecisionTreeClassifier()
    print(d)