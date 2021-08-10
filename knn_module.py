# This module contains the KNN class
## This module will be used to classify new investors based on their investments and their Euclidian distance to already-classified investors.

class KNN():
    '''This class allows for classifying a new investor based on its own investment portfolio in accord to a list of already-classified investors. Investor will be labeled as either "Conservative" or "Moderate" or "Aggressive".
    The label will be assigned based on the mode of labels of the K nearest already-classified investors (neighbors). The K nearest neighbours will those already-classified investors whose Euclidian distances to the new investor are the shortest ones.
    
    Args:
        classified_dataset: A list of lists, each containing information on the already-classified investors. Every sublist must comprise an investor ID (string), an investor label (string) and an investment portfolio (tuple of float).
                                                        
                            Example: [[53164949799, 'Conservative', (5400.0, 3900.0, 1700.0, 400.0)],
                                     [63858864633, 'Conservative', (5400.0, 3700.0, 1500.0, 200.0)],
                                     [52163844491, 'Conservative', (5500.0, 4200.0, 1400.0, 200.0)],
                                     [65441690046, 'Conservative', (5500.0, 3500.0, 1300.0, 200.0)],
                                     [49212614633, 'Aggressive', (5900., 3000., 5100., 1800.)]]
                              
        no_classified_subject: A list containing information on the new client. Information are ID (string), label as an empty string and an investment portfolio (tuple of float).
                            
                            Example: [64703873108, '', (6000., 2200., 5000., 1500.)]
    
    Attributes:
            client_identification: New investor's ID.
            client_portfolio: New investor's portfolio.
            client_information: List of all provided information on the new investor.
            classified_dataset: Fullfilled dataset with information on all already-classified investors.
    '''
    
    
    # constructor method
    def __init__(self, classified_dataset, no_classified_subject):
        self.client_identification = no_classified_subject[0]
        self.client_portfolio = no_classified_subject[2]
        self.client_information = no_classified_subject
        self.classified_dataset = [x for x in classified_dataset]
    
    
    # All METHODS BELOW
    
    # calculation of Euclidian distance
    def euclidian_distances_no_class(self):
        ''' This method calculates the Euclidian distance between the portfolios of a new client and every single already-classified investor.
        
        Args:
            None.
            
        Return: List of tuples as follows -> [(euclidian_distance, ID_investor_already_classified)].
        '''
        
        # list that will store the distances and ID
        distance_between_individuals = []
        
        for id_, class_, invest in self.classified_dataset:
            distance = 0
            for i in range(len(invest)):
                distance += ((self.client_portfolio[i] - invest[i]) ** 2)
            distance = (distance ** 0.5)
            distance_between_individuals.append((distance, id_))
        
        return distance_between_individuals
    
    
    
    # distances in ascending order
    def distances_sort(self, distance_between_individuals):
        '''Organize the list passed in from the shortest to the largest Euclidian distance.
        
        Args:
            distance_between_individuals: List of tuples as follows -> [(euclidian_distance, ID_investor_already_classified)].
        
        Return: List of tuples organized by the Euclidian distances.
        '''
        # sorting
        distance = distance_between_individuals
        distance.sort()
        return distance
        
    
    
    # classificaton of new investor
    def investor_classification(self, sorted_data, k = 3):
        ''' This method labels a new investor based on its portfolio of investments.
                 
        Args:
            sorted_data: List of tuples [(euclidian_distance, ID_investor_already_classified)] sorted in ascending order.
            k: K refrers to the number of nearest neighbours that will be taken into account in order to find a new investor a label.
        
        Return: It labels a single new investor as either "Conservative" or "Moderate" or "Aggressive".
        
        '''
        
        # sorted list assignment
        neighbour = sorted_data
        
        # storage of the k-nearest-neighbours
        classifications = []
        
        for i in range(k):
            for id_, class_, invest in self.classified_dataset:
                if id_ == neighbour[i][1]:
                    classifications.append(class_)
        
        # finding the mode of k neighbours
        countings = {'Conservative': 0, 'Moderate': 0, 'Aggressive': 0}
        
        for i in classifications:
            if i == 'Conservative':
                countings['Conservative'] += 1
            elif i == 'Moderate':
                countings['Moderate'] += 1
            else:
                countings['Aggressive'] += 1

        # Finding the mode
        countings = list(countings.items())
        countings.sort(key=lambda x: x[1], reverse = True)
        return countings[0][0]
    