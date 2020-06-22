import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model import ADDSTCN
import random
import heapq
import pandas as pd
import numpy as np
import networkx as nx
import copy
import matplotlib.pyplot as plt


class TCDF():
    r"""TCDF — Temporal Causal Discovery Framework

    Parameters
    ----------
    
    data : str
        Path to one or more datasets to analyse by TCDF containing multiple time series.
        data format: csv with a column (incl. header) for each time series.
    ground_truth_provided : str
        Provide dataset(s) and the ground truth(s) to evaluate the results of TCDF.
        datasets format: csv, a column with header for each time series.
        ground truth format: csv, no header, index of cause in first column,
        index of effect in second column, time delay
        between cause and effect in third column.


    Attributes
    ---------

    cuda : bool
        Use CUDA (GPU)
    epochs : int
        Number of epochs
    kernel_size : int
        Size of kernel, i.e. window size.
        Maximum delay to be found is kernel size - 1.
        Recommended to be equal to dilation coeffient
    hidden_layers : int
        Number of hidden layers in the depthwise convolution
    dilation_coefficient : int
        Dilation coefficient, recommended to be equal to kernel size
    significance : float
        Significance number stating when an increase in loss
        is significant enough to label a potential cause as true
        (validated) cause. See paper for more details.
    learning_rate : float
        Learning rate
    optimizer : str
        Optimizer to use (Adam, RMSprop)
    log_interval : int
        Epoch interval to report loss
    seed : int
        Random seed
    plot : bool
        Show causal graph
    """
    def __init__(self, data, ground_truth_provided, cuda=False,
                 epochs=1000, kernel_size=4, hidden_layers=0, dilation_coefficient=4, significance=0.8,
                 learning_rate=0.01, optimizer='Adam', log_interval=500, seed=1111, plot=False):
        if torch.cuda.is_available():
            if not cuda:
                print('WARNING: You have a CUDA device, you should probably run\n\
                       with --cuda to speed up training.')

        if kernel_size != dilation_coefficient:
            print('WARNING: The dilation coefficient is not equal to the kernel size.\n\
                   Multiple paths can lead to the same delays.\n\
                   Set kernel_size equal to dilation_c to have exaxtly one path for each delay.')
        self.datasets = {}
        self.ground_truth_provided = ground_truth_provided
        self.cuda = cuda
        self.num_epochs = epochs
        self.kernel_size = kernel_size
        self.levels = hidden_layers + 1
        self.learningrate = learning_rate
        self.optimizername = optimizer
        self.dilation_c = dilation_coefficient
        self.significance = significance
        self.loginterval = log_interval
        self.seed = seed
        self.plot = plot
        self.models = {}
        self.verbose = None

        # Create dictionary containing datasets as keys and ground truth files as values
        if self.ground_truth_provided:
            for kv in data.split(','):
                k, v = kv.split('=')
                self.datasets[k] = v
        else:
            for dataset in data.split(','):
                self.datasets[dataset] = ''

    def solve(self):
        if self.ground_truth_provided:
            totalF1direct = []  # contains F1-scores of all datasets
            totalF1 = []  # contains F1'-scores of all datasets

            receptivefield = 1
            for level in range(0, self.levels):
                receptivefield += (self.kernel_size - 1) * self.dilation_c**(level)

        for dataset in self.datasets.keys():
            dataset_name = str(dataset)
            if '/' in dataset_name:
                dataset_name = str(dataset).rsplit('/', 1)[1]
            
            print(f'\n Dataset: {dataset_name}')

            # run TCDF
            allcauses, alldelays, allreallosses, allscores, columns = self._runTCDF(dataset)  # results of TCDF containing indices of causes and effects

            print(f'\n=================== Results for {dataset_name} ==================================')
            for pair in alldelays:
                print(columns[pair[1]], "causes", columns[pair[0]], "with a delay of", alldelays[pair], "time steps.")
            
            if self.ground_truth_provided:
                # evaluate TCDF by comparing discovered causes with ground truth
                print(f'\n=================== Evaluation for {dataset_name} ===============================')
                FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = self._evaluate(self.datasets[dataset], allcauses, columns)
                totalF1.append(F1)
                totalF1direct.append(F1direct)

                # evaluate delay discovery
                extendeddelays, readgt, extendedreadgt = self._getextendeddelays(self.datasets[dataset], columns)
                percentagecorrect = self._evaluatedelay(extendeddelays, alldelays, TPs, receptivefield) * 100
                print("Percentage of delays that are correctly discovered: ", percentagecorrect, "%")
                
            print('==================================================================================')
            
            if self.plot:
                self._plotgraph(dataset_name, alldelays, columns)

        # In case of multiple datasets, calculate average F1-score over all datasets and standard deviation
        if len(self.datasets.keys()) > 1 and self.ground_truth_provided:
            print("\nOverall Evaluation: \n")
            print("F1' scores: ")
            for f in totalF1:
                print(f)
            print("Average F1': ", np.mean(totalF1))
            print("Standard Deviation F1': ", np.std(totalF1), "\n")
            print("F1 scores: ")
            for f in totalF1direct:
                print(f)
            print("Average F1: ", np.mean(totalF1direct))
            print("Standard Deviation F1: ", np.std(totalF1direct))

    def _getextendeddelays(self, gtfile, columns):
        """Collects the total delay of indirect causal relationships."""
        gtdata = pd.read_csv(gtfile, header=None)

        readgt = dict()
        effects = gtdata[1]
        causes = gtdata[0]
        delays = gtdata[2]
        gtnrrelations = 0
        pairdelays = dict()
        for k in range(len(columns)):
            readgt[k] = []
        for i in range(len(effects)):
            key = effects[i]
            value = causes[i]
            readgt[key].append(value)
            pairdelays[(key, value)] = delays[i]
            gtnrrelations += 1

        g = nx.DiGraph()
        g.add_nodes_from(readgt.keys())
        for e in readgt:
            cs = readgt[e]
            for c in cs:
                g.add_edge(c, e)

        extendedreadgt = copy.deepcopy(readgt)
        
        for c1 in range(len(columns)):
            for c2 in range(len(columns)):
                paths = list(nx.all_simple_paths(g, c1, c2, cutoff=2))  # indirect path max length 3, no cycles
                
                if len(paths) > 0:
                    for path in paths:
                        for p in path[:-1]:
                            if p not in extendedreadgt[path[-1]]:
                                extendedreadgt[path[-1]].append(p)
                                
        extendedgtdelays = dict()
        for effect in extendedreadgt:
            causes = extendedreadgt[effect]
            for cause in causes:
                if (effect, cause) in pairdelays:
                    delay = pairdelays[(effect, cause)]
                    extendedgtdelays[(effect, cause)] = [delay]
                else:
                    # find extended delay
                    paths = list(nx.all_simple_paths(g, cause, effect, cutoff=2))  # indirect path max length 3, no cycles
                    extendedgtdelays[(effect, cause)] = []
                    for p in paths:
                        delay = 0
                        for i in range(len(p) - 1):
                            delay += pairdelays[(p[i + 1], p[i])]
                        extendedgtdelays[(effect, cause)].append(delay)

        return extendedgtdelays, readgt, extendedreadgt


    def _evaluate(self, gtfile, validatedcauses, columns):
        """Evaluates the results of TCDF by comparing it to the ground truth graph,
        and calculating precision, recall and F1-score.
        F1'-score, precision' and recall' include indirect causal relationships."""
        extendedgtdelays, readgt, extendedreadgt = self._getextendeddelays(gtfile, columns)
        FP = 0
        FPdirect = 0
        TPdirect = 0
        TP = 0
        FN = 0
        FPs = []
        FPsdirect = []
        TPsdirect = []
        TPs = []
        FNs = []
        for key in readgt:
            for v in validatedcauses[key]:
                if v not in extendedreadgt[key]:
                    FP += 1
                    FPs.append((key, v))
                else:
                    TP += 1
                    TPs.append((key, v))
                if v not in readgt[key]:
                    FPdirect += 1
                    FPsdirect.append((key, v))
                else:
                    TPdirect += 1
                    TPsdirect.append((key, v))
            for v in readgt[key]:
                if v not in validatedcauses[key]:
                    FN += 1
                    FNs.append((key, v))
        
        print("Total False Positives': ", FP)
        print("Total True Positives': ", TP)
        print("Total False Negatives: ", FN)
        print("Total Direct False Positives: ", FPdirect)
        print("Total Direct True Positives: ", TPdirect)
        print("TPs': ", TPs)
        print("FPs': ", FPs)
        print("TPs direct: ", TPsdirect)
        print("FPs direct: ", FPsdirect)
        print("FNs: ", FNs)
        precision = recall = 0.

        if float(TP + FP) > 0:
            precision = TP / float(TP + FP)
        print("Precision': ", precision)
        if float(TP + FN) > 0:
            recall = TP / float(TP + FN)
        print("Recall': ", recall)
        if (precision + recall) > 0:
            F1 = 2 * (precision * recall) / (precision + recall)
        else:
            F1 = 0.
        print("F1' score: ", F1, "(includes direct and indirect causal relationships)")

        precision = recall = 0.
        if float(TPdirect + FPdirect) > 0:
            precision = TPdirect / float(TPdirect + FPdirect)
        print("Precision: ", precision)
        if float(TPdirect + FN) > 0:
            recall = TPdirect / float(TPdirect + FN)
        print("Recall: ", recall)
        if (precision + recall) > 0:
            F1direct = 2 * (precision * recall) / (precision + recall)
        else:
            F1direct = 0.
        print("F1 score: ", F1direct, "(includes only direct causal relationships)")
        return FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct


    def _evaluatedelay(self, extendedgtdelays, alldelays, TPs, receptivefield):
        """Evaluates the delay discovery of TCDF by comparing the discovered
        time delays with the ground truth."""
        zeros = 0
        total = 0.
        for i in range(len(TPs)):
            tp = TPs[i]
            discovereddelay = alldelays[tp]
            gtdelays = extendedgtdelays[tp]
            for d in gtdelays:
                if d <= receptivefield:
                    total += 1.
                    error = d - discovereddelay
                    if error == 0:
                        zeros += 1
                else:
                    next
            
        return 0. if zeros == 0 else zeros / float(total)

    def _runTCDF(self, datafile):
        """Loops through all variables in a dataset and return the discovered
        causes, time delays, losses, attention scores and variable names."""
        df_data = pd.read_csv(datafile)

        allcauses = dict()
        alldelays = dict()
        allreallosses = dict()
        allscores = dict()

        columns = list(df_data)
        for c in columns:
            idx = df_data.columns.get_loc(c)
            causes, causeswithdelay, realloss, scores = self._findcauses(
                c, file=datafile, cuda=self.cuda, epochs=self.num_epochs,
                kernel_size=self.kernel_size, layers=self.levels,
                dilation_c=self.dilation_c, significance=self.significance,
                lr=self.learningrate, optimizername=self.optimizername,
                log_interval=self.loginterval, seed=self.seed)

            allscores[idx] = scores
            allcauses[idx] = causes
            alldelays.update(causeswithdelay)
            allreallosses[idx] = realloss

        return allcauses, alldelays, allreallosses, allscores, columns


    def _plotgraph(self, stringdatafile, alldelays, columns):
        """Plots a temporal causal graph showing all discovered causal relationships
        annotated with the time delay between cause and effect."""
        G = nx.DiGraph()
        for c in columns:
            G.add_node(c)
        for pair in alldelays:
            p1, p2 = pair
            nodepair = (columns[p2], columns[p1])

            G.add_edges_from([nodepair], weight=alldelays[pair])

        edge_labels = dict([((u, v, ), d['weight'])
                            for u, v, d in G.edges(data=True)])

        pos = nx.circular_layout(G)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.draw(G, pos, node_color='white', edge_color='black', node_size=1000, with_labels=True)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000")

        plt.show()

    def _preparedata(self, file, target):
        """Reads data from csv file and transforms it to two PyTorch tensors: dataset x and target time series y that has to be predicted."""
        df_data = pd.read_csv(file)
        df_y = df_data.copy(deep=True)[[target]]
        df_x = df_data.copy(deep=True)
        df_yshift = df_y.copy(deep=True).shift(periods=1, axis=0)
        df_yshift[target] = df_yshift[target].fillna(0.)
        df_x[target] = df_yshift
        data_x = df_x.values.astype('float32').transpose()
        data_y = df_y.values.astype('float32').transpose()
        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)

        x, y = Variable(data_x), Variable(data_y)
        return x, y

    def _train(self, epoch, traindata, traintarget, modelname, optimizer, log_interval, epochs):
        """Trains model by performing one epoch and returns attention scores and loss."""

        modelname.train()
        x, y = traindata[0:1], traintarget[0:1]
            
        optimizer.zero_grad()
        epochpercentage = (epoch / float(epochs)) * 100
        output = modelname(x)

        attentionscores = modelname.fs_attention
        
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0 or epoch % epochs == 0 or epoch == 1:
            print('Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}'.format(epoch, epochpercentage, loss))

        return attentionscores.data, loss

    def _findcauses(self, target, cuda, epochs, kernel_size, layers, log_interval,
                    lr, optimizername, seed, dilation_c, significance, file):
        """Discovers potential causes of one target time series, validates these
        potential causes with PIVM and discovers the corresponding time delays
        """

        print("\n", "Analysis started for target: ", target)
        torch.manual_seed(seed)
        
        X_train, Y_train = self._preparedata(file, target)
        X_train = X_train.unsqueeze(0).contiguous()
        Y_train = Y_train.unsqueeze(2).contiguous()

        input_channels = X_train.size()[1]
        
        targetidx = pd.read_csv(file).columns.get_loc(target)
            
        self.models[target] = ADDSTCN(targetidx, input_channels, layers, cuda=cuda,
                                      kernel_size=kernel_size, dilation_c=dilation_c)
        if cuda:
            self.models[target].cuda()
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()

        optimizer = getattr(optim, optimizername)(self.models[target].parameters(), lr=lr)
        
        scores, firstloss = self._train(1, X_train, Y_train, self.models[target], optimizer, log_interval, epochs)
        firstloss = firstloss.cpu().data.item()
        for ep in range(2, epochs + 1):
            scores, realloss = self._train(ep, X_train, Y_train, self.models[target], optimizer, log_interval, epochs)
        realloss = realloss.cpu().data.item()
        
        s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
        indices = np.argsort(-1 * scores.view(-1).cpu().detach().numpy())
        
        # attention interpretation to find tau: the threshold that distinguishes potential causes from non-causal time series
        if len(s) <= 5:
            potentials = []
            for i in indices:
                if scores[i] > 1.:
                    potentials.append(i)
        else:
            potentials = []
            gaps = []
            for i in range(len(s) - 1):
                if s[i] < 1.:  # tau should be greater or equal to 1, so only consider scores >= 1
                    break
                gap = s[i] - s[i + 1]
                gaps.append(gap)
            sortgaps = sorted(gaps, reverse=True)
            
            for i in range(0, len(gaps)):
                largestgap = sortgaps[i]
                index = gaps.index(largestgap)
                ind = -1
                if index < ((len(s) - 1) / 2):  # gap should be in first half
                    if index > 0:
                        ind = index  # gap should have index > 0, except if second score <1
                        break
            if ind < 0:
                ind = 0
                    
            potentials = indices[:ind + 1].tolist()
        print("Potential causes: ", potentials)
        validated = copy.deepcopy(potentials)
        
        # Apply PIVM (permutes the values) to check if potential cause is true cause
        for idx in potentials:
            # zeros instead of intervention
            # X_test2 = np.zeros(X_train.shape)
            # shuffled = torch.from_numpy(X_test2).float()
            
            # original TCDF solution
            random.seed(seed)
            X_test2 = X_train.clone().cpu().numpy()
            random.shuffle(X_test2[:, idx, :][0])
            shuffled = torch.from_numpy(X_test2)
            if cuda:
                shuffled = shuffled.cuda()
            self.models[target].eval()
            output = self.models[target](shuffled)
            testloss = F.mse_loss(output, Y_train)
            testloss = testloss.cpu().data.item()
            
            diff = firstloss - realloss
            testdiff = firstloss - testloss
            print('firstloss - realloss = diff')
            print('firstloss - testloss = testdiff')
            print(f'{firstloss} - {realloss} = {diff}')
            print(f'{firstloss} - {testloss} = {testdiff}')

            if testdiff > (diff * significance):
                validated.remove(idx)
        
    
        weights = []
        
        # Discover time delay between cause and effect by interpreting kernel weights
        for layer in range(layers):
            shapes = self.models[target].dwn.network[layer].net[0].weight.size()
            weight = self.models[target].dwn.network[layer].net[0].weight.abs().view(shapes[0], shapes[2])
            weights.append(weight)

        causeswithdelay = dict()
        for v in validated:
            totaldelay = 0
            for k in range(len(weights)):
                w = weights[k]
                row = w[v]
                twolargest = heapq.nlargest(2, row)
                m = twolargest[0]
                m2 = twolargest[1]
                if m > m2:
                    index_max = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
                else:
                    # take first filter
                    index_max = 0
                delay = index_max * (dilation_c**k)
                totaldelay += delay
            if targetidx != v:
                causeswithdelay[(targetidx, v)] = totaldelay
            else:
                causeswithdelay[(targetidx, v)] = totaldelay + 1
        print("Validated causes: ", validated)
        
        return validated, causeswithdelay, realloss, scores.view(-1).cpu().detach().numpy().tolist()
