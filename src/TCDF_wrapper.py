import torch
import torch.optim
from torch.nn.functional import mse_loss
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
    def __init__(self,
                 cuda=False,
                 epochs=1000,
                 kernel_size=4,
                 hidden_layers=0,
                 dilation_coefficient=4,
                 significance=0.8,
                 learning_rate=0.01,
                 optimizer='Adam',
                 log_interval=500,
                 seed=1111):
        if torch.cuda.is_available():
            if not cuda:
                print('WARNING: You have a CUDA device, you should probably run\n\
                       with --cuda to speed up training.')

        if kernel_size != dilation_coefficient:
            print('WARNING: The dilation coefficient is not equal to the kernel size.\n\
                   Multiple paths can lead to the same delays.\n\
                   Set kernel_size equal to dilation_c to have exaxtly one path for each delay.')
        
        self.cnn_parameters = {'cuda': cuda,
                               'num_epochs': epochs,
                               'kernel_size': kernel_size,
                               'levels': hidden_layers + 1,
                               'dilation_c': dilation_coefficient,
                               'significance': significance,
                               'learning_rate': learning_rate,
                               'optimizer': optimizer,
                               'log_interval': log_interval,
                               'seed': seed}
        self.models = None
        self.verbose = None
    
    def fit(self, X, verbose=None):
        """Loops through all variables in a dataset and return the discovered
        causes, time delays, losses, attention scores and variable names."""
        self.models = {}
        if verbose is not None:
            self.verbose = verbose
        
        dataframe = pd.read_csv(X)

        self.causes = dict()
        self.delays = dict()
        self.logs = {'real_losses': dict(), 'scores': dict()}

        self.columns = list(dataframe)
        for c in self.columns:
            idx = dataframe.columns.get_loc(c)
            causes, delays, real_loss, scores = self._findcauses(dataframe, c)

            self.causes[idx] = causes
            self.delays.update(delays)
            self.logs['real_losses'][idx] = real_loss
            self.logs['scores'][idx] = scores
    
    def _findcauses(self, dataframe, target):
        """Discovers potential causes of one target time series, validates these
        potential causes with PIVM and discovers the corresponding time delays
        """

        if self.verbose == 1:
            print('\nAnalysis started for target: ', target)
        
        # => Step 1: Preparing datasets and creating CNN with given parameters
        torch.manual_seed(self.cnn_parameters['seed'])
        
        X_train, Y_train = self._preparedata(dataframe, target)
        X_train = X_train.unsqueeze(0).contiguous()
        Y_train = Y_train.unsqueeze(2).contiguous()

        input_channels = X_train.size()[1]
            
        self.models[target] = ADDSTCN(
            input_size=input_channels,
            num_levels=self.cnn_parameters['levels'],
            kernel_size=self.cnn_parameters['kernel_size'],
            dilation_c=self.cnn_parameters['dilation_c'],
            cuda=self.cnn_parameters['cuda'])
        
        if self.cnn_parameters['cuda']:
            self.models[target].cuda()
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()

        optimizer = getattr(torch.optim, self.cnn_parameters['optimizer'])(self.models[target].parameters(),
                                                                           lr=self.cnn_parameters['learning_rate'])
        
        # => Step 2: Train CNN
        scores, first_loss = self._train(self.models[target], optimizer, X_train, Y_train, 1)
        first_loss = first_loss.cpu().data.item()
        for epoch in range(2, self.cnn_parameters['num_epochs'] + 1):
            scores, real_loss = self._train(self.models[target], optimizer, X_train, Y_train, epoch)
        real_loss = real_loss.cpu().data.item()
        
        # => Step 3: Attention interpretation
        # to find tau, threshold distinguishes potential causes
        # from non-causal time series
        s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
        indices = np.argsort(-1 * scores.view(-1).cpu().detach().numpy())
        
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
        
        if self.verbose == 1:
            print(f'Potential causes: {potentials}')
        validated = copy.deepcopy(potentials)
        
        # => Step 4: Validate potential causes
        # Apply PIVM (permutes the values) to check if potential cause is true cause
        for idx in potentials:
            # zeros instead of intervention
            # X_test2 = np.zeros(X_train.shape)
            # shuffled = torch.from_numpy(X_test2).float()
            
            # original TCDF solution
            random.seed(self.cnn_parameters['seed'])
            X_test2 = X_train.clone().cpu().numpy()
            random.shuffle(X_test2[:, idx, :][0])
            shuffled = torch.from_numpy(X_test2)
            if self.cnn_parameters['cuda']:
                shuffled = shuffled.cuda()
            self.models[target].eval()
            output = self.models[target](shuffled)
            testloss = mse_loss(output, Y_train)
            testloss = testloss.cpu().data.item()
            
            diff = first_loss - real_loss
            testdiff = first_loss - testloss

            if self.verbose == 2:
                print('diff = first_loss - real_loss')
                print('testdiff = first_loss - testloss')
                print(f'{diff} = {first_loss} - {real_loss}')
                print(f'{testdiff} = {first_loss} - {testloss}')

            if testdiff > (diff * self.cnn_parameters['significance']):
                validated.remove(idx)
        
        # => Step 5: Delay discovery
        weights = []
        
        # Discover time delay between cause and effect
        # by interpreting kernel weights
        for layer in range(self.cnn_parameters['levels']):
            shapes = self.models[target].dwn.network[layer].net[0].weight.size()
            weight = self.models[target].dwn.network[layer].net[0].weight.abs().view(shapes[0], shapes[2])
            weights.append(weight)

        delays = dict()
        target_idx = dataframe.columns.get_loc(target)
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
                delay = index_max * (self.cnn_parameters['dilation_c']**k)
                totaldelay += delay
            if target_idx != v:
                delays[(target_idx, v)] = totaldelay
            else:
                delays[(target_idx, v)] = totaldelay + 1
        if self.verbose == 1:
            print(f'Validated causes: {validated}')
        
        return validated, delays, real_loss, scores.view(-1).cpu().detach().numpy().tolist()
    
    def _preparedata(self, dataframe, target):
        """Reads data from csv file and transforms it to two PyTorch tensors:
        dataset x and target time series y that has to be predicted."""
        df_y = dataframe.copy(deep=True)[[target]]
        df_x = dataframe.copy(deep=True)
        df_y_shift = df_y.copy(deep=True).shift(periods=1, axis=0)
        df_y_shift[target] = df_y_shift[target].fillna(0.)
        df_x[target] = df_y_shift
        data_x = df_x.values.astype('float32').transpose()
        data_y = df_y.values.astype('float32').transpose()
        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)

        x, y = Variable(data_x), Variable(data_y)
        return x, y
    
    def _train(self, model, optimizer, train_data, train_target, epoch):
        """Trains model by performing one epoch and returns attention scores and loss."""

        model.train()
        x, y = train_data[0:1], train_target[0:1]
            
        optimizer.zero_grad()
        output = model(x)

        attention_scores = model.fs_attention
        
        loss = mse_loss(output, y)
        loss.backward()
        optimizer.step()

        if self.verbose == 2 and (epoch % self.cnn_parameters['log_interval'] == 0
                                  or epoch % self.cnn_parameters['num_epochs'] == 0
                                  or epoch == 1):
            print('Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}'.format(
                epoch, epoch / self.cnn_parameters['num_epochs'] * 100, loss))

        return attention_scores.data, loss
    
    def get_causes(self):
        print('\n========================== RESULTS =========================\n')
        for pair in self.delays:
            print(f'{self.columns[pair[1]]} causes {self.columns[pair[0]]} '
                  + f'with a delay of {self.delays[pair]} time steps.')
        print('\n============================================================\n')

    def check_with_ground_truth(self, y):
        """Evaluate TCDF by comparing discovered causes with ground truth"""

        receptivefield = 1
        for level in range(0, self.cnn_parameters['levels']):
            receptivefield += (self.cnn_parameters['kernel_size'] - 1) \
                              * self.cnn_parameters['dilation_c']**(level)
        
        print('\n======================== EVALUATION ========================\n')
        FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = \
            self._evaluate(y)

        # evaluate delay discovery
        extended_delays, readgt, extendedreadgt = self._get_extended_delays(y)
        percentagecorrect = self._evaluate_delay(extended_delays, TPs, receptivefield) * 100
        print(f'Percentage of delays that are correctly discovered: {percentagecorrect} %')
        print('\n============================================================\n')

    def _evaluate(self, gtfile):
        """Evaluates the results of TCDF by comparing it to the ground truth graph,
        and calculating precision, recall and F1-score.
        F1'-score, precision' and recall' include indirect causal relationships."""
        extendedgtdelays, readgt, extendedreadgt = self._get_extended_delays(gtfile)
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
            for v in self.causes[key]:
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
                if v not in self.causes[key]:
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

    def _get_extended_delays(self, gtfile):
        """Collects the total delay of indirect causal relationships."""
        gtdata = pd.read_csv(gtfile, header=None)

        readgt = dict()
        effects = gtdata[1]
        causes = gtdata[0]
        delays = gtdata[2]
        gtnrrelations = 0
        pairdelays = dict()
        for k in range(len(self.columns)):
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
        
        for c1 in range(len(self.columns)):
            for c2 in range(len(self.columns)):
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
    
    def _evaluate_delay(self, extended_gt_delays, TPs, receptive_field):
        """Evaluates the delay discovery of TCDF by comparing the discovered
        time delays with the ground truth."""
        zeros = 0
        total = 0.
        for i in range(len(TPs)):
            tp = TPs[i]
            discovered_delay = self.delays[tp]
            gtdelays = extended_gt_delays[tp]
            for d in gtdelays:
                if d <= receptive_field:
                    total += 1.
                    error = d - discovered_delay
                    if error == 0:
                        zeros += 1
                else:
                    next
            
        return 0. if zeros == 0 else zeros / float(total)

    # self._plotgraph(dataset_name, alldelays, columns)
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
    
    def visualize_weights(self, pointwise=True):
        for target, model in self.models.items():
            # layer_num = len(self.model.dwn.network)
            layer_num = self.cnn_parameters['levels'] + 1
            fig, ax = plt.subplots(1, layer_num + int(pointwise), figsize=(5 * (layer_num + int(pointwise)), 5))
          
            for layer in range(layer_num):
                data = model.dwn.network[layer].net[0].weight.data.numpy()[:, 0, :]

                ax[layer].axis("off")
                ax[layer].imshow(data.T, cmap='viridis')
                for (i, j), z in np.ndenumerate(data):
                    ax[layer].text(j, i, '{:0.5f}'.format(z), ha='center', va='center', color='w')
                ax[layer].set_title(f'Layer {layer}')
          
            if pointwise:
                data = model.pointwise.weight.data.numpy()[0, :, :]

                ax[layer_num].axis("off")
                ax[layer_num].imshow(data, cmap='viridis')
                for (i, j), z in np.ndenumerate(data):
                    ax[layer_num].text(j, i, '{:0.5f}'.format(z), ha='center', va='center', color='w')
                ax[layer_num].set_title('Pointwise conv')
          
            fig.suptitle(f'CNN for target: {target}')
            plt.show(fig)
