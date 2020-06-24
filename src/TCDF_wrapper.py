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
        Recommended to be equal to dilation coefficient
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
                   Set kernel_size equal to dilation_c to have exactly one path for each delay.')
        
        receptive_field = 1
        for level in range(0, hidden_layers + 1):
            receptive_field += (kernel_size - 1) * dilation_coefficient**(level)
        
        self.cnn_parameters = {'cuda': cuda,
                               'num_epochs': epochs,
                               'kernel_size': kernel_size,
                               'levels': hidden_layers + 1,
                               'dilation_c': dilation_coefficient,
                               'receptive_field': receptive_field,
                               'significance': significance,
                               'learning_rate': learning_rate,
                               'optimizer': optimizer,
                               'log_interval': log_interval,
                               'seed': seed}
        self.models = None
        self.verbose = None
    
    def fit(self, X_path, verbose=None):
        """Loops through all variables in a dataset and return the discovered
        causes, time delays, losses, attention scores and variable names."""
        self.models = {}
        if verbose is not None:
            self.verbose = verbose
        
        dataframe = pd.read_csv(X_path)

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

    def check_with_ground_truth(self, y_path):
        """Evaluate TCDF by comparing discovered causes with ground truth"""
        dataframe = pd.read_csv(y_path, header=None)

        print('\n======================== EVALUATION ========================\n')
        self.stats = self._calculate_stats(dataframe)
        
        print('⎡ Counts')
        print('|')
        print(f"| True Positives: {self.stats['num_true_positives']}")
        print(f"| False Positives: {self.stats['num_false_positives']}")
        print(f"| False Negatives: {self.stats['num_false_negatives']}")
        print(f"| Direct True Positives: {self.stats['num_true_positives_direct']}")
        print(f"⎣ Direct False Positives: {self.stats['num_false_positives_direct']}\n")

        print(f"⎡ True Positives: {self.stats['true_positives']}")
        print(f"| False Positives: {self.stats['false_positives']}")
        print(f"| False Negatives: {self.stats['false_negatives']}")
        print(f"| True Positives (direct): {self.stats['true_positives_direct']}")
        print(f"⎣ False Positives (direct): {self.stats['false_positives_direct']}\n")

        print('⎡ F1 score (includes direct and indirect causal relationships):')
        print(f"| {self.stats['f1_score']} (precision: {self.stats['precision']}, recall: {self.stats['recall']})")
        print('|')
        print('| F1 score (direct) (includes only direct causal relationships):')
        print(f"⎣ {self.stats['f1_score_direct']} (precision: {self.stats['precision_direct']}, recall: {self.stats['recall_direct']})\n")
        
        print(f'[ Percentage of delays that are correctly discovered: {self._evaluate_delay(dataframe) * 100} %')
        print('\n============================================================\n')

    def _calculate_stats(self, dataframe):
        """Evaluates the results of TCDF by comparing it to the ground truth graph,
        and calculating precision, recall and F1-score.
        F1'-score, precision' and recall' include indirect causal relationships."""
        gt, ext_gt, _ = self._extract_ground_truth_information(dataframe)
        

        true_positives, false_positives, false_negatives = [], [], []
        true_positives_direct, false_positives_direct = [], []
        f1_score = f1_score_direct = 0

        for key in gt:
            for value in self.causes[key]:
                if value in ext_gt[key]:
                    true_positives.append((key, value))
                else:
                    false_positives.append((key, value))
                if value in gt[key]:
                    true_positives_direct.append((key, value))
                else:
                    false_positives_direct.append((key, value))
            for value in gt[key]:
                if value not in self.causes[key]:
                    false_negatives.append((key, value))
        
        num_true_positives = len(true_positives)
        num_false_positives = len(false_positives)
        num_false_negatives = len(false_negatives)
        num_true_positives_direct = len(true_positives_direct)
        num_false_positives_direct = len(false_positives_direct)

        # F1-score calculation
        if 0 not in [num_true_positives + num_false_positives,
                     num_true_positives + num_false_negatives]:
            precision = num_true_positives / (num_true_positives + num_false_positives)
            recall = num_true_positives / (num_true_positives + num_false_negatives)
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        # F1-score direct calculation
        if 0 not in [num_true_positives_direct + num_false_positives_direct,
                     num_true_positives_direct + num_false_negatives]:
            precision_direct = num_true_positives_direct / (num_true_positives + num_false_positives_direct)
            recall_direct = num_true_positives_direct / (num_true_positives_direct + num_false_negatives)
            f1_score_direct = 2 * (precision_direct * recall_direct) / (precision_direct + recall_direct)
        else:
            f1_score_direct = 0
        
        return {'num_true_positives': num_true_positives,
                'num_false_positives': num_false_positives,
                'num_false_negatives': num_false_negatives,
                'num_true_positives_direct': num_true_positives_direct,
                'num_false_positives_direct': num_false_positives_direct,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'true_positives_direct': true_positives_direct,
                'false_positives_direct': false_positives_direct,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall,
                'f1_score_direct': f1_score_direct,
                'precision_direct': precision_direct,
                'recall_direct': recall_direct}

    def _extract_ground_truth_information(self, dataframe):
        # gt — ground truth
        # ext_gt — extended ground truth

        gt = dict((idx, []) for idx in range(len(self.columns)))
        causes = dataframe[0]
        effects = dataframe[1]

        for i in range(len(effects)):
            gt[effects[i]].append(causes[i])

        g = nx.DiGraph()
        g.add_nodes_from(gt.keys())
        for effect in gt:
            causes = gt[effect]
            for cause in causes:
                g.add_edge(cause, effect)

        ext_gt = copy.deepcopy(gt)
        
        for c1 in range(len(self.columns)):
            for c2 in range(len(self.columns)):
                # indirect path max length 3, no cycles
                paths = list(nx.all_simple_paths(g, c1, c2, cutoff=2))
                
                if len(paths) > 0:
                    for path in paths:
                        for p in path[:-1]:
                            if p not in ext_gt[path[-1]]:
                                ext_gt[path[-1]].append(p)
        return gt, ext_gt, g

    def _get_extended_delays(self, dataframe):
        """Collects the total delay of indirect causal relationships."""
        causes = dataframe[0]
        effects = dataframe[1]
        delays = dataframe[2]
        pairdelays = dict(((effects[i], causes[i]), delays[i]) for i in range(len(effects)))
        gt, ext_gt, g = self._extract_ground_truth_information(dataframe)

        ext_gt_delays = dict()
        for effect in ext_gt:
            causes = ext_gt[effect]
            for cause in causes:
                if (effect, cause) in pairdelays:
                    delay = pairdelays[(effect, cause)]
                    ext_gt_delays[(effect, cause)] = [delay]
                else:
                    # find extended delay
                    # indirect path max length 3, no cycles
                    paths = list(nx.all_simple_paths(g, cause, effect, cutoff=2))
                    ext_gt_delays[(effect, cause)] = []
                    for p in paths:
                        delay = 0
                        for i in range(len(p) - 1):
                            delay += pairdelays[(p[i + 1], p[i])]
                        ext_gt_delays[(effect, cause)].append(delay)

        return ext_gt_delays
    
    def _evaluate_delay(self, dataframe):
        """Evaluates the delay discovery of TCDF by comparing the discovered
        time delays with the ground truth."""
        ext_gt_delays = self._get_extended_delays(dataframe)

        zeros = total = 0
        for i in range(len(self.stats['true_positives'])):
            tp = self.stats['true_positives'][i]
            discovered_delay = self.delays[tp]
            gt_delays = ext_gt_delays[tp]
            for d in gt_delays:
                if d <= self.cnn_parameters['receptive_field']:
                    total += 1.
                    error = d - discovered_delay
                    if error == 0:
                        zeros += 1
                else:
                    next
            
        return 0. if zeros == 0 else zeros / total

    def plotgraph(self):
        """Plots a temporal causal graph showing all discovered causal relationships
        annotated with the time delay between cause and effect."""
        G = nx.DiGraph()
        for c in self.columns:
            G.add_node(c)
        for pair in self.delays:
            p1, p2 = pair
            nodepair = (self.columns[p2], self.columns[p1])

            G.add_edges_from([nodepair], weight=self.delays[pair])

        edge_labels = dict([((u, v, ), d['weight'])
                            for u, v, d in G.edges(data=True)])

        pos = nx.circular_layout(G)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.draw(G, pos, node_color='white', edge_color='black', node_size=1000, with_labels=True)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000")

        plt.show()
    
    def visualize_weights(self, pointwise=True, attention_scores=True):
        for target, model in self.models.items():
            # layer_num = len(self.model.dwn.network)
            layer_num = self.cnn_parameters['levels'] + 1
            fig, ax = plt.subplots(
                1,
                layer_num + int(pointwise) + int(attention_scores),
                figsize=(5 * (layer_num + int(pointwise) + int(attention_scores) + 1), 5))

            if attention_scores:
                data = model.fs_attention.data.numpy()

                ax[0].axis("off")
                ax[0].imshow(data, cmap='viridis')
                for (i, j), z in np.ndenumerate(data):
                    ax[0].text(j, i, '{:0.5f}'.format(z), ha='center', va='center', color='w')
                ax[0].set_title('Attention scores')
            
            for layer in range(layer_num):
                data = model.dwn.network[layer].net[0].weight.data.numpy()[:, 0, :]

                idx = layer + int(attention_scores)
                ax[idx].axis("off")
                ax[idx].imshow(data.T, cmap='viridis')
                for (i, j), z in np.ndenumerate(data):
                    ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center', color='w')
                ax[idx].set_title(f'Layer {layer}')
          
            if pointwise:
                idx = layer_num + int(attention_scores)
                data = model.pointwise.weight.data.numpy()[0, :, :]

                ax[idx].axis("off")
                ax[idx].imshow(data, cmap='viridis')
                for (i, j), z in np.ndenumerate(data):
                    ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center', color='w')
                ax[idx].set_title('Pointwise conv')
          
            fig.suptitle(f'CNN for target: {target}')
            plt.show(fig)
