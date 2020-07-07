import torch
import torch.optim
from torch.nn.functional import mse_loss
# from torch.autograd import Variable
from model import ADDSTCN
import random
import heapq
import pandas as pd
import numpy as np
import networkx as nx
# from networkx.drawing.nx_agraph import to_agraph
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
    
    def fit(self, X, normalize='std', verbose=None):
        """Loops through all variables in a dataset and return the discovered
        causes, time delays, losses, attention scores and variable names."""
        self.models = {}
        if verbose is not None:
            self.verbose = verbose
        
        if type(X) is str:
            X = pd.read_csv(X)
        
        if normalize == 'std':
            X = (X - X.mean()) / X.std()
        elif normalize == 'minmax':
            X = (X - X.min()) / (X.max() - X.min())
        
        self.causes = dict()
        self.delays = dict()
        self.logs = {'real_losses': dict(), 'scores': dict()}

        self.columns = X.columns.to_numpy(dtype=str)
        for c in self.columns:
            idx = X.columns.get_loc(c)
            causes, delays, real_loss, scores = self._find_causes(X, c)

            self.causes[idx] = causes
            self.delays.update(delays)
            self.logs['real_losses'][idx] = real_loss
            self.logs['scores'][idx] = scores
    
    def _find_causes(self, dataframe, target):
        """Discovers potential causes of one target time series, validates these
        potential causes with PIVM and discovers the corresponding time delays
        """

        if self.verbose >= 1:
            print('\nAnalysis started for target: ', target)
        
        # => Step 1: Preparing datasets and creating CNN with given parameters
        torch.manual_seed(self.cnn_parameters['seed'])
        
        X_train, Y_train = self._prepare_data(dataframe, target)
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
        self.models[target].eval()
        first_loss = mse_loss(self.models[target](X_train), Y_train)
        first_loss = first_loss.cpu().detach().item()
        
        for epoch in range(1, self.cnn_parameters['num_epochs'] + 1):
            scores, real_loss = self._train(self.models[target], optimizer, X_train, Y_train, epoch)
        real_loss = real_loss.cpu().detach().item()
        
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
        
        if self.verbose >= 1:
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
            test_loss = mse_loss(output, Y_train)
            test_loss = test_loss.cpu().detach().item()
            
            diff = first_loss - real_loss
            testdiff = first_loss - test_loss

            if self.verbose == 2:
                print('\n⎡ diff = first_loss - real_loss')
                print(f'| {diff} = {first_loss} - {real_loss}')
                print('|')
                print('| testdiff = first_loss - test_loss')
                print(f'⎣ {testdiff} = {first_loss} - {test_loss}\n')

            if testdiff > (diff * self.cnn_parameters['significance']):
                validated.remove(idx)
        
        # => Step 5: Delay discovery
        weights = []
        
        # Discover time delay between cause and effect
        # by interpreting kernel weights
        for layer in range(self.cnn_parameters['levels']):
            shapes = self.models[target].depthwise[layer].conv1.weight.size()
            weight = self.models[target].depthwise[layer].conv1.weight.abs().view(shapes[0], shapes[2])
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
        if self.verbose >= 1:
            print(f'Validated causes: {validated}')
        
        return validated, delays, real_loss, scores.view(-1).cpu().detach().numpy().tolist()
    
    def _prepare_data(self, dataframe, target):
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

        return data_x, data_y
        # x, y = Variable(data_x), Variable(data_y)
        # return x, y
    
    def _train(self, model, optimizer, train_data, train_target, epoch):
        """Trains model by performing one epoch and returns attention scores and loss."""

        model.train()
        x, y = train_data[0:1], train_target[0:1]
            
        optimizer.zero_grad()
        output = model(x)
        
        loss = mse_loss(output, y)
        loss.backward()
        optimizer.step()

        if self.verbose == 2 and (epoch % self.cnn_parameters['log_interval'] == 0
                                  or epoch % self.cnn_parameters['num_epochs'] == 0
                                  or epoch == 1):
            print('Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}'.format(
                epoch, epoch / self.cnn_parameters['num_epochs'] * 100, loss))

        return model.fs_attention.detach(), loss
    
    def get_causes(self):
        print('\n========================== RESULTS =========================\n')
        for pair in self.delays:
            print(f'{self.columns[pair[1]]} causes {self.columns[pair[0]]} '
                  + f'with a delay of {self.delays[pair]} time steps.')
        print('\n============================================================\n')

    def check_with_ground_truth(self, y, normalize=True):
        """Evaluate TCDF by comparing discovered causes with ground truth"""
        if type(y) is str:
            y = pd.read_csv(y)

        self.stats = self._calculate_stats(y)
        print('\n======================== EVALUATION ========================\n')
        print(f"⎡ Total connections: {self.stats['Total connections']}")
        print('|')
        print(f"| Correct connections: {self.stats['Correct connections']}")
        print(f"| Incorrect connections: {self.stats['Incorrect connections'][0], self.stats['Incorrect connections'][1]}")
        print(f"| Incorrect directions: {self.stats['Incorrect directions'][0], self.stats['Incorrect directions'][1]}")
        print(f"⎣ Undetected connections: {self.stats['Undetected connections'][0], self.stats['Undetected connections'][1]}\n")

        print('⎡ Delays')
        print(f"|   Correct: {self.stats['Delays']['Correct']}")
        print(f"|   Incorrect AND correct direction: {self.stats['Delays']['Incorrect AND correct direction']}")
        print(f"⎣   Correct AND incorrect direction: {self.stats['Delays']['Correct AND incorrect direction']}\n")

        print('\n============================================================\n')

    def _calculate_stats(self, dataframe):
        """Evaluates the results of TCDF by comparing it to the ground truth graph.
        """
        num_connections = len(dataframe)
        gt_np = dataframe.to_numpy()
        gt_connections = [(dataframe['cause'][i], dataframe['effect'][i])
                          for i in range(num_connections)]

        stats = {}
        stats['Total connections'] = num_connections
        stats['Correct connections'] = 0
        stats['Incorrect connections'] = [0, set()]
        stats['Incorrect directions'] = [0, set()]
        stats['Undetected connections'] = [0, set()]

        stats['Delays'] = {}
        stats['Delays']['Correct'] = 0
        stats['Delays']['Incorrect AND correct direction'] = 0
        stats['Delays']['Correct AND incorrect direction'] = 0

        for c, e, d in gt_np:
            if (c, e) in self.delays:
                stats['Correct connections'] += 1
                if self.delays[(c, e)] == d:
                    stats['Delays']['Correct'] += 1
                else:
                    stats['Delays']['Incorrect AND correct direction'] += 1
            elif (e, c) in self.delays:
                stats['Correct connections'] += 1
                stats['Incorrect directions'][1].add((c, e))
                if self.delays[(e, c)] == d:
                    stats['Delays']['Correct'] += 1
                    stats['Delays']['Correct AND incorrect direction'] += 1
            else:
                stats['Undetected connections'][1].add((c, e))

        for c, e in self.delays.keys():
            if (c, e) not in gt_connections and (e, c) not in gt_connections:
                stats['Incorrect connections'][1].add((c, e))

        stats['Incorrect connections'][0] = len(stats['Incorrect connections'][1])
        stats['Incorrect directions'][0] = len(stats['Incorrect directions'][1])
        stats['Undetected connections'][0] = len(stats['Undetected connections'][1])

        return stats

    def plot_graph(self, ground_truth=None, print_delays=False):
        """Plots a temporal causal graph showing all discovered causal relationships
        annotated with the time delay between cause and effect.
        
        https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx_edge_labels.html
        """
        if print_delays:
            print(self.delays)
        
        # TCDF graph
        G = nx.DiGraph()
        nodes = list(self.columns)
        for idx, node in enumerate(self.columns):
            if (idx, idx) in self.delays:
                nodes[idx] += f' ({self.delays[(idx, idx)]})'
            G.add_node(nodes[idx])
        for pair in self.delays:
            p1, p2 = pair
            nodepair = (nodes[p2], nodes[p1])

            G.add_edges_from([nodepair], weight=self.delays[pair])

        edge_labels = dict([((u, v, ), d['weight'])
                            for u, v, d in G.edges(data=True)])

        pos = nx.circular_layout(G)

        
        fig1 = plt.figure(figsize=(10, 5))
        ax = fig1.add_subplot(111)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
        nx.draw(G, pos, node_color='white', edge_color='black', node_size=2000, with_labels=True)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000")
        # G = to_agraph(G)
        # G.layout('dot')
        # G.draw('graph.png')

        # ground truth graph
        if ground_truth is not None:
            if type(ground_truth) is str:
                ground_truth = pd.read_csv(ground_truth)
            dataframe = ground_truth
            causes = dataframe['cause'].values
            effects = dataframe['effect'].values
            delays = dataframe['delay'].values
            delays = {(c, e): d for c, e, d in zip(causes, effects, delays)}
            
            G = nx.DiGraph()

            nodes = list(self.columns)
            for idx, node in enumerate(self.columns):
                if (idx, idx) in delays:
                    nodes[idx] += f' ({delays[(idx, idx)]})'
                G.add_node(nodes[idx])
            for pair in delays:
                p1, p2 = pair
                nodepair = (nodes[p2], nodes[p1])

                G.add_edges_from([nodepair], weight=delays[pair])

            edge_labels = dict([((u, v, ), d['weight'])
                                for u, v, d in G.edges(data=True)])

            pos = nx.circular_layout(G)

            fig2 = plt.figure(figsize=(10, 5))
            ax = fig2.add_subplot(111)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
            nx.draw(G, pos, node_color='white', edge_color='black', node_size=2000, with_labels=True)
            ax = plt.gca()
            ax.collections[0].set_edgecolor("#000000")
            # G = to_agraph(G)
            # G.layout('dot')
            # G.draw('graph.png')
        
        plt.show()
    
    def visualize_weights(self, pointwise=True, attention_scores=True, cmap='viridis'):
        for target, model in self.models.items():
            # layer_num = len(self.model.dwn.network)
            layer_num = self.cnn_parameters['levels'] + 1
            figsize_base = len(self.columns) + layer_num
            fig, ax = plt.subplots(
                1,
                layer_num + int(pointwise) + int(attention_scores),
                figsize=(figsize_base * (layer_num + int(pointwise) + int(attention_scores) + 1), figsize_base))

            if attention_scores:
                data = model.fs_attention.detach().numpy()

                ax[0].axis("off")
                ax[0].imshow(data, cmap=cmap)
                for (i, j), z in np.ndenumerate(data):
                    ax[0].text(j, i, '{:0.5f}'.format(z), ha='center', va='center', color='b',
                               bbox=dict(facecolor='w', alpha=1.0))
                ax[0].set_title('Attention scores')
            
            for layer in range(layer_num):
                data = model.dwn.network[layer].net[0].weight.detach().numpy()[:, 0, :]

                idx = layer + int(attention_scores)
                ax[idx].axis("off")
                ax[idx].imshow(data, cmap=cmap)
                for (i, j), z in np.ndenumerate(data):
                    ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center', color='b',
                                 bbox=dict(facecolor='w', alpha=1.0))
                ax[idx].set_title(f'Layer {layer}')
          
            if pointwise:
                idx = layer_num + int(attention_scores)
                data = model.pointwise.weight.detach().numpy()[0, :, :]

                ax[idx].axis("off")
                ax[idx].imshow(data, cmap=cmap)
                for (i, j), z in np.ndenumerate(data):
                    ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center', color='b',
                                 bbox=dict(facecolor='w', alpha=1.0))
                ax[idx].set_title('Pointwise conv')
          
            fig.suptitle(f'CNN for target: {target}')
            plt.show(fig)
