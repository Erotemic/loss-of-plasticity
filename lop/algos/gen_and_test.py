# import sys, ubelt  # NOQA
# sys.path.append(ubelt.expandpath('~/code/loss-of-plasticity'))  # NOQA
import ubelt as ub
import rich
from collections import defaultdict
from lop.utils.AdamGnT import AdamGnT
from math import sqrt
from torch.nn import Conv2d, Linear
import numpy as np
import torch
from torch.nn.init import calculate_gain


def get_layer_bound(layer, init, gain):
    if isinstance(layer, Conv2d):
        return sqrt(1 / (layer.in_channels *
                    layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


KNOWN_LAYERS = (torch.nn.Linear, torch.nn.Conv2d)


def test_complex():
    from watch.tasks.fusion.methods.channelwise_transformer import MultimodalTransformer
    from watch.tasks.fusion import datamodules
    print('(STEP 0): SETUP THE DATA MODULE')
    datamodule = datamodules.KWCocoVideoDataModule(
        train_dataset='special:vidshapes-watch', num_workers=4, channels='auto')
    datamodule.setup('fit')
    dataset = datamodule.torch_datasets['train']
    print('(STEP 1): ESTIMATE DATASET STATS')
    dataset_stats = dataset.cached_dataset_stats(num=3)
    print('dataset_stats = {}'.format(ub.urepr(dataset_stats, nl=3)))
    loader = datamodule.train_dataloader()
    print('(STEP 2): SAMPLE BATCH')
    batch = next(iter(loader))
    for item_idx, item in enumerate(batch):
        print(f'item_idx={item_idx}')
        item_summary = dataset.summarize_item(item)
        print('item_summary = {}'.format(ub.urepr(item_summary, nl=2)))
    print('(STEP 3): THE REST OF THE TEST')
    #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
    net = MultimodalTransformer(arch_name='smt_it_joint_p1',
                                dataset_stats=dataset_stats,
                                classes=datamodule.classes,
                                decoder='mlp', change_loss='dicefocal',
                                attention_impl='exact')

    meta = MetaNetwork(net)
    meta._build()
    input_data = batch[0:1]
    meta.trace(input_data=input_data)
    meta.tv_graph.visual_graph.render(format='png')

    ### DEV
    for layer_name in meta.layer_names:
        next_name = list([t[0] for t in meta.next_layers(layer_name)])
        print(f'connections {layer_name} -> {next_name}')

    from torch.optim import AdamW
    opt = AdamW(net.parameters())
    net(batch)
    self = GenerateAndTest(net, 'relu', opt, input_data)
    self.gen_and_test()

    # Do a forward pass so activations are populated
    net.train()
    for i in ub.ProgIter(range(100)):
        outputs = net(input_data)
        loss = outputs['loss']
        # loss = outputs.sum()
        loss.backward()
        opt.zero_grad()
        opt.step()
        self.gen_and_test()


class GenerateAndTest(object):
    """
    Generate-and-Test algorithm for feed forward neural networks, based on
    maturity-threshold based replacement

    CommandLine:
        xdoctest -m lop/algos/gen_and_test.py GenerateAndTest

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/loss-of-plasticity'))
        >>> from lop.algos.gen_and_test import *  # NOQA
        >>> import torchvision
        >>> net = torchvision.models.resnet18()
        >>> from torch.optim import AdamW
        >>> opt = AdamW(net.parameters())
        >>> hidden_activation = 'relu'
        >>> #inputs = torch.rand(2, 3, 224, 224)
        >>> #outputs = net(inputs)
        >>> #loss = outputs.sum()
        >>> #loss.backward()
        >>> #opt.step()
        >>> #opt.zero_grad()
        >>> input_data = torch.rand(2, 3, 224, 224)
        >>> self = GenerateAndTest(net, hidden_activation, opt, input_data=input_data)
        >>> # Do a forward pass so activations are populated
        >>> for i in ub.ProgIter(range(100)):
        >>>     outputs = net(input_data)
        >>>     loss = outputs.sum()
        >>>     loss.backward()
        >>>     opt.zero_grad()
        >>>     opt.step()
        >>>     self.gen_and_test()

        opt.param_groups[0]['params']
    """

    def __init__(
            self,
            net,
            hidden_activation,
            opt,
            input_data,
            decay_rate=0.99,
            replacement_rate=1e-4,
            init='kaiming',
            device="cpu",
            maturity_threshold=20,
            util_type='contribution',
            num_last_filter_outputs=4,
            accumulate=False,
    ):
        super().__init__()
        self.device = device
        self.net = net

        self.meta = MetaNetwork(net)
        self.meta._build()
        self.meta.trace(input_data)

        self.accumulate = accumulate
        self.num_last_filter_outputs = num_last_filter_outputs

        self.opt = opt
        self.opt_type = 'sgd'
        if isinstance(self.opt, AdamGnT):
            self.opt_type = 'adam'

        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        """
        Utility of all features/neurons
        """
        self.util = {}
        self.bias_corrected_util = {}
        self.ages = {}
        self.mean_feature_act = {}
        self.mean_abs_feature_act = {}
        self.accumulated_num_features_to_replace = {}

        self.tracked_layer_names = []

        for name, layer in self.meta.named_layers:
            # FIXME: multiple inputs to the same layer may need to be handled
            # differently.

            if isinstance(layer, Conv2d):
                self.util[name] = torch.zeros(layer.out_channels)
                self.bias_corrected_util[name] = torch.zeros(layer.out_channels)
                self.ages[name] = torch.zeros(layer.out_channels)
                self.mean_feature_act[name] = torch.zeros(layer.out_channels)
                self.mean_abs_feature_act[name] = torch.zeros(
                    layer.out_channels)
                self.tracked_layer_names.append(name)
            elif isinstance(layer, Linear):
                self.util[name] = torch.zeros(layer.out_features)
                self.bias_corrected_util[name] = torch.zeros(layer.out_features)
                self.ages[name] = torch.zeros(layer.out_features)
                self.mean_feature_act[name] = torch.zeros(layer.out_features)
                self.mean_abs_feature_act[name] = torch.zeros(
                    layer.out_features)
                self.tracked_layer_names.append(name)
            self.accumulated_num_features_to_replace[name] = 0

        """
        Calculate uniform distribution's bound for random feature initialization
        """
        self.bounds = self.compute_bounds(
            hidden_activation=hidden_activation, init=init)

        """
        Pre calculate number of features to replace per layer per update
        """
        self.num_new_features_to_replace = {}
        for name in self.tracked_layer_names:
            layer = self.meta.name_to_layer[name]
            if isinstance(layer, Linear):
                self.num_new_features_to_replace[name] = (
                    self.replacement_rate * layer.out_features)
            elif isinstance(layer, Conv2d):
                self.num_new_features_to_replace[name] = (
                    self.replacement_rate * layer.out_channels)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        if hidden_activation == 'selu':
            init = 'lecun'
        if hidden_activation in ['swish', 'elu']:
            hidden_activation = 'relu'
        bounds = {}
        gain = calculate_gain(nonlinearity=hidden_activation)
        for name in self.tracked_layer_names:
            layer = self.meta.name_to_layer[name]
            bound = get_layer_bound(layer=layer, init=init, gain=gain)
            bounds[name] = bound
        # # TODO: bounds seem to need before / after for each layer
        # bounds.append(get_layer_bound(layer=self.net[-1], init=init, gain=1))
        return bounds

    @staticmethod
    def _layer_type(layer):
        if isinstance(layer, Linear):
            typename = 'Linear'
        elif isinstance(layer, Conv2d):
            typename = 'Conv2d'
        else:
            raise AssertionError
        return typename

    def update_utility(self, layer_name, feature):
        with torch.no_grad():
            self.util[layer_name] *= self.decay_rate

            preserve_rate = (1 - self.decay_rate)
            bias_correction = 1 - (self.decay_rate ** self.ages[layer_name])

            curr_layer = self.meta.name_to_layer[layer_name]
            curr_type = self._layer_type(curr_layer)

            for next_name, next_layer in self.meta.next_layers(layer_name):
                # Not sure how to deal with multiple outputs here.

                next_type = self._layer_type(next_layer)
                try:
                    if next_type == 'Linear':
                        output_wight_mag = next_layer.weight.data.abs().mean(dim=0)
                    elif next_type == 'Conv2d':
                        output_wight_mag = next_layer.weight.data.abs().mean(dim=(0, 2, 3))
                    else:
                        raise NotImplementedError

                    self.mean_feature_act[layer_name] *= self.decay_rate
                    self.mean_abs_feature_act[layer_name] *= self.decay_rate

                    if curr_type == 'Linear':
                        input_wight_mag = curr_layer.weight.data.abs().mean(dim=1)
                    elif curr_type == 'Conv2d':
                        input_wight_mag = curr_layer.weight.data.abs().mean(dim=(1, 2, 3))
                    else:
                        raise AssertionError

                    if curr_type == 'Linear':
                        self.mean_feature_act[layer_name] -= -preserve_rate * feature.mean(dim=0)
                        self.mean_abs_feature_act[layer_name] -= -preserve_rate * feature.abs().mean(dim=0)
                    elif curr_type == 'Conv2d' and next_type == 'Conv2d':
                        self.mean_feature_act[layer_name] -= -preserve_rate * feature.mean(dim=(0, 2, 3))
                        self.mean_abs_feature_act[layer_name] -= -preserve_rate * feature.abs().mean(dim=(0, 2, 3))
                    elif curr_type == 'Conv2d' and next_type == 'Linear':
                        self.mean_feature_act[layer_name] -= -(preserve_rate * feature.mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1))
                        self.mean_abs_feature_act[layer_name] -= -(preserve_rate * feature.abs().mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1))
                    else:
                        raise AssertionError

                    bias_corrected_act = self.mean_feature_act[layer_name] / bias_correction

                    if self.util_type == 'adaptation':
                        new_util = 1 / input_wight_mag
                    elif self.util_type in ['contribution', 'zero_contribution', 'adaptable_contribution']:
                        if self.util_type == 'contribution':
                            bias_corrected_act = 0
                        else:
                            if curr_type == 'Conv2d' and next_type == 'Conv2d':
                                bias_corrected_act = bias_corrected_act.view(1, -1, 1, 1)
                            elif curr_type == 'Conv2d' and next_type == 'Linear':
                                bias_corrected_act = bias_corrected_act.repeat_interleave(
                                    self.num_last_filter_outputs).view(1, -1)

                        if curr_type == 'Linear' and next_type == 'Linear':
                            new_util = output_wight_mag * (feature - bias_corrected_act).abs().mean(dim=0)
                        elif curr_type == 'Conv2d' and next_type == 'Linear':
                            new_util = (output_wight_mag * (feature - bias_corrected_act).abs().mean(dim=0)).view(-1, self.num_last_filter_outputs).mean(dim=1)
                        elif next_type == 'Conv2d':
                            new_util = output_wight_mag * (feature - bias_corrected_act).abs().mean(dim=(0, 2, 3))

                        if self.util_type == 'adaptable_contribution':
                            new_util = new_util / input_wight_mag

                    if self.util_type == 'random':
                        self.bias_corrected_util[layer_name] = torch.rand(self.util[layer_name].shape)
                    else:
                        self.util[layer_name] -= -preserve_rate * new_util
                        # correct the bias in the utility computation
                        self.bias_corrected_util[layer_name] = self.util[layer_name] / bias_correction
                except Exception as ex:
                    e = ex
                    print(f'warning: {layer_name} -> {next_name}, e={e}')
                    # raise

    def test_features(self, features):
        """
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features_to_replace = {n: torch.empty(0, dtype=torch.long).to(self.device)
                               for n in self.tracked_layer_names}
        features_to_replace_input_indices = {n: torch.empty(0, dtype=torch.long)
                                             for n in self.tracked_layer_names}
        features_to_replace_output_indices = {n: torch.empty(0, dtype=torch.long)
                                              for n in self.tracked_layer_names}
        num_features_to_replace = {n: 0 for n in self.tracked_layer_names}
        if self.replacement_rate == 0:
            return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace

        for layer_name, feature in features.items():
            self.ages[layer_name] += 1
            """
            Update feature utility
            """
            self.update_utility(layer_name, feature)

            """
            Find the no. of features to replace
            """
            eligible_feature_indices = torch.where(self.ages[layer_name] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue
            num_new_features_to_replace = self.replacement_rate * eligible_feature_indices.shape[0]
            self.accumulated_num_features_to_replace[layer_name] += num_new_features_to_replace

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            if self.accumulate:
                num_new_features_to_replace = int(self.accumulated_num_features_to_replace[layer_name])
                self.accumulated_num_features_to_replace[layer_name] -= num_new_features_to_replace
            else:
                if num_new_features_to_replace < 1:
                    if torch.rand(1) <= num_new_features_to_replace:
                        num_new_features_to_replace = 1
                num_new_features_to_replace = int(num_new_features_to_replace)

            if num_new_features_to_replace == 0:
                continue

            """
            Find features to replace in the current layer
            """
            try:
                layer_bias_correction = self.bias_corrected_util[layer_name]
                _tmp = -layer_bias_correction[eligible_feature_indices]
                new_features_to_replace = torch.topk(_tmp, num_new_features_to_replace)[1]
                new_features_to_replace = eligible_feature_indices[new_features_to_replace]
            except Exception as ex:  # NOQA
                raise
                # rich.print('\n' + ub.codeblock(
                #     f'''
                #     [red]fixme {layer_name} with feature.shape={feature.shape},
                #     num_new_features_to_replace={num_new_features_to_replace},
                #     eligible_feature_indices={eligible_feature_indices.shape}
                #     layer_bias_correction.shape={layer_bias_correction.shape}
                #     ex={ex!r}
                #     '''))
            else:
                """
                Initialize utility for new features
                """
                # rich.print('\n' + ub.codeblock(
                #     f'''
                #     [green]good {layer_name} with feature.shape={feature.shape},
                #     num_new_features_to_replace={num_new_features_to_replace},
                #     eligible_feature_indices={eligible_feature_indices.shape}
                #     layer_bias_correction.shape={layer_bias_correction.shape}
                #     '''))
                self.util[layer_name][new_features_to_replace] = 0
                self.mean_feature_act[layer_name][new_features_to_replace] = 0.

                features_to_replace[layer_name] = new_features_to_replace
                num_features_to_replace[layer_name] = num_new_features_to_replace

        return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace

    def gen_new_features(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with torch.no_grad():
            for layer_name in self.tracked_layer_names:

                num = num_features_to_replace[layer_name]

                if num == 0:
                    continue

                curr_layer = self.meta.name_to_layer[layer_name]
                in_feat_idx = features_to_replace_input_indices[layer_name]
                out_feat_idx = features_to_replace_output_indices[layer_name]
                bound = self.bounds[layer_name]

                if isinstance(curr_layer, Linear):
                    curr_layer.weight.data[in_feat_idx, :] *= 0.0
                    curr_layer.weight.data[in_feat_idx, :] -= -(
                        torch.empty(num, curr_layer.in_features).uniform_(
                            -bound, bound).to(self.device))
                elif isinstance(curr_layer, Conv2d):
                    _shape = [num] + list(curr_layer.weight.shape[1:])
                    curr_layer.weight.data[in_feat_idx, :] *= 0.0
                    curr_layer.weight.data[in_feat_idx, :] -= -(
                        torch.empty(_shape).uniform_(-bound, bound))
                else:
                    raise AssertionError('should be linear or conv2d for now')

                if curr_layer.bias is not None:
                    curr_layer.bias.data[in_feat_idx] *= 0.0

                # Set ages to zero
                self.ages[layer_name][in_feat_idx] = 0

                # Set the outgoing weights to zero
                for next_name, next_layer in self.meta.next_layers(layer_name):
                    next_layer.weight.data[:, out_feat_idx] = 0

    def update_optim_params(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace):
        """
        Update Optimizer's state
        """
        if self.opt_type == 'adam':
            for layer_name in self.tracked_layer_names:

                # input weights
                if num_features_to_replace == 0:
                    continue

                layer = self.meta.name_to_layer[layer_name]

                for next_name, next_layer in self.meta.next_layers(layer_name):

                    in_feat_idx = features_to_replace_input_indices[layer_name]
                    out_feat_idx = features_to_replace_output_indices[layer_name]
                    self.opt.state[layer.weight]['exp_avg'][in_feat_idx, :] = 0.0
                    self.opt.state[layer.bias]['exp_avg'][in_feat_idx] = 0.0
                    self.opt.state[layer.weight]['exp_avg_sq'][in_feat_idx, :] = 0.0
                    self.opt.state[layer.bias]['exp_avg_sq'][in_feat_idx] = 0.0
                    self.opt.state[layer.weight]['step'][in_feat_idx, :] = 0
                    self.opt.state[layer.bias]['step'][in_feat_idx] = 0
                    # output weights
                    self.opt.state[next_layer.weight]['exp_avg'][:, out_feat_idx] = 0.0
                    self.opt.state[next_layer.weight]['exp_avg_sq'][:, out_feat_idx] = 0.0
                    self.opt.state[next_layer.weight]['step'][:, out_feat_idx] = 0

    def gen_and_test(self):
        """
        Perform generate-and-test
        """
        features = self.meta.activation_cache
        features = {k: v for k, v in features.items() if k in self.tracked_layer_names}
        features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace)
        self.update_optim_params(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace)


def model_layers(model):
    """
    Note:
        This was moved to netharn.initializers.functional.
        Move it back here, or do some other refactoring.

    Example:
        >>> import torchvision
        >>> model = torchvision.models.AlexNet()
        >>> list(model_layers(model))
    """
    stack = [('', '', model)]
    while stack:
        prefix, basename, item = stack.pop()
        name = '.'.join([p for p in [prefix, basename] if p])
        if isinstance(item, torch.nn.modules.conv._ConvNd):
            yield name, item
        elif isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
            yield name, item
        elif hasattr(item, 'reset_parameters'):
            yield name, item

        child_prefix = name
        for child_basename, child_item in list(item.named_children())[::-1]:
            stack.append((child_prefix, child_basename, child_item))


class MetaNetwork:
    """
    Stores extra information that we need about the network

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/loss-of-plasticity/'))
        >>> from lop.algos.gen_and_test import *  # NOQA
        >>> import torchvision
        >>> net = torchvision.models.resnet18()
        >>> meta = MetaNetwork(net)._build()
        >>> meta.trace(torch.rand(2, 3, 224, 224))
        >>> import networkx as nx
        >>> nx.write_network_text(meta.nx_graph, vertical_chains=1)
        >>> import torchvision
        >>> inputs = torch.rand(2, 3, 224, 224)
        >>> outputs = net(inputs)
        >>> act_shapes = ub.udict(meta.activation_cache).map_values(lambda x: x.shape)
        >>> print('act_shapes = {}'.format(ub.urepr(act_shapes, nl=1)))

    Ignore:
        traced = torch.jit.trace(net, inputs)
    """
    def __init__(meta, net):
        meta.net = net
        meta.activation_cache = {}
        meta.named_layers = None
        meta.id_to_names = None
        meta.name_to_layer = None

    def _build(meta):
        meta._build_layer_information()
        meta._register_layer_hooks()
        return meta

    def _build_layer_information(meta):
        net = meta.net
        named_layers = list(model_layers(net))
        name_to_layer = dict(named_layers)
        id_to_names = defaultdict(list)
        for name, layer in named_layers:
            layer_id = id(layer)
            id_to_names[layer_id].append(name)
        meta.id_to_names = id_to_names
        meta.named_layers = named_layers
        meta.name_to_layer = name_to_layer
        meta.layers = [t[1] for t in meta.named_layers]
        meta.layer_names = [t[0] for t in meta.named_layers]

        meta.type_to_layers = defaultdict(list)
        for _, layer in meta.named_layers:
            meta.type_to_layers[layer.__class__].append(layer)

    def _register_layer_hooks(meta):

        def make_layer_hook(name):
            def record_hidden_activation(layer, input, output):
                activation = output.detach()
                meta.activation_cache[name] = activation
            return record_hidden_activation

        for name, layer in meta.named_layers:
            layer._forward_hooks.clear
            hook = make_layer_hook(name)
            layer.register_forward_hook(hook)

    def trace(meta, input_data=None, input_shape=None):
        """
        Requires an example input
        """
        # from torchview import draw_graph
        import networkx as nx
        import copy

        net = meta.net
        net_copy = copy.deepcopy(net)

        id_to_names = defaultdict(list)
        for name, layer in list(model_layers(net_copy)):
            layer_id = id(layer)
            id_to_names[layer_id].append(name)

        tv_graph = patched_trace_graph(net_copy, input_data)

        def make_label(n_id, data):
            """ Create a nice printable label """
            n_id_str = str(n_id)
            parts = []
            if 'layer_name' in data:
                parts.append(data['layer_name'] + ':')
            if 'compute_node' in data:
                n = data['compute_node']
                parts.append(n.name)
            else:
                parts.append(n_id_str)
            if n_id_str in tv_graph.id_dict:
                idx = tv_graph.id_dict[n_id_str]
                parts.append(f':{idx}')

            if n_id in id_to_names:
                parts.append(' ' + id_to_names[n_id])

            label = ''.join(parts)
            return label

        # Build a networkx version of the torchview model nx_graph
        nx_graph = nx.DiGraph()
        for node in tv_graph.node_set:
            nx_graph.add_node(node)

        for tv_u, tv_v in tv_graph.edge_list:
            tv_u_id = id(tv_u)
            tv_v_id = id(tv_v)
            nx_graph.add_edge(tv_u_id, tv_v_id)
            nx_graph.nodes[tv_u_id]['compute_node'] = tv_u
            nx_graph.nodes[tv_v_id]['compute_node'] = tv_v

        name_to_n_id = {}

        # Enrich each node with more info
        for n_id, data in nx_graph.nodes(data=True):
            if 'compute_node' in data:
                tv_node = data['compute_node']
                if hasattr(tv_node, 'compute_unit_id'):
                    if tv_node.compute_unit_id in id_to_names:
                        layer_names = id_to_names[tv_node.compute_unit_id]
                        if len(layer_names) == 1:
                            layer_name = data['layer_name'] = layer_names[0]
                            name_to_n_id[layer_name] = n_id
                        else:
                            data['layer_names'] = layer_names[0]
            data['label'] = make_label(n_id, data)

        # Not sure what the rando singleton node is.
        if len(nx_graph.nodes) > 1:
            singleton_nodes = []
            for nx_node in nx_graph.nodes:
                if nx_graph.in_degree[nx_node] == 0 and nx_graph.out_degree[nx_node] == 0:
                    singleton_nodes.append(nx_node)
            nx_graph.remove_nodes_from(singleton_nodes)

        meta.nx_graph = nx_graph
        # nx.write_network_text(nx_graph, vertical_chains=1)

        # Determine which nodes have associated layer names
        named_ids = []
        for n_id, data in nx_graph.nodes(data=True):
            if 'layer_name' in data:
                named_ids.append(n_id)

        import ubelt as ub
        topo_order = ub.OrderedSet(nx.topological_sort(nx_graph))
        meta.named_topo_order = (topo_order & named_ids)
        meta.name_to_n_id = name_to_n_id
        meta.tv_graph = tv_graph

        # import netharn as nh
        meta.layer_dependencies = ub.ddict(list)
        for layer_name in meta.layer_names:
            next_names = list([t[0] for t in meta.next_layers(layer_name)])
            meta.layer_dependencies[layer_name].extend(next_names)

        # tv_graph.visual_graph.view()

    def next_layers(meta, layer_name):
        """
        import netharn as nh
        import rich
        for layer_name in meta.layer_names:
            rich.print('[blue] layer_name = {}'.format(ub.urepr(layer_name, nl=1)))
            children = list([t[0] for t in meta.next_layers(layer_name)])
            print('children = {}'.format(ub.urepr(children, nl=1)))
        """
        import networkx as nx
        curr_name = layer_name
        if curr_name in meta.name_to_n_id:
            curr_nx_node = meta.name_to_n_id[curr_name]

            node_data = meta.nx_graph.nodes[curr_nx_node]
            if 'compute_node' in node_data:
                tv_node = node_data['compute_node']
                # curr_in_shape = n.input_shape
                curr_out_shape = tv_node.output_shape

            named_descendants = meta.named_topo_order & nx.descendants(meta.nx_graph, curr_nx_node)
            for next_nx_node in named_descendants:
                next_name = meta.nx_graph.nodes[next_nx_node]['layer_name']
                next_layer = meta.name_to_layer[next_name]
                flag = 0
                if isinstance(next_layer, KNOWN_LAYERS):
                    next_node_data = meta.nx_graph.nodes[next_nx_node]
                    if 'compute_node' in next_node_data:
                        path = nx.shortest_path(meta.nx_graph, curr_nx_node, next_nx_node)
                        next_tv_node = next_node_data['compute_node']
                        next_in_shape = next_tv_node.input_shape
                        between = path[1:-1]
                        between_datas = [meta.nx_graph.nodes[_n] for _n in between]
                        between_named = [d.get('layer_name', None) for d in between_datas]
                        between_layers = [meta.name_to_layer[n] for n in between_named if n is not None]
                        n_between = len([layer for layer in between_layers if isinstance(layer, KNOWN_LAYERS)])
                        # between_flags = sum([b is None for b in between_named])
                        # meta.name_to_layer
                        # next_out_shape = next_tv_node.output_shape
                        if n_between == 0:
                            if (next_in_shape) == curr_out_shape:
                                flag = 1
                    if flag:
                        # print(f'{curr_name} -> {next_name}')
                        # print(between_named)
                        # # named_path_datas = [meta.nx_graph.nodes[_n] for _n in path]
                        # # path_labels = [d['label'] for d in named_path_datas]
                        # # print(path_labels)
                        # print(f'{curr_out_shape} -> {next_in_shape}')
                        yield next_name, next_layer


def patched_trace_graph(net_copy, input_data):
    """
    References:
        https://discuss.pytorch.org/t/tracing-a-graph-of-torch-layers/187615/2
        https://github.com/mert-kurttutan/torchview/issues/104
        https://github.com/mert-kurttutan/torchview/issues/103
        https://github.com/mert-kurttutan/torchview/pull/105
    """
    input_size = None

    if 1:
        ### monkeypatch torchview
        from torchview import torchview as torchview_submod
        if not getattr(torchview_submod, '_geowatch_monkey_patched', 0):
            torchview_submod._geowatch_monkey_patched = 1
            _orig_traverse_data = getattr(
                torchview_submod, '_orig_traverse_data',
                torchview_submod.traverse_data)
            torchview_submod._orig_traverse_data = _orig_traverse_data
            assert torchview_submod.traverse_data.__name__ != 'new_traverse_data'
            def new_traverse_data(data, action_fn, aggregate_fn):
                if isinstance(data, np.ndarray):
                    return data
                return _orig_traverse_data(data, action_fn, aggregate_fn)
            torchview_submod.traverse_data = new_traverse_data
    if 1:
        depth = 9001
        dtypes = None
        graph_name = 'model'
        filename = f'{graph_name}.gv'
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # if torch.cuda.is_available() else "cpu")
        model_mode = 'eval'
        graph_dir = 'TB'
        from torchview.torchview import validate_user_params
        from torchview.torchview import ComputationGraph
        from torchview.torchview import process_input
        from torchview.torchview import module_forward_wrapper
        from torchview.torchview import _orig_module_forward
        from torchview.torchview import Recorder
        import graphviz
        from typing import Mapping
        model = net_copy
        validate_user_params(
            model, input_data, input_size, depth, device, dtypes,
        )
        graph_attr = {
            'ordering': 'in',
            'rankdir': graph_dir,
        }

        # visual settings from torchviz
        # seems to work visually well
        node_attr = {
            'style': 'filled',
            'shape': 'plaintext',
            'align': 'left',
            'fontsize': '10',
            'ranksep': '0.1',
            'height': '0.2',
            'fontname': 'Linux libertine',
            'margin': '0',
        }

        edge_attr = {
            'fontsize': '10',
        }
        visual_graph = graphviz.Digraph(
            name=graph_name, engine='dot', strict=True,
            graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr,
            directory='.', filename=filename
        )

        input_recorder_tensor, kwargs_record_tensor, input_nodes = process_input(
            input_data, input_size, {}, device, dtypes
        )

        tv_graph = ComputationGraph(
            visual_graph, input_nodes, show_shapes=1, expand_nested=1,
            hide_inner_tensors=1, hide_module_functions=1, roll=0,
            depth=depth
        )

        # forward_prop(
        #     model, input_recorder_tensor, device, tv_graph,
        #     model_mode, **kwargs_record_tensor
        # )
        saved_model_mode = model.training
        try:
            mode = model_mode
            if mode == 'train':
                model.train()
            elif mode == 'eval':
                model.eval()
            else:
                raise RuntimeError(
                    f"Specified model mode not recognized: {mode}"
                )
            x = input_recorder_tensor
            new_module_forward = module_forward_wrapper(tv_graph)
            recorder = Recorder(_orig_module_forward, new_module_forward, tv_graph)
            with recorder:
                with torch.no_grad():
                    try:
                        _ = model.to(device)(x)
                    except Exception:
                        kwargs = {}
                        if isinstance(x, (list, tuple)):
                            _ = model.to(device)(*x, **kwargs)
                        elif isinstance(x, Mapping):
                            _ = model.to(device)(**x, **kwargs)
        except Exception as e:
            raise RuntimeError(
                "Failed to run torchgraph see error message"
            ) from e
        finally:
            model.train(saved_model_mode)

        tv_graph.fill_visual_graph()
        # tv_graph.visual_graph.render(format='png')
        # tv_graph.visual_graph.view()
    else:
        from torchview import draw_graph
        tv_graph = draw_graph(
            net_copy,
            input_data=input_data,
            input_size=input_size,
            expand_nested=True,
            hide_inner_tensors=True,
            device='meta', depth=np.inf)
        # tv_graph.visual_graph.view()
    return tv_graph
