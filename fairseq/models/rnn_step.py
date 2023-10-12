import torch
import torch.nn as nn
import torch.nn.functional as F


# Peephole LSTM
class LSTMStepCell(nn.RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMStepCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(self, input, hx=None):
        gates = F.linear(hx, self.weight_hh, self.bias_hh).add_(F.linear(input, self.weight_ih, self.bias_ih))
        chunked_gates = gates.chunk(4, -1)
        ingate = chunked_gates[0].sigmoid()
        forgetgate = chunked_gates[1].sigmoid()
        cellgate = chunked_gates[2].tanh()
        outgate = chunked_gates[3].sigmoid()
        cy = (forgetgate * hx).add_(ingate * cellgate)
        hy = outgate * cy.tanh()
        return hy


class StackedLSTMStep(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTMStep, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(LSTMStepCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hx=None):
        for i, layer in enumerate(self.layers):
            hx = layer(input, hx)
            hx = F.dropout(hx, p=self.dropout, training=self.training)
        return hx


class GRUStepCell(nn.RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUStepCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(self, input, hx=None):
        igates = F.linear(input, self.weight_ih, self.bias_ih)
        hgates = F.linear(hx, self.weight_hh, self.bias_hh)

        chunked_igates = igates.chunk(3, -1)
        chunked_hgates = hgates.chunk(3, -1)

        reset_gate = chunked_hgates[0].add(chunked_igates[0]).sigmoid()
        input_gate = chunked_hgates[1].add(chunked_igates[1]).sigmoid()
        new_gate = chunked_igates[2].add(chunked_hgates[2].mul(reset_gate)).tanh()
        return (hx - new_gate).mul(input_gate).add(new_gate), reset_gate, input_gate#, chunked_igates, chunked_hgates


class StackedGRUStep(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRUStep, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(GRUStepCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hx=None):
        for i, layer in enumerate(self.layers):
            hx, reset_gate, input_gate = layer(input, hx)
            hx = F.dropout(hx, p=self.dropout, training=self.training)
        return hx, reset_gate, input_gate
