# Copyright (c) 2021 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

from tmu.tmulib import ffi, lib
from tmu.tools import CFFISerializable
import numpy as np


class WeightBank(CFFISerializable):

    def __init__(self, weights: np.ndarray, copy: bool = True):
        self.number_of_clauses = weights.shape[0]
        self.weights = weights.copy(order="C") if copy else weights
        
        
        self._cffi_init()

    def _cffi_init(self):
        self.cw_p = ffi.cast("int *", self.weights.ctypes.data)

    def increment(self, clause_output, update_p, clause_active, positive_weights):
        co_p = ffi.cast("unsigned int *", clause_output.ctypes.data)
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.wb_increment(self.cw_p, self.number_of_clauses, co_p, update_p, ca_p, int(positive_weights))

    def decrement(self, clause_output, update_p, clause_active, negative_weights):
        co_p = ffi.cast("unsigned int *", clause_output.ctypes.data)
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.wb_decrement(self.cw_p, self.number_of_clauses, co_p, update_p, ca_p, int(negative_weights))

    def get_weights(self):
        return self.weights
