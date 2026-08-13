# Copyright (c) 2023 Ole-Christoffer Granmo

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

from cffi import FFI
import pathlib

current_dir = pathlib.Path(__file__).parent
source_dir = current_dir.joinpath("src")
header_dir = current_dir.joinpath("include")

HEADERS = [
    "ClauseBank.h", "Tools.h", "WeightBank.h", "ClauseBankSparse.h",
    # fast_rand_seed.h was missing: clause_bank.py calls lib.pcg32_seed()
    # directly (not through the fast_rand() macro), so cffi needs its
    # prototype in cdef() to expose it at all -- without this, the build
    # succeeded but failed at *call* time with "cffi library ... has no
    # function ... named 'pcg32_seed'", even after fixing the SOURCES gap
    # below made the function actually get compiled in.
    "fast_rand_seed.h",
]
# random/pcg32_fast.c and random/xorshift128.c were missing from this list --
# Tools.c's fast_rand() macro expands to pcg32_fast() (see include/fast_rand.h),
# so without pcg32_fast.c the extension built but failed to *load* with
# "undefined symbol: pcg32_fast". xorshift128.c is the other RNG backend
# declared in the same header and included alongside it for the same reason.
SOURCES = [
    "ClauseBank.c", "Tools.c", "WeightBank.c", "ClauseBankSparse.c",
    "random/pcg32_fast.c", "random/xorshift128.c",
]

header_content = '\n'.join([header_dir.joinpath(x).open("r").read() for x in HEADERS])
source_content = '\n'.join([source_dir.joinpath(x).open("r").read() for x in SOURCES])

ffibuilder = FFI()
ffibuilder.cdef(header_content)
ffibuilder.set_source(
    "tmu.tmulib",
    source_content,
    include_dirs=[header_dir.absolute()],
    extra_compile_args=["-O3"]
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
