# STL source adaptation

`src/decomposition.rs`: `Decomposition::stl` and the private `stl_*` helpers
adapt the pipeline, local-linear regular-grid smoothing, seasonal extension,
moving averages and robustness weights from statsmodels 0.14.6:

https://github.com/statsmodels/statsmodels/blob/v0.14.6/statsmodels/tsa/stl/_stl.pyx

The source header states:

    (c) 2019 Kevin Sheppard
    License: NCSA/BSD-3 Clause
    Based on NETLIB STL code

The Rust adaptation restricts degree and jump to one, keeps Greeners' public
window normalisation and result type, and distinguishes arithmetic errors from
zero support. It is not an independently invented STL implementation.
Classical decomposition and MSTL are not part of this adaptation.

Source SHA-256:
`9b24f6fc0aff80bae0b93712853cc1499b99c5c0722be84c387ad0c3b4088c15`.

The following is the complete applicable statsmodels licence text from
https://github.com/statsmodels/statsmodels/blob/v0.14.6/LICENSE.txt
(SHA-256 `1ca78e1dec9dcebc55f3b96a862317f23e76422c5fc943568d955aad1f6b5fad`).
These third-party conditions remain applicable; this notice does not resolve
the repository's differing package and contribution licensing statements.

```text
Copyright (C) 2006, Jonathan E. Taylor
All rights reserved.

Copyright (c) 2006-2008 Scipy Developers.
All rights reserved.

Copyright (c) 2009-2018 statsmodels Developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of statsmodels nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
```
