# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx

def update_dim(model):
    
    # New width and height. Using -1,-1 so that we can use variable input size in model while using triton inference server.
    value = -1 
    inputs = model.graph.input
    outputs = model.graph.output

    inputs[0].type.tensor_type.shape.dim[0].dim_value = -1
    inputs[0].type.tensor_type.shape.dim[2].dim_value = value
    inputs[0].type.tensor_type.shape.dim[3].dim_value = value
    
    for output in outputs:
        output.type.tensor_type.shape.dim[0].dim_value = -1
        output.type.tensor_type.shape.dim[2].dim_value = value # 
        output.type.tensor_type.shape.dim[3].dim_value = value

def change(update_dim, infile, outfile):
    model = onnx.load(infile)
    update_dim(model)
    onnx.save(model, outfile) # Save the new model with updated dimension 

## Update the input and output dimension of model layers ##
change(update_dim, "centerface.onnx", "model.onnx")
