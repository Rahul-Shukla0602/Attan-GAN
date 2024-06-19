"""
Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.
"""

import torch
import torch.nn as nn

# utility function for creating a 1x1 convolutional layer in a neural network
# The function returns a 1x1 convolutional layer using nn.Conv2d from the PyTorch library. It applies the convolution operation on 2D images (hence Conv2d) with a kernel size of 1x1. It's worth noting that a 1x1 convolution is often used for dimensionality reduction or channel-wise feature transformation in convolutional neural networks (CNNs).
# in_planes: Number of input channels to the convolutional layer.
# out_planes: Number of output channels from the convolutional layer.
# kernel_size=1: Specifies the size of the convolutional kernel.
# stride=1: The stride of the convolution. Here, it's set to 1, meaning the convolution operation moves one pixel at a time.
# padding=0: Padding added to the input image before applying the convolution. Here, it's set to 0, meaning no padding is added.
# bias=False: Indicates whether to include bias parameters in the convolution. Here, it's set to False, meaning no bias terms are included.
# Purpose: This function is likely used to create 1x1 convolutional layers within a neural network architecture. T
def conv1x1(in_planes, out_planes):
    '''
    name: 
    test: test font
    msg: 
    param {*}
    return {*}
    '''    
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)

#  implements an attention mechanism between a query and a context, where the context consists of image features. 
# Implementation:
# The function first extracts the dimensions of the input tensors.
# It reshapes the context tensor to make it compatible for the attention operation.
# Attention scores are computed using matrix multiplication between the transposed context and the query tensor.
# Softmax activation is applied to normalize the attention scores.
# The attention scores are scaled by the hyperparameter gamma1.
# Further normalization with softmax is performed.
# Weighted context is calculated using matrix multiplication between the context and the transposed attention scores.
# Finally, the weighted context and reshaped attention scores are returned.
# func_attention facilitates attention mechanism computation between textual queries and image 
# contexts, producing weighted context representations and attention scores. This is crucial for models aiming 
# to generate descriptive text based on visual inputs.
# Description: Describes the operation performed by the function, which involves calculating attention weights between a query and a context.
# Parameters:
# query: A tensor representing word embeddings with shape (batch_size, ndf, queryL).
# context: A tensor representing image features with shape (batch_size, ndf, ih, iw).
# gamma1: A scalar hyperparameter used for scaling the attention scores.
# Returns:
# weightedContext: A tensor representing the context weighted by the attention scores, with shape (batch_size, ndf, queryL).
# attn: A tensor representing the attention scores reshaped to match the image dimensions, with shape (batch_size, queryL, ih, iw).
def func_attention(query, context, gamma1): 
    '''
    description: : context * contextT * query 
            (256,17*17) * (17*17,256) * (256,cap_len) => (256,cap_len)
    param {*} query(word): tensor(14,256,12)
    param {*} context(img_feat): tensor(14,256,17,17)
    param {*} gamma1 4.0,超参,用于放缩attn
    return {*} weightedContext: tensor(14, 256, 12),
    return {*} attn: tensor(14,12,17,17), 每个单词对每个像素点的attn值
    '''
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    '''
        
        query(word): tensor(14,256,12)
        context(img_feat): tensor(14,256,17,17),单词和文本
    '''
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL) # ! tensor(14,256,17*17=289)
    contextT = torch.transpose(context, 1, 2).contiguous() # ! tensor(14,289,256)

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper # ! tensor(14,289,12)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL) # ! tensor(14*289,12)
    attn = nn.Softmax(dim=1)(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL) # ! tensor(14,289,12)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous() # ! tensor(14,12,289)
    attn = attn.view(batch_size*queryL, sourceL) # ! tensor(14*12,289)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL) # ! tensor(14,12,289)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous() # ! tensor(14,289,12)

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT) # ! tensor(14,256,12)

    return weightedContext, attn.view(batch_size, -1, ih, iw) # ! tensor(14, 256,12), tensor(14,12,17,17)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn

# Constructor (__init__):
# Parameters:
# idf: Dimensionality of the input feature vectors (query).
# cdf: Dimensionality of the context feature vectors.
# Functionality:
# Initializes the module by setting up a 1x1 convolutional layer (conv_context) to transform the context features to match the input feature dimensions.
# Initializes a softmax layer (sm) along the specified dimension (dimension 1).
# Initializes the mask attribute to None.
# applyMask Method:
# Parameters:
# mask: Mask tensor to apply to the attention scores.
# Functionality:
# Sets the mask attribute to the provided mask tensor. This method is used to apply a mask to the attention scores during the forward pass.
# forward Method:
# Parameters:
# input: Input tensor representing query features with shape (batch_size, idf, ih, iw).
# context: Context tensor representing features from the source with shape (batch_size, cdf, sourceL).
# Functionality:
# Extracts the dimensions of the input tensors.
# Reshapes the input tensor to make it compatible for the attention operation.
# Applies a 1x1 convolution to the context tensor to transform it to match the input feature dimensions.
# Computes attention scores between the reshaped input and transformed context tensors.
# Applies masking to the attention scores if a mask is provided.
# Applies softmax activation to normalize the attention scores.
# Reshapes the attention scores to their original dimensions.
# Computes the weighted context by performing a weighted sum of context features based on the attention scores.
# Reshapes the weighted context and attention scores to their original dimensions and returns them.
# Purpose:
# This class serves as a module for implementing a global attention mechanism between query and context features.
# It enables the model to focus on relevant parts of the context during computations, aiding in tasks such as image captioning or language translation where alignment between input and output sequences is crucial.
# In summary, GlobalAttentionGeneral encapsulates the functionality required for computing attention scores between query and context features,


# A global attention mechanism works by selectively focusing on different parts of a source sequence (context) when generating or processing a target sequence (query). Here's how it typically works:

# Input Sequences:
# Source Sequence (Context): This is a sequence of feature vectors representing the input data. For example, in machine translation, it could be the sequence of word embeddings or image features.
# Target Sequence (Query): This is the sequence being generated or processed based on the context. For example, in machine translation, it could be the sequence of target language words.
# Alignment Scores:
# The first step is to compute alignment scores between each element in the target sequence and each element in the source sequence. These alignment scores represent the relevance or importance of each source element with respect to each target element.
# Alignment scores are often calculated using a similarity function between the target and source elements. Common similarity functions include dot product, scaled dot product, or concatenation followed by a linear layer.
# Attention Weights:
# Alignment scores are converted into attention weights using a softmax function. These attention weights represent the importance of each source element relative to the target element. Higher weights indicate greater importance.
# The softmax function ensures that the attention weights sum up to 1, making them interpretable as probabilities.
# Context Vector:
# Once the attention weights are obtained, a context vector is computed as the weighted sum of the source elements, where each element is weighted by its corresponding attention weight.
# The context vector captures the relevant information from the source sequence that is most useful for processing or generating the target element.
# Output:
# The context vector is then used in conjunction with the target element (or its embedding) to produce the output. This could involve passing the context vector through a neural network layer, such as an LSTM or a feedforward network, to generate the next element in the target sequence.
# This process is repeated for each element in the target sequence, with the attention mechanism dynamically adjusting the focus of the context based on the current target element being processed.
# Optional: Masking:
# In some cases, it might be necessary to mask certain elements in the source sequence to prevent the model from attending to them. For example, in machine translation, padding tokens in the source sequence are often masked to avoid considering them during attention calculation.