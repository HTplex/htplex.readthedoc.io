Paper Reading Notes, Fall 2023
===============================

llama 2
--------
Comparing to llama 1, there's a lot less information about model details and training data, which makes it intentionally hard to reproduce the results. 
But we can just refer those details from llama 1 as lots of the parts should be the same.
In this note, we will mainly focus on:

1. Modern tricks used by the model and pre triaining
2. Analysis of evaluation metrics and results used in the paper
3. Methods of Supervised Fine Tuning, RLHF
4. Model behavior analysis 

Modern Tricks
^^^^^^^^^^^^^

Comparing to llama 1, llama 2 mainly have 3 new tricks: 
``RMSNorm pre-normalization`` ``SwiGLU activation function`` ``rotary position embedding`` and ``grouped-query attention (GQA)``
We will discuss them one by one.


``RMSNorm`` `Paper <https://arxiv.org/abs/1910.07467>`_ 

In standard ``Layer Normalization``, the mean and variance are computed across all the features for each data sample in a batch,
and these statistics are used to normalize the data. 
This process helps in addressing the issue of internal covariate shift, where the distribution of each layer's inputs changes during training, 
making the training process more stable and faster.

.. math::

    x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}

``RMSNorm`` simplifies the normalization process used in Layer Normalization. 
Unlike LayerNorm, which normalizes based on both the mean and variance of a layer's inputs, 
RMSNorm only uses the root mean square (RMS) value, essentially the standard deviation without subtracting the mean.

.. math::

    \text{RMS} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}   

    x_i = \frac{x_i}{\sqrt{\text{RMS}^2 + \epsilon}}

Why RMSNorm Might Be Better?

* Simplification: RMSNorm simplifies the computation by eliminating the need to calculate the mean. This can lead to minor computational efficiency improvements.
* Stability in Training: It has been observed in some cases that RMSNorm can provide more stable training for deep networks. This is particularly relevant in architectures like Transformers where Layer Normalization plays a crucial role.
* Effectiveness in Deep Networks: Some studies suggest that RMSNorm can be more effective than LayerNorm in very deep networks, potentially due to its simpler normalization dynamic.
* Robustness: RMSNorm might offer more robustness in scenarios where the mean calculation in LayerNorm could introduce instability.

``pre-normalization``

Pre-normalization:
In pre-normalization (PreNorm), normalization is applied to the input of a sub-layer (such as a self-attention or feed-forward layer) before the actual operation of the sub-layer. 
The typical structure of a PreNorm layer in a Transformer model would be:

1. Normalize the input.
2. Apply the sub-layer operation (like attention or feed-forward).
3. Add the result to the original input (residual connection).
The PreNorm structure can be represented as:

.. math::

    \text{Output} = \text{SubLayer}(\text{Normalize}(X)) + X

Post-normalization:
Post-normalization (PostNorm), on the other hand, applies normalization after the sub-layer operation. This is the approach originally used in the Transformer model by Vaswani et al. The structure is:

1. Apply the sub-layer operation.
2. Add the output to the original input (residual connection).
3. Normalize this result.
The PostNorm structure can be represented as:

.. math::

    \text{Output} = \text{Normalize}(\text{SubLayer}(X) + X)


Comparison:
* Stability in Training: PreNorm is often found to be more stable in training, especially for very deep models. It can lead to faster convergence and is less sensitive to hyperparameter choices.
* Performance: The performance of PreNorm versus PostNorm can depend on the specific task and model architecture. In some cases, PostNorm might yield slightly better results, whereas in others, PreNorm is superior.
* Gradient Flow: PreNorm can help with the flow of gradients through the network, potentially alleviating issues with vanishing or exploding gradients in very deep networks.
* Implementation Ease: PreNorm might be easier to implement, especially in architectures where layers are added or removed dynamically, as it avoids the need to handle normalization at the beginning and end of sequences of layers.

``SwiGLU activation function``
`Paper <https://arxiv.org/abs/2002.05202>`_

Swish-Gated Linear Unit (SwiGLU) is a new activation function that combines the Swish and GLU activation functions.
Just another Highly mathmaticallly complicated activation function which is better, it is very hard for us to understand.

``rotary position embedding (RoPE)``
`Paper <https://arxiv.org/abs/2104.09864>`_

TBD 

``grouped-query attention (GQA)``

TBD


Pre-training details
^^^^^^^^^^^^^^^^^^^^^^^^^
Tokenizer: `bytepair encoding` (BPE) algorithm (Sennrich et al., 2016) using the implementation from `SentencePiece` (Kudo and Richardson, 2018).
Plus split all numbers into **individual** digits and use bytes to **decompose** unknown UTF-8 characters. The total vocabulary size is **32k** tokens.

Optimizer: AdamW with  ``beta1=0.9``, ``beta2=0.95``, ``epsilon=1e-5``

Learning rate schedule: ``warmup_steps=2000``, ``minimum = 1e-6``

Weight decay: ``0.1``, Gradient clipping: ``1.0``

Loss Function:

Loss Progression:

.. image:: ./imgs/llama2_lossline.jpg
    :width: 500px
    :align: center


Pre-Training Time (2T tokens):

+-------------+------------+--------------------+---------+-----------------------+
|             | A100 Hours | 8*A100 Node Months | $ (aws) | token*BParam/sec/A100 |
+=============+============+====================+=========+=======================+
| LLAMA 2 7B  | 184,320    | 32                 | $1M     | 21098                 |
+-------------+------------+--------------------+---------+-----------------------+
| LLAMA 2 13B | 368,640    | 64                 | $2M     | 19591                 |
+-------------+------------+--------------------+---------+-----------------------+
| LLAMA 2 34B | 1,038,336  | 180                | $6M     | 18191                 |
+-------------+------------+--------------------+---------+-----------------------+
| LLAMA 2 70B | 1,720,320  | 298                | $10M    | 22605                 |
+-------------+------------+--------------------+---------+-----------------------+

In-Domain Pre-Training Time (1B tokens):

+-------------+------------+------------------+---------+
|             | A100 Hours | 8*A100 Node Days | $ (aws) |
+=============+============+==================+=========+
| LLAMA 2 7B  | 92         | 0.5              | $350    |
+-------------+------------+------------------+---------+
| LLAMA 2 13B | 184        | 1                | $700    |
+-------------+------------+------------------+---------+
| LLAMA 2 34B | 519        | 3                | $2000   |
+-------------+------------+------------------+---------+
| LLAMA 2 70B | 860        | 5                | $3500   |
+-------------+------------+------------------+---------+


Evaluation Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^

``Perplexity``

``BLEU``

``F1``

``ROUGE``

``BERTScore``

``GPT3Score``

``GPT

llama 1
--------




