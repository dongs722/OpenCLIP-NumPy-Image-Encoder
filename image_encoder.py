"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        B, N, D = x.shape
        # Reshape to separate the feature dimension into 12 heads (12 * 64 = 768)
        x = x.reshape(B, N, self.heads, self.head_dim)
        # Transpose to put the heads dimension forward: [B, N, 12, 64] -> [B, 12, N, 64]
        return x.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # Scale the dot product scores by the square root of head_dim 
        # to prevent gradient vanishing and ensure float32 precision
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(self.head_dim))

        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)

        # Multiply attention weights with values: [B, 12, N, N] @ [B, 12, N, 64] -> [B, 12, N, 64]
        context = coefficients @ v
        # Merge the 12 heads back together: [B, 12, N, 64] -> [B, N, 12, 64] -> [B, N, 768]
        context = context.transpose(0, 2, 1, 3)
        merged = context.reshape(x.shape[0], x.shape[1], self.width)

        # Apply the final linear projection and bias, ensuring float32 output
        return (merged @ self.out_weight.T + self.out_bias).astype(np.float32)


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        update = self.attention(self.norm1(x), causal=causal)

        # First residual connection: add the attention update to the original input
        residual = x + update

        u = self.norm2(residual)

        # Expand to 4D (3072) via fc1, apply GELU activation, then project back to D (768) via fc2
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        
        # Second residual connection, ensuring float32 precision
        return (residual + mlp_update).astype(np.float32)


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj'].astype(np.float32)  # [768,512]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        B = images.shape[0]
        # Reshape to separate the image into a 7x7 grid of 32x32 patches: [B, 3, 7, 32, 7, 32]
        x = images.reshape(B, 3, 7, 32, 7, 32)
        # Reorder axes to group patches together: [B, 7, 7, 3, 32, 32]
        x = x.transpose(0, 2, 4, 1, 3, 5)
        # Flatten the patch and channel dimensions to get [B, 49, 3072]
        return x.reshape(B, 49, 3072)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # Project patches to 768 dimensions (no bias) via matrix multiplication
        patches_projected = (patches @ self.patch_weight.T).astype(np.float32)

        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # Prepend the CLS token to the sequence of patches along the token axis
        x = np.concatenate([cls, patches_projected], axis=1)
        
        # Add positional embeddings using broadcasting: [B, 50, 768] + [50, 768]
        return (x + self.position).astype(np.float32)

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Extract the CLS token (index 0) from the sequence: [B, 50, 768] -> [B, 768]
        cls_token = tokens[:, 0, :]
        
        normalized = self.post_norm(cls_token)
        
        # Project to the final 512 dimensions: [B, 768] @ [768, 512] -> [B, 512]
        output = normalized @ self.projection

        # Normalize to unit length (L2 norm) and ensure float32 precision
        return unit_vector(output).astype(np.float32)