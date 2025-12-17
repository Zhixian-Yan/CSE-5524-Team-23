"""
LoRA (Low-Rank Adaptation) implementation for BioCLIP 2
Based on: https://arxiv.org/abs/2106.09685
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps a linear layer with low-rank adaptation.
    """
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        scale: float = None,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = scale if scale is not None else alpha / rank
        
        # Get dimensions from base layer
        if isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type: {type(base_layer)}")
        
        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False
        
        # LoRA parameters
        # Get device from base layer (may be CPU initially, will be moved with model)
        try:
            device = next(base_layer.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(self, x):
        # Base layer output
        base_output = self.base_layer(x)
        
        # LoRA output - ensure dtype matches input
        x_dropout = self.lora_dropout(x)
        input_dtype = x.dtype
        
        # Convert LoRA parameters to match input dtype
        lora_A = self.lora_A.to(dtype=input_dtype)
        lora_B = self.lora_B.to(dtype=input_dtype)
        
        lora_output = (x_dropout @ lora_A.t() @ lora_B.t()) * self.scale
        
        return base_output + lora_output


class LoRAMultiheadAttention(nn.Module):
    """
    LoRA wrapper for nn.MultiheadAttention module.
    """
    def __init__(
        self,
        attention_module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = attention_module
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # Get dimensions from MultiheadAttention
        embed_dim = attention_module.embed_dim
        
        # Freeze original parameters
        for param in attention_module.parameters():
            param.requires_grad = False
        
        # LoRA parameters for Q, K, V projections
        # Get device from attention module (may be CPU initially, will be moved with model)
        try:
            device = next(attention_module.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        self.lora_A_q = nn.Parameter(torch.randn(rank, embed_dim, device=device) * 0.02)
        self.lora_B_q = nn.Parameter(torch.zeros(embed_dim, rank, device=device))
        self.lora_A_k = nn.Parameter(torch.randn(rank, embed_dim, device=device) * 0.02)
        self.lora_B_k = nn.Parameter(torch.zeros(embed_dim, rank, device=device))
        self.lora_A_v = nn.Parameter(torch.randn(rank, embed_dim, device=device) * 0.02)
        self.lora_B_v = nn.Parameter(torch.zeros(embed_dim, rank, device=device))
        
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, query, key, value, need_weights=False, attn_mask=None, **kwargs):
        """
        Forward pass compatible with nn.MultiheadAttention interface.
        Returns: (output, attn_weights) if need_weights else (output, None)
        """
        # Get original projections using in_proj_weight
        # MultiheadAttention stores Q, K, V projections in in_proj_weight (3*embed_dim, embed_dim)
        embed_dim = self.attention.embed_dim
        
        # Get dtype from input to match precision
        input_dtype = query.dtype
        
        # Project query, key, value using original weights
        if self.attention.batch_first:
            # Convert to (seq_len, batch, embed_dim) for in_proj
            q_seq = query.transpose(0, 1)
            k_seq = key.transpose(0, 1)
            v_seq = value.transpose(0, 1)
        else:
            q_seq = query
            k_seq = key
            v_seq = value
        
        # Original projections
        q_orig = F.linear(q_seq, self.attention.in_proj_weight[:embed_dim], self.attention.in_proj_bias[:embed_dim] if self.attention.in_proj_bias is not None else None)
        k_orig = F.linear(k_seq, self.attention.in_proj_weight[embed_dim:2*embed_dim], self.attention.in_proj_bias[embed_dim:2*embed_dim] if self.attention.in_proj_bias is not None else None)
        v_orig = F.linear(v_seq, self.attention.in_proj_weight[2*embed_dim:], self.attention.in_proj_bias[2*embed_dim:] if self.attention.in_proj_bias is not None else None)
        
        # LoRA adaptation - ensure dtype matches input
        q_dropout = self.lora_dropout(q_seq)
        k_dropout = self.lora_dropout(k_seq)
        v_dropout = self.lora_dropout(v_seq)
        
        # Convert LoRA parameters to match input dtype
        lora_A_q = self.lora_A_q.to(dtype=input_dtype)
        lora_B_q = self.lora_B_q.to(dtype=input_dtype)
        lora_A_k = self.lora_A_k.to(dtype=input_dtype)
        lora_B_k = self.lora_B_k.to(dtype=input_dtype)
        lora_A_v = self.lora_A_v.to(dtype=input_dtype)
        lora_B_v = self.lora_B_v.to(dtype=input_dtype)
        
        q_lora = (q_dropout @ lora_A_q.t() @ lora_B_q.t()) * self.scale
        k_lora = (k_dropout @ lora_A_k.t() @ lora_B_k.t()) * self.scale
        v_lora = (v_dropout @ lora_A_v.t() @ lora_B_v.t()) * self.scale
        
        q = q_orig + q_lora
        k = k_orig + k_lora
        v = v_orig + v_lora
        
        # Reshape for multi-head attention
        num_heads = self.attention.num_heads
        head_dim = embed_dim // num_heads
        seq_len, batch_size = q.size(0), q.size(1)
        
        q = q.view(seq_len, batch_size * num_heads, head_dim).transpose(0, 1)
        k = k.view(seq_len, batch_size * num_heads, head_dim).transpose(0, 1)
        v = v.view(seq_len, batch_size * num_heads, head_dim).transpose(0, 1)
        
        # Compute attention
        scale = head_dim ** -0.5
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.masked_fill(~attn_mask, float('-inf'))
            # attn_mask shape: (L, L) or (N*num_heads, L, L)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            attn = attn + attn_mask
        
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.attention.dropout, training=self.training)
        
        out = torch.bmm(attn, v)
        
        # Reshape back: (batch*num_heads, seq_len, head_dim) -> (seq_len, batch, embed_dim)
        out = out.transpose(0, 1).contiguous().view(seq_len, batch_size, embed_dim)
        
        # Apply output projection
        out = self.attention.out_proj(out)
        
        # Convert back to batch_first if needed
        if self.attention.batch_first:
            out = out.transpose(0, 1)
            if need_weights:
                attn = attn.view(batch_size, num_heads, seq_len, seq_len)
        
        if need_weights:
            return out, attn
        else:
            return out, None


class LoRAAttention(nn.Module):
    """
    LoRA wrapper for Attention module's QKV projection.
    """
    def __init__(
        self,
        attention_module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = attention_module
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # Get dimensions from in_proj_weight
        dim = attention_module.in_proj_weight.shape[1]
        dim_out = attention_module.in_proj_weight.shape[0] // 3  # Q, K, V each have dim_out
        
        # Freeze original parameters
        attention_module.in_proj_weight.requires_grad = False
        if attention_module.in_proj_bias is not None:
            attention_module.in_proj_bias.requires_grad = False
        
        # LoRA parameters for Q, K, V
        # Get device from attention module (may be CPU initially, will be moved with model)
        try:
            device = next(attention_module.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        self.lora_A_q = nn.Parameter(torch.randn(rank, dim, device=device) * 0.02)
        self.lora_B_q = nn.Parameter(torch.zeros(dim_out, rank, device=device))
        self.lora_A_k = nn.Parameter(torch.randn(rank, dim, device=device) * 0.02)
        self.lora_B_k = nn.Parameter(torch.zeros(dim_out, rank, device=device))
        self.lora_A_v = nn.Parameter(torch.randn(rank, dim, device=device) * 0.02)
        self.lora_B_v = nn.Parameter(torch.zeros(dim_out, rank, device=device))
        
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x, attn_mask=None):
        # Match original attention forward logic exactly
        if self.attention.batch_first:
            x = x.transpose(0, 1)
        
        L, N, C = x.shape
        
        # Original QKV projection
        qkv_orig = F.linear(x, self.attention.in_proj_weight, self.attention.in_proj_bias)
        q_orig, k_orig, v_orig = qkv_orig.chunk(3, dim=-1)
        
        # LoRA adaptation - ensure dtype matches input
        x_dropout = self.lora_dropout(x)
        input_dtype = x.dtype
        
        # Convert LoRA parameters to match input dtype
        lora_A_q = self.lora_A_q.to(dtype=input_dtype)
        lora_B_q = self.lora_B_q.to(dtype=input_dtype)
        lora_A_k = self.lora_A_k.to(dtype=input_dtype)
        lora_B_k = self.lora_B_k.to(dtype=input_dtype)
        lora_A_v = self.lora_A_v.to(dtype=input_dtype)
        lora_B_v = self.lora_B_v.to(dtype=input_dtype)
        
        q_lora = (x_dropout @ lora_A_q.t() @ lora_B_q.t()) * self.scale
        k_lora = (x_dropout @ lora_A_k.t() @ lora_B_k.t()) * self.scale
        v_lora = (x_dropout @ lora_A_v.t() @ lora_B_v.t()) * self.scale
        
        q = q_orig + q_lora
        k = k_orig + k_lora
        v = v_orig + v_lora
        
        # Reshape for attention
        q = q.reshape(L, N * self.attention.num_heads, -1).transpose(0, 1)
        k = k.reshape(L, N * self.attention.num_heads, -1).transpose(0, 1)
        v = v.reshape(L, N * self.attention.num_heads, -1).transpose(0, 1)
        
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        
        if self.attention.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.attention.logit_scale, max=self.attention.logit_scale_max).exp()
            attn = attn.view(N, self.attention.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attention.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            if self.attention.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attention.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.attention.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attention.attn_drop(attn)
                x = torch.bmm(attn, v)
        
        if self.attention.head_scale is not None:
            x = x.view(N, self.attention.num_heads, L, C) * self.attention.head_scale
            x = x.view(-1, L, C)
        
        x = x.transpose(0, 1).reshape(L, N, C)
        
        if self.attention.batch_first:
            x = x.transpose(0, 1)
        
        x = self.attention.out_proj(x)
        x = self.attention.out_drop(x)
        return x


def apply_lora_to_model(
    model,
    target_modules=None,
    rank=8,
    alpha=16.0,
    dropout=0.0,
    enable_vision=True,
    enable_text=True,
):
    """
    Apply LoRA to a CLIP model.
    
    Args:
        model: CLIP model instance
        target_modules: List of module names to apply LoRA to. 
                       If None, applies to attention layers by default.
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: LoRA dropout rate
        enable_vision: Whether to apply LoRA to vision encoder
        enable_text: Whether to apply LoRA to text encoder
    
    Returns:
        Modified model with LoRA layers
    """
    if target_modules is None:
        target_modules = ['attn', 'attention']
    
    lora_modules = {}
    
    def apply_lora_recursive(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            # Check if this is a target module
            should_apply_lora = any(target in name.lower() for target in target_modules)
            
            if should_apply_lora and isinstance(child, nn.Module):
                # Handle nn.MultiheadAttention
                if isinstance(child, nn.MultiheadAttention):
                    lora_attn = LoRAMultiheadAttention(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, name, lora_attn)
                    lora_modules[full_name] = lora_attn
                # Handle Attention module (custom Attention class)
                elif hasattr(child, 'in_proj_weight') and hasattr(child, 'num_heads'):
                    lora_attn = LoRAAttention(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, name, lora_attn)
                    lora_modules[full_name] = lora_attn
                # Handle Linear layers
                elif isinstance(child, nn.Linear):
                    lora_linear = LoRALayer(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, name, lora_linear)
                    lora_modules[full_name] = lora_linear
                else:
                    # Recursively apply to children
                    apply_lora_recursive(child, full_name)
            else:
                # Recursively apply to children
                apply_lora_recursive(child, full_name)
    
    # Apply to vision encoder
    if enable_vision and hasattr(model, 'visual'):
        apply_lora_recursive(model.visual, "visual")
    
    # Apply to text encoder
    if enable_text and hasattr(model, 'transformer'):
        apply_lora_recursive(model.transformer, "transformer")
    
    return model, lora_modules


def get_lora_parameters(model):
    """
    Get all LoRA parameters from the model.
    """
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, (LoRALayer, LoRAAttention, LoRAMultiheadAttention)):
            lora_params.extend([
                module.lora_A if hasattr(module, 'lora_A') else None,
                module.lora_B if hasattr(module, 'lora_B') else None,
            ])
            # For LoRAAttention and LoRAMultiheadAttention, also get Q, K, V parameters
            if isinstance(module, (LoRAAttention, LoRAMultiheadAttention)):
                lora_params.extend([
                    module.lora_A_q, module.lora_B_q,
                    module.lora_A_k, module.lora_B_k,
                    module.lora_A_v, module.lora_B_v,
                ])
    # Filter out None values
    lora_params = [p for p in lora_params if p is not None]
    return lora_params


def save_lora_weights(model, path):
    """
    Save only LoRA weights to a file.
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora_B
        elif isinstance(module, (LoRAAttention, LoRAMultiheadAttention)):
            lora_state_dict[f"{name}.lora_A_q"] = module.lora_A_q
            lora_state_dict[f"{name}.lora_B_q"] = module.lora_B_q
            lora_state_dict[f"{name}.lora_A_k"] = module.lora_A_k
            lora_state_dict[f"{name}.lora_B_k"] = module.lora_B_k
            lora_state_dict[f"{name}.lora_A_v"] = module.lora_A_v
            lora_state_dict[f"{name}.lora_B_v"] = module.lora_B_v
    
    torch.save(lora_state_dict, path)
    print(f"Saved LoRA weights to {path}")


def load_lora_weights(model, path):
    """
    Load LoRA weights into a model.
    """
    lora_state_dict = torch.load(path, map_location='cpu')
    model_state_dict = model.state_dict()
    
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model")
    
    model.load_state_dict(model_state_dict)
    print(f"Loaded LoRA weights from {path}")

