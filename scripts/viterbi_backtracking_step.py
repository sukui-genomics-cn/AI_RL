import torch

def viterbi_backtracking_step(prev_states, gamma_state, transition_matrix_transposed,output_type=torch.int64, non_homogeneous_mask=None):
    """
    Computes a Viterbi backtracking step in parallel for all models and batch elements.
    :param prev_states: Previous decoded states. Shape: (num_model, b, 1)
    :param gamma_state: Viterbi values of the previously decoded states. Shape: (num_model, b, q)
    :param transition_matrix_transposed: Transposed logarithmic transition matrices. Shape: (num_models, q, q) or (num_models, b, q, q)
    :param output_type: Datatype of the output states (defalult: torch.int64)
    :param non_homogeneous_mask: Optional mask of shape (num_models, b, q, q) that specifies which transitions are allowed.
    :return:
    """
    # Determine batch dimensions for indexing
    batch_dims = transition_matrix_transposed.dim() - 2
    if non_homogeneous_mask is None:
        # Gather columns (starting state q' when transitioning to q)
        A_prev_states = torch.gather(
            transition_matrix_transposed, dim=-1,
            index=prev_states.expand(-1, -1, transition_matrix_transposed.size(-1))
        )
    else:
        # Combine transition matrix with masked log probabilities
        masked_trans = transition_matrix_transposed + torch.log(torch.transpose(non_homogeneous_mask, -1, -2))
        A_prev_states = torch.gather(masked_trans, dim=-1, index=prev_states.expand(-1, -1, masked_trans.size(-1)))

    # Compute next states using argmax
    next_states = torch.argmax(A_prev_states + gamma_state, dim=-1)

    # Expand dimensions to match input shape
    next_states = next_states.unsqueeze(-1)
    return next_states

if __name__ == '__main__':
    # Dummu data
    num_models, batch_size, q = 2, 3, 4
    prev_states = torch.tensor([[[0]], [[1]]], dtype=torch.int64)  # (2, 3, 1)
    gamma_state = torch.randn(num_models, batch_size, q)  # (2, 3, 4)'
    transition_matrix = torch.randn(num_models, q, q)  # (2, 4, 4)
    transition_matrix_transposed = transition_matrix.transpose(-1, -2)
    non_homogeneous_mask = torch.ones(num_models, batch_size, q, q)  # Example mask

    result = viterbi_backtracking_step(
        prev_states, gamma_state, transition_matrix_transposed,non_homogeneous_mask=None
    )
    print("Next states shape:", result.shape)  # Expected: (2, 3, 1)
    print("Next states:", result)