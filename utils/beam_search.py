"""
Beam search decoding utilities
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from config import Config


class BeamSearchNode:
    """Node in beam search tree"""

    def __init__(self, token_id: int, log_prob: float, hidden_state, prev_node=None):
        self.token_id = token_id
        self.log_prob = log_prob
        self.hidden_state = hidden_state
        self.prev_node = prev_node
        self.length = 1 if prev_node is None else prev_node.length + 1

    def get_sequence(self) -> List[int]:
        """Get sequence from root to current node"""
        sequence = []
        node = self
        while node is not None:
            sequence.append(node.token_id)
            node = node.prev_node
        return sequence[::-1]

    def get_avg_log_prob(self) -> float:
        """Get average log probability (normalized by length)"""
        return self.log_prob / self.length


def greedy_decode(model, src, src_len, tgt_vocab, max_length: int, device: str) -> List[int]:
    """
    Greedy decoding

    Args:
        model: Translation model
        src: Source sequence (1, src_len)
        src_len: Source length
        tgt_vocab: Target vocabulary
        max_length: Maximum decoding length
        device: Device

    Returns:
        Decoded sequence (list of token ids)
    """
    model.eval()

    with torch.no_grad():
        # Start with BOS token
        tgt = torch.tensor([[tgt_vocab.bos_idx]], dtype=torch.long, device=device)
        decoded = []

        # Encode source (for transformer) or initialize hidden state (for RNN)
        if hasattr(model, 'encode'):
            # Transformer
            memory = model.encode(src, src_len)
        else:
            # RNN: get encoder outputs and hidden
            encoder_outputs, hidden = model.encoder(src, src_len)

        for _ in range(max_length):
            if hasattr(model, 'decode'):
                # Transformer
                output = model.decode(tgt, memory, src)
                logits = model.generator(output[:, -1:, :])
            else:
                # RNN
                if len(decoded) == 0:
                    decoder_input = tgt
                else:
                    decoder_input = torch.tensor([[decoded[-1]]], dtype=torch.long, device=device)

                output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
                logits = model.fc_out(output)

            # Get most likely token
            token_id = logits.argmax(dim=-1).item()

            # Stop if EOS token
            if token_id == tgt_vocab.eos_idx:
                break

            decoded.append(token_id)

            # Update tgt for transformer
            if hasattr(model, 'decode'):
                tgt = torch.cat([tgt, torch.tensor([[token_id]], dtype=torch.long, device=device)], dim=1)

    return decoded


def beam_search_decode(model, src, src_len, tgt_vocab, beam_size: int, max_length: int, device: str) -> List[int]:
    """
    Beam search decoding

    Args:
        model: Translation model
        src: Source sequence (1, src_len)
        src_len: Source length
        tgt_vocab: Target vocabulary
        beam_size: Beam size
        max_length: Maximum decoding length
        device: Device

    Returns:
        Decoded sequence (list of token ids)
    """
    model.eval()

    with torch.no_grad():
        # Encode source
        if hasattr(model, 'encode'):
            # Transformer
            memory = model.encode(src, src_len)
        else:
            # RNN
            encoder_outputs, hidden = model.encoder(src, src_len)

        # Initialize beam with BOS token
        start_node = BeamSearchNode(
            token_id=tgt_vocab.bos_idx,
            log_prob=0.0,
            hidden_state=hidden if not hasattr(model, 'encode') else None,
            prev_node=None
        )

        beam = [start_node]
        completed = []

        for step in range(max_length):
            candidates = []

            for node in beam:
                # Stop if this is EOS
                if node.token_id == tgt_vocab.eos_idx:
                    completed.append(node)
                    continue

                # Get sequence so far
                sequence = node.get_sequence()
                tgt_input = torch.tensor([sequence], dtype=torch.long, device=device)

                # Get next token probabilities
                if hasattr(model, 'decode'):
                    # Transformer
                    output = model.decode(tgt_input, memory, src)
                    logits = model.generator(output[:, -1:, :])
                else:
                    # RNN
                    decoder_input = torch.tensor([[sequence[-1]]], dtype=torch.long, device=device)
                    output, new_hidden = model.decoder(decoder_input, node.hidden_state, encoder_outputs)
                    logits = model.fc_out(output)

                log_probs = F.log_softmax(logits.squeeze(0).squeeze(0), dim=-1)

                # Get top-k tokens
                top_log_probs, top_indices = log_probs.topk(beam_size)

                # Create new nodes
                for log_prob, token_id in zip(top_log_probs, top_indices):
                    new_node = BeamSearchNode(
                        token_id=token_id.item(),
                        log_prob=node.log_prob + log_prob.item(),
                        hidden_state=new_hidden if not hasattr(model, 'encode') else None,
                        prev_node=node
                    )
                    candidates.append(new_node)

            # Select top beam_size candidates
            if len(candidates) == 0:
                break

            beam = sorted(candidates, key=lambda x: x.get_avg_log_prob(), reverse=True)[:beam_size]

            # Stop if all beams have generated EOS
            if len(beam) == 0:
                break

        # Add remaining beams to completed
        completed.extend(beam)

        # Select best sequence
        if len(completed) == 0:
            return []

        best_node = max(completed, key=lambda x: x.get_avg_log_prob())
        sequence = best_node.get_sequence()[1:]  # Remove BOS token

        # Remove EOS token if present
        if len(sequence) > 0 and sequence[-1] == tgt_vocab.eos_idx:
            sequence = sequence[:-1]

        return sequence


if __name__ == "__main__":
    print("Beam search utilities implemented")
    print("Use with model.decode() for Transformer or model.decoder() for RNN")
