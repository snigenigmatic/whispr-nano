"""Teacher-forced candidate scores; forced prefixes and padding never count."""
import torch


def summed_target_logprobs(logits, inputs, lengths, prefix_length):
    # logits[:, t] predicts inputs[:, t+1]. The last target is EOT, supplied
    # in inputs; no prediction after EOT and no forced language token is scored.
    assert prefix_length>0 and logits.shape[:2]==inputs[:, :-1].shape
    target=inputs[:, 1:]
    selected=torch.log_softmax(logits.float(),dim=-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
    pos=torch.arange(target.shape[1],device=target.device).unsqueeze(0)
    mask=(pos>=prefix_length-1)&(pos<(lengths-1).unsqueeze(1))
    assert torch.all(mask.sum(1)==lengths-prefix_length)
    return selected.masked_fill(~mask,0).sum(1)


def choose(scores, null_scores, token_counts):
    """All four fixed comparisons; first candidate breaks exact ties."""
    import numpy as np
    s=np.asarray(scores);z=np.asarray(null_scores);n=np.asarray(token_counts)
    assert s.shape==z.shape==n.shape and np.isfinite(s).all() and np.isfinite(z).all() and (n>0).all()
    return {k:int(np.argmax(v)) for k,v in dict(sum_ll=s,mean_ll=s/n,
        sum_prior_corrected=s-z,mean_prior_corrected=(s-z)/n).items()}
