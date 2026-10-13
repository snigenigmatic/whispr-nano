import torch
from publication_study.candidate_scores import summed_target_logprobs,choose


def test_forced_prefix_and_padding_do_not_affect_likelihood():
    # Two forced prefix tokens, then text/EOT (EOT=0). Row 2 is empty text.
    tokens=torch.tensor([[3,4,2,0],[3,4,0,0]])
    lengths=torch.tensor([4,3]);logits=torch.zeros(2,3,5)
    scores=summed_target_logprobs(logits,tokens,lengths,2)
    assert torch.allclose(scores,-torch.log(torch.tensor(5.))*torch.tensor([2.,1.]))
    logits[:,0,:]=torch.randn(2,5)*20
    logits[1,2,:]=torch.randn(5)*20
    assert torch.allclose(scores,summed_target_logprobs(logits,tokens,lengths,2))
    logits[0,2,0]=10 # EOT likelihood must count.
    assert summed_target_logprobs(logits,tokens,lengths,2)[0]>scores[0]


def test_raw_and_prior_corrected_decisions_are_distinct():
    assert choose([-1.,-2.],[-.5,-4.],[1,2])==dict(sum_ll=0,mean_ll=0,sum_prior_corrected=1,mean_prior_corrected=1)
