import torch


def test_sequence_encoder_outputs_user_embedding():
    from cognid_genrec.models.sequence_encoder import SequenceEncoder

    model = SequenceEncoder(num_items=100, num_actions=4, hidden_dim=16, max_len=8)
    item_ids = torch.tensor([[1, 2, 3, 0]])
    action_ids = torch.tensor([[2, 1, 3, 0]])
    time_deltas = torch.tensor([[0.0, 1.0, 2.0, 0.0]])
    output = model(item_ids=item_ids, action_ids=action_ids, time_deltas=time_deltas)

    assert output.shape == (1, 16)


def test_info_nce_loss_is_lower_for_matching_pairs():
    from cognid_genrec.models.contrastive import info_nce_loss

    user_embeddings = torch.eye(3)
    positive_item_embeddings = torch.eye(3)
    shuffled_item_embeddings = positive_item_embeddings[[2, 0, 1]]

    good_loss = info_nce_loss(user_embeddings, positive_item_embeddings, temperature=0.2)
    bad_loss = info_nce_loss(user_embeddings, shuffled_item_embeddings, temperature=0.2)

    assert good_loss < bad_loss
