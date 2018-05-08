import torch
def metric_loss(fc, batch_person):
    person_center = torch.unsqueeze(torch.mean(input=fc, dim=1), 1)
    center_diff = torch.norm(fc - person_center, dim=-1)
    center_loss = torch.mean(center_diff)
    cross_center = person_center.view(1, -1, 128)
    cross_center = cross_center.expand(batch_person, batch_person, 128)
    cross_diff = torch.norm(person_center - cross_center, dim=-1)
    values, idx = torch.topk(-cross_diff.view(-1, batch_person*batch_person), batch_person*2)
    cross_loss = torch.sum(-values) / batch_person
    return center_loss/cross_loss
