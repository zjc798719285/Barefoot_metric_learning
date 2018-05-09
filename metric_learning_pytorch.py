import torch
def metric_loss(fc, batch_person, num_fc):
    person_center = torch.unsqueeze(torch.mean(input=fc, dim=1), 1)
    center_diff = torch.norm(fc - person_center, dim=-1)
    center_loss = torch.mean(center_diff)
    cross_center = person_center.view(1, -1, num_fc)
    cross_center = cross_center.expand(batch_person, batch_person, num_fc)
    cross_diff = torch.norm(person_center - cross_center, dim=-1)
    values, idx = torch.topk(-cross_diff.view(-1, batch_person*batch_person), batch_person*2)
    cross_loss = torch.sum(-values) / batch_person
    return center_loss, cross_loss, center_loss/cross_loss


def metric_loss2(fc):
    person_center = torch.unsqueeze(torch.mean(input=fc, dim=1), 1)
    center_diff = torch.norm(fc - person_center, dim=-1)
    center_loss = torch.mean(center_diff)
    values, idx = torch.min(center_diff, -1)
    cross_loss = torch.mean(values)
    return center_loss, cross_loss, center_loss/cross_loss

def metric_loss3(fc,batch_person, num_file, fcs):
    person_center = torch.unsqueeze(torch.mean(input=fc, dim=1), 1)
    center_diff = torch.norm(fc - person_center, dim=-1)
    center_loss = torch.mean(center_diff)
    reshape_center = person_center.view(1, batch_person, -1)
    cross_centers = reshape_center.expand(batch_person, batch_person, fcs)
    cross_loss = 0
    for i in range(batch_person):
        cross_i = torch.unsqueeze(cross_centers[:, i, :], 1)
        cross_diss = torch.norm(fc - cross_i, dim=-1)
        if i == 0:
            cross_diss_ = cross_diss[1:batch_person]
        elif i == batch_person - 1:
            cross_diss_ = cross_diss[0:batch_person - 1]
        else:
            cross_diss_ = torch.cat((cross_diss[0: i], cross_diss[i + 1: batch_person]), 0)
        cross_diss_ = cross_diss_.view(-1, (batch_person-1)*num_file)
        values, idx = torch.min(cross_diss_, -1)
        cross_loss += values
    cross_loss = cross_loss / batch_person
    return center_loss, cross_loss, center_loss / cross_loss