from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features, labels):
        """
        input:
            - features: hidden feature representation of shape [b, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, dim = features.size()
        # print(b)
        # print(dim)
        # 得到相似度矩阵
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        # 得到label矩阵，相同label的位置为1
        mask = (torch.ones_like(similarity_matrix) * (labels.expand(b, b).eq(labels.expand(b, b).t()))).cuda()
        # 得到不同类的矩阵，不同类的位置为1
        mask_no_sim = (torch.ones_like(mask) - mask).cuda()
        #这步产生一个对象线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = (torch.ones(b, b) - torch.eye(b, b)).cuda()
        #这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = torch.exp(similarity_matrix/self.temperature)
        #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix*mask_dui_jiao_0
        #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask*similarity_matrix
        #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim
        #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim , dim=1)

        
        # 将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        # 至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        # 每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        
         #no_sim_sum_expend = no_sim_sum.repeat(b, 1).T
        # sim_sum  = sim + no_sim_sum_expend
        loss = torch.div(1 , no_sim_sum)

        
        # 由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        # 全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        
        loss = mask_no_sim + loss + torch.eye(b,b).cuda()

        #接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  #求-log
        loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))  #将所有数据都加起来除以2n

        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # anchor_dot_contrast = torch.div(
        #     torch.cosine_similarity(anchor_feature, contrast_feature),
        #     self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
        
class ConsistencyCos(nn.Module):
    def __init__(self):
        super(ConsistencyCos, self).__init__()
        self.mse_fn = nn.MSELoss()

    def forward(self, feat):
        # feat = nn.functional.normalize(feat, dim=1)

        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        cos = torch.einsum('nc,nc->n', [feat_0, feat_1]).unsqueeze(-1)
        labels = torch.ones((cos.shape[0],1), dtype=torch.float, requires_grad=False)
        if torch.cuda.is_available():
            labels = labels.cuda()
        loss = self.mse_fn(cos, labels)
        return loss

class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, label_smooth=None, class_num=2):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        ''' 
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)	# softmax + log
            target = F.one_hot(target, self.class_num)	# 转换成one-hot
            
            # label smoothing
            # 实现 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num 	
            # 实现 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
            loss = -1*torch.sum(target*logprobs, 1)
        
        else:
            # standard cross entropy loss
            loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))

        return loss.mean()

"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# EPS=1e-8


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        # if not (mask != 0).any():
        #     raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing
        
    def forward(self, pred, target):
        """
        Loss function during self-labeling
        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        
        weak_anchors_prob = self.softmax(pred)

        # print(weak_anchors_prob)
        # print("----------")
        max_prob, _ = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        # print(mask)
        # print("----------")
        # print(target)
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        # print(target_masked)
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = pred

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
        return loss

    # def forward(self, anchors_weak, anchors_strong):
    #     """
    #     Loss function during self-labeling
    #     input: logits for original samples and for its strong augmentations 
    #     output: cross entropy 
    #     """
    #     # Retrieve target and mask based on weakly augmentated anchors
    #     weak_anchors_prob = self.softmax(anchors_weak) 
    #     max_prob, target = torch.max(weak_anchors_prob, dim = 1)
    #     mask = max_prob > self.threshold 
    #     b, c = weak_anchors_prob.size()
    #     target_masked = torch.masked_select(target, mask.squeeze())
    #     n = target_masked.size(0)

    #     # Inputs are strongly augmented anchors
    #     input_ = anchors_strong

    #     # Class balancing weights
    #     if self.apply_class_balancing:
    #         idx, counts = torch.unique(target_masked, return_counts = True)
    #         freq = 1/(counts.float()/n)
    #         weight = torch.ones(c).cuda()
    #         weight[idx] = freq

    #     else:
    #         weight = None
        
    #     # Loss
    #     loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
    #     return loss
