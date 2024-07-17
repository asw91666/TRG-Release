import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, corr_wt=1.0, l1_wt=1.0, uv_wt=1.0, mat_wt=0.1, seg_wt=0.1):
        super(Loss, self).__init__()
        self.threshold = 0.01
        self.SegLoss=nn.CrossEntropyLoss(reduce=False)
        self.UVloss = torch.nn.L1Loss()
        self.corr_wt = corr_wt
        self.l1_wt = l1_wt
        self.uv_wt = uv_wt
        self.seg_wt = seg_wt
        self.mat_wt = mat_wt
        mask = torch.ones(3660)      
        idx = [600,601,602,591,592,593,981,982,983,987,988,989,1044,1045,1046,2343,2344,2345,2289,2290,2291,2286,2287,2288,1938,1939,1940,1947,1948,1949,45,46,47,39,40,41,30,31,32,24,25,26,924,925,926,228,229,230,12,13,14,1575,1576,1577,2229,2230,2231,3303,3304,3305,3288,3289,3290,3276,3277,3278,3579,3580,3581,3588,3589,3590,3318,3319,3320,3504,3505,3506,3234,3235,3236,3222,3223,3224,3207,3208,3209,3483,3484,3485,3495,3496,3497,570,571,572,324,325,326,282,283,284,63,64,65,1629,1630,1631,1671,1672,1673,1917,1918,1919,2139,2140,2141,2118,2119,2120,84,85,86,813,814,815,834,835,836,1218,1219,1220,285,286,287,69,70,71,1632,1633,1634,2508,2509,2510,2124,2125,2126,78,79,80,819,820,821] 
        mask[idx] = 10
        self.register_buffer('mask', torch.FloatTensor(mask).to(device='cuda'))

    def forward(self, assign_mat, seg_pred, ver3d_pred, uv_pred, nocs, model, seg_gt, mat_gt, uv_gt):
        """
        Args:
            assign_mat: bs x n_pts x nv
            deltas: bs x nv x 3
            prior: bs x nv x 3
        """
        inst_shape = ver3d_pred/9.0
        bs,n_pts,_=inst_shape.shape
        # smooth L1 loss for correspondences
        soft_assign = F.softmax(assign_mat, dim=2)
        #mat_loss=mat_wt* torch.mean(torch.sum(torch.abs(soft_assign-mat_gt),dim=2))
        coords = torch.bmm(soft_assign, inst_shape*4.5)  # bs x n_pts x 3
        diff = torch.abs(coords - nocs)
        less = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher = diff - self.threshold / 2.0
        corr_loss = torch.where(diff > self.threshold, higher, less)
        corr_loss = torch.mean(torch.sum(corr_loss, dim=2))
        corr_loss = self.corr_wt * corr_loss
        # kl loss to for  distribution
        log_assign = F.log_softmax(assign_mat, dim=2)
        #kl_loss =self.klLoss(log_assign.view(-1, n_pts),F.softmax(mat_gt, dim=2).view(-1,n_pts))
        #kl_loss =self.klLoss(log_assign,F.softmax(mat_gt, dim=2))
        #print(corr_loss.item())
        if corr_loss.item()<0.01:
            self.mat_wt=0.001
            self.uv_wt=0.5
            self.l1_wt=0.5
        elif corr_loss.item()<0.1:
            self.mat_wt =0.01
            self.seg_wt=0.01
            self.l1_wt =1.0
        kl_loss =F.kl_div(log_assign, mat_gt, size_average = False)
        kl_loss =self.mat_wt * kl_loss/(bs*n_pts)
        # entropy loss to encourage peaked distribution
        entropy_loss = torch.mean(-torch.sum(soft_assign * log_assign, 2))
        entropy_loss = 0.001 * entropy_loss

        uv_loss= self.UVloss(uv_pred, uv_gt)*self.uv_wt
        l1_loss= self.UVloss(ver3d_pred, model*9.0)

        l1_loss = self.l1_wt * l1_loss
        seg_loss = self.SegLoss(seg_pred, seg_gt)

        seg_loss =self.seg_wt * seg_loss.mean()

        # total loss
        total_loss = corr_loss + l1_loss + kl_loss + seg_loss+ entropy_loss
        return total_loss, corr_loss, l1_loss, uv_loss, kl_loss, seg_loss



